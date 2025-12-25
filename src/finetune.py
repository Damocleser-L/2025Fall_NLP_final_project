"""Fine-tune TCN+Transformer with multi-task losses (color classification + summary alignment).

Core ideas:
- Slice slot sequences per run by start/end time from CSV; never cross files.
- Encode long summaries via Sentence-BERT for semantic alignment.
- Use pretrained TCN backbone, add Transformer encoder, dual heads (cls + projector).
- Multi-task loss: CE for color, MSE for summary embedding (weighted).
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from tcn_model import NextVectorTCN

try:
	from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
	SentenceTransformer = None


# ---------------------
# Dataset
# ---------------------


class RunNarrativeDataset(Dataset):
	"""Slice slot sequences by [start_time, end_time] and attach labels + summary embeddings.

	CSV requirements:
		- Columns: start_time, end_time, color, summary (run_number optional).
		- No file column needed; the slot file is inferred from the CSV file name unless provided.
	"""

	def __init__(
		self,
		csv_path: str,
		npy_dir: str,
		slot_seconds: int = 5,
		sbert_model: str = "all-MiniLM-L6-v2",
		slot_file: str = None,
		use_summary: bool = True,
	) -> None:
		self.df = pd.read_csv(csv_path)
		self.npy_cache: Dict[str, np.ndarray] = {}
		self.npy_dir = npy_dir
		self.slot_seconds = slot_seconds
		self.color_map = {"Green": 0, "Blue": 1, "Red": 2, "Purple": 3}
		self.use_summary = use_summary

		# Infer slot file name from CSV if not provided: <basename>_slots.npy
		if slot_file is None:
			base = os.path.splitext(os.path.basename(csv_path))[0]
			self.slot_file = f"{base}_slots.npy"
		else:
			self.slot_file = slot_file

		self.summary_dim = 384
		if self.use_summary:
			if SentenceTransformer is None:
				raise ImportError("sentence-transformers not installed. pip install sentence-transformers")
			self.text_encoder = SentenceTransformer(sbert_model)
			self.summary_embs = self.text_encoder.encode(self.df["summary"].tolist(), convert_to_numpy=True)
		else:
			self.summary_embs = None

	def __len__(self) -> int:
		return len(self.df)

	def _load_npy(self, file_name: str) -> np.ndarray:
		if file_name not in self.npy_cache:
			path = os.path.join(self.npy_dir, file_name)
			self.npy_cache[file_name] = np.load(path)
		return self.npy_cache[file_name]

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]

		# Use inferred slot file (one CSV corresponds to one run/slots file)
		arr = self._load_npy(self.slot_file)
		start_idx = int(row["start_time"] // self.slot_seconds)
		end_idx = int(row["end_time"] // self.slot_seconds)
		seq = arr[start_idx:end_idx]  # (L, 129)

		label = self.color_map[row["color"]]
		if self.summary_embs is not None:
			summary_vec = self.summary_embs[idx]
		else:
			summary_vec = np.zeros(self.summary_dim, dtype=np.float32)

		return (
			torch.from_numpy(seq.astype(np.float32)),
			torch.tensor(label, dtype=torch.long),
			torch.from_numpy(summary_vec.astype(np.float32)),
		)


def collate_batch(batch):
	sequences, labels, summaries = zip(*batch)
	lengths = [seq.size(0) for seq in sequences]
	padded = pad_sequence(sequences, batch_first=True)  # (B, L_max, 129)
	# mask: True for pad positions
	max_len = padded.size(1)
	mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
	for i, l in enumerate(lengths):
		if l < max_len:
			mask[i, l:] = True
	labels_t = torch.stack(labels)
	summaries_t = torch.stack(summaries)
	return padded, labels_t, summaries_t, mask, torch.tensor(lengths, dtype=torch.long)


# ---------------------
# Model
# ---------------------


class FineTuneNarrativeModel(nn.Module):
	def __init__(
		self,
		backbone: NextVectorTCN,
		nhead: int = 4,
		num_layers: int = 2,
		num_classes: int = 4,
		summary_dim: int = 384,
	):
		super().__init__()
		self.backbone = backbone.tcn  # use pretrained TCN body

		# infer backbone output channels (last block out_channels)
		backbone_out = self.backbone.network[-1].conv2.conv.out_channels

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=backbone_out,
			nhead=nhead,
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(
			encoder_layer,
			num_layers=num_layers,
			enable_nested_tensor=False,  # ensure attn_weights are returned for hooks/analysis
		)
		self.last_attn = None

		self.classifier = nn.Linear(backbone_out, num_classes)
		self.projector = nn.Linear(backbone_out, summary_dim)

	def forward(
		self,
		x: torch.Tensor,
		padding_mask: torch.Tensor,
		return_attn: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
		# x: (B, L, 129); mask: (B, L) with True on pads
		self.last_attn = None
		x = x.transpose(1, 2)
		h = self.backbone(x)              # (B, C, L)
		h = h.transpose(1, 2)             # (B, L, C)


		h_enc = h
		last_attn = None
		for i, layer in enumerate(self.transformer.layers):
			# self-attention with weights
			attn_out, attn_weights = self._sa_block_with_attn(layer, h_enc, None, padding_mask)
			if layer.norm_first:
				h_enc = h_enc + attn_out
				h_enc = h_enc + self._ff_block(layer, layer.norm2(h_enc))
			else:
				h_enc = layer.norm1(h_enc + attn_out)
				h_enc = layer.norm2(h_enc + self._ff_block(layer, h_enc))
			if i == len(self.transformer.layers) - 1:
				last_attn = attn_weights
		self.last_attn = last_attn

		valid_counts = (~padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
		h_sum = (h_enc * (~padding_mask).unsqueeze(-1)).sum(dim=1)
		global_feat = h_sum / valid_counts

		color_logits = self.classifier(global_feat)
		summary_proj = self.projector(global_feat)
		attn = self.last_attn if return_attn else None
		return color_logits, summary_proj, attn

	@staticmethod
	def _sa_block_with_attn(layer: nn.TransformerEncoderLayer, x: torch.Tensor, attn_mask, key_padding_mask):
		attn_output, attn_weights = layer.self_attn(
			x,
			x,
			x,
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			need_weights=True,
			average_attn_weights=False,
		)
		attn_output = layer.dropout1(attn_output)
		return attn_output, attn_weights

	@staticmethod
	def _ff_block(layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> torch.Tensor:
		x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
		x = layer.dropout2(x)
		return x


# ---------------------
# Training
# ---------------------


def load_backbone(ckpt_path: str, device: torch.device) -> NextVectorTCN:
	ckpt = torch.load(ckpt_path, map_location=device)
	cfg = ckpt.get("config", {})
	model = NextVectorTCN(
		input_dim=129,
		tcn_channels=cfg.get("channels", (128, 128, 128)),
		kernel_size=cfg.get("kernel_size", 3),
		dilations=cfg.get("dilations", (1, 2, 4)),
		dropout=cfg.get("dropout", 0.1),
	)
	model.load_state_dict(ckpt["model_state"], strict=False)
	return model


def split_indices(num_samples: int, val_ratio: float = 0.1) -> Tuple[List[int], List[int]]:
	idx = list(range(num_samples))
	split = int(num_samples * (1 - val_ratio))
	return idx[:split], idx[split:]


def train(
	csv_path: str,
	npy_dir: str,
	ckpt_path: str,
	output_path: str,
	batch_size: int = 4,
	epochs: int = 10,
	lr_backbone: float = 1e-5,
	lr_head: float = 1e-4,
	val_ratio: float = 0.1,
	summary_weight: float = 0.0,
	slot_seconds: int = 5,
	device: str = None,
) -> None:
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

	use_summary = summary_weight > 0
	dataset = RunNarrativeDataset(csv_path, npy_dir, slot_seconds=slot_seconds, use_summary=use_summary)
	train_idx, val_idx = split_indices(len(dataset), val_ratio=val_ratio)
	train_ds = Subset(dataset, train_idx)
	val_ds = Subset(dataset, val_idx)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

	backbone = load_backbone(ckpt_path, device).to(device)
	model = FineTuneNarrativeModel(backbone).to(device)

	# different lrs for backbone vs heads
	params = [
		{"params": model.backbone.parameters(), "lr": lr_backbone},
		{"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")], "lr": lr_head},
	]
	optimizer = torch.optim.Adam(params)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

	criterion_cls = nn.CrossEntropyLoss()
	criterion_sim = nn.MSELoss()

	best_val = float("inf")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	for epoch in range(1, epochs + 1):
		model.train()
		train_loss = 0.0
		for seqs, labels, sums, mask, _ in train_loader:
			seqs = seqs.to(device)
			labels = labels.to(device)
			sums = sums.to(device)
			mask = mask.to(device)

			logits, proj, _ = model(seqs, mask)
			loss_cls = criterion_cls(logits, labels)
			if use_summary and summary_weight > 0:
				loss_sim = criterion_sim(proj, sums)
				loss = loss_cls + summary_weight * loss_sim
			else:
				loss = loss_cls

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item() * seqs.size(0)

		train_loss /= max(1, len(train_ds))

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for seqs, labels, sums, mask, _ in val_loader:
				seqs = seqs.to(device)
				labels = labels.to(device)
				sums = sums.to(device)
				mask = mask.to(device)

				logits, proj, _ = model(seqs, mask)
				loss_cls = criterion_cls(logits, labels)
				if use_summary and summary_weight > 0:
					loss_sim = criterion_sim(proj, sums)
					loss = loss_cls + summary_weight * loss_sim
				else:
					loss = loss_cls
				val_loss += loss.item() * seqs.size(0)

		val_loss /= max(1, len(val_ds))
		scheduler.step(val_loss)
		current_lr = optimizer.param_groups[0]["lr"]
		print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.6f}")

		if val_loss < best_val:
			best_val = val_loss
			torch.save({
				"model_state": model.state_dict(),
				"config": {
					"summary_weight": summary_weight,
					"lr_backbone": lr_backbone,
					"lr_head": lr_head,
				},
			}, output_path)
			print(f"Saved best model to {output_path} (val_loss={best_val:.4f})")


def parse_args():
	parser = argparse.ArgumentParser(description="Fine-tune TCN+Transformer with summary alignment")
	parser.add_argument("--csv_path", default="../data/labels/train.csv", help="CSV with file, start_time, end_time, color, summary")
	parser.add_argument("--npy_dir", default="../data/slots/", help="Directory with slot .npy files")
	parser.add_argument("--ckpt_path", default="../models/tcn/next_vector_tcn.pt", help="Pretrained TCN checkpoint")
	parser.add_argument("--output_path", default="../models/tcn/finetune.pt", help="Where to save finetuned model")
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr_backbone", type=float, default=1e-5)
	parser.add_argument("--lr_head", type=float, default=1e-4)
	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--summary_weight", type=float, default=0.0)
	parser.add_argument("--slot_seconds", type=int, default=5)
	parser.add_argument("--device", default=None, help="cuda or cpu")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(
		csv_path=args.csv_path,
		npy_dir=args.npy_dir,
		ckpt_path=args.ckpt_path,
		output_path=args.output_path,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr_backbone=args.lr_backbone,
		lr_head=args.lr_head,
		val_ratio=args.val_ratio,
		summary_weight=args.summary_weight,
		slot_seconds=args.slot_seconds,
		device=args.device,
	)

