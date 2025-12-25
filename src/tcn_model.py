"""
Temporal Convolutional Network (TCN) for next-slot vector prediction.

Task:
	- Input: past N slots (default 12), each a 129-dim feature (128-dim semantic + 1 density).
	- Output: predict the next slot (N+1) 129-dim vector.
	- Loss: SmoothL1Loss (Huber) by default.

Key constraints:
	- No sliding across files; windows are contained within each .npy file.
	- Causal convolutions only look at past context.
	- Vectors are assumed already normalized (L2 for semantic, log+Min-Max for density).
"""

import argparse
import glob
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------
# Dataset
# ---------------------

class SlotDataset(Dataset):
	"""Dataset over slot feature sequences without crossing file boundaries."""

	def __init__(self, slot_dir: str, window_size: int = 12):
		self.window_size = window_size
		self.arrays: List[np.ndarray] = []
		self.index_map: List[Tuple[int, int]] = []  # (file_idx, start_idx)

		npy_files = sorted(glob.glob(os.path.join(slot_dir, "*.npy")))
		if not npy_files:
			raise FileNotFoundError(f"No .npy files found in {slot_dir}")

		for file_idx, path in enumerate(npy_files):
			arr = np.load(path)
			if arr.ndim != 2 or arr.shape[1] != 129:
				continue  # skip malformed files
			num_slots = arr.shape[0]
			if num_slots <= window_size:
				continue
			self.arrays.append(arr.astype(np.float32))
			for start in range(num_slots - window_size):
				# start window [start, start+window_size), target at start+window_size
				self.index_map.append((file_idx, start))

		if not self.index_map:
			raise RuntimeError("No valid training windows found. Check slot files and window size.")

	def __len__(self) -> int:
		return len(self.index_map)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		file_idx, start = self.index_map[idx]
		arr = self.arrays[file_idx]
		x = arr[start : start + self.window_size]          # (N, 129)
		y = arr[start + self.window_size]                  # (129,)
		x_t = torch.from_numpy(x)                          # float32
		y_t = torch.from_numpy(y)
		return x_t, y_t


# ---------------------
# TCN components
# ---------------------

class CausalConv1d(nn.Module):
	"""1D causal convolution with manual right-trim to enforce causality."""

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
		super().__init__()
		padding = (kernel_size - 1) * dilation
		self.conv = nn.Conv1d(
			in_channels,
			out_channels,
			kernel_size,
			padding=padding,
			dilation=dilation,
		)
		self.trim = padding

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.conv(x)
		if self.trim > 0:
			out = out[:, :, :-self.trim]
		return out


class TemporalBlock(nn.Module):
	"""A single TCN residual block."""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		dilation: int,
		dropout: float = 0.1,
	):
		super().__init__()
		self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
		self.gn1 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
		self.relu1 = nn.ReLU()
		self.drop1 = nn.Dropout(dropout)

		self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
		self.gn2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
		self.relu2 = nn.ReLU()
		self.drop2 = nn.Dropout(dropout)

		self.downsample = (
			nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
		)
		self.final_relu = nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.conv1(x)
		out = self.gn1(out)
		out = self.relu1(out)
		out = self.drop1(out)

		out = self.conv2(out)
		out = self.gn2(out)
		out = self.relu2(out)
		out = self.drop2(out)

		res = x if self.downsample is None else self.downsample(x)
		return self.final_relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(
		self,
		num_inputs: int,
		channels: Sequence[int],
		kernel_size: int = 3,
		dilations: Sequence[int] = (1, 2, 4),
		dropout: float = 0.1,
	):
		super().__init__()
		layers: List[nn.Module] = []
		in_ch = num_inputs
		for out_ch, dilation in zip(channels, dilations):
			layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
			in_ch = out_ch
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


class NextVectorTCN(nn.Module):
	"""TCN backbone with a prediction head for next-slot vector."""

	def __init__(
		self,
		input_dim: int = 129,
		tcn_channels: Sequence[int] = (128, 128, 128),
		kernel_size: int = 3,
		dilations: Sequence[int] = (1, 2, 4),
		dropout: float = 0.1,
	):
		super().__init__()
		self.tcn = TemporalConvNet(
			num_inputs=input_dim,
			channels=tcn_channels,
			kernel_size=kernel_size,
			dilations=dilations,
			dropout=dropout,
		)
		self.head = nn.Linear(tcn_channels[-1], input_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, seq_len, input_dim) -> reshape to (batch, input_dim, seq_len)
		x = x.transpose(1, 2)
		h = self.tcn(x)                     # (batch, C, seq_len)
		h_last = h[:, :, -1]                # use last time step (causal)
		out = self.head(h_last)             # (batch, input_dim)
		return out


# ---------------------
# Training utilities
# ---------------------

def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def split_indices(num_samples: int, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
	idx = list(range(num_samples))
	random.Random(seed).shuffle(idx)
	val_size = int(num_samples * val_ratio)
	val_idx = idx[:val_size]
	train_idx = idx[val_size:]
	return train_idx, val_idx


def collate_batch(batch):
	xs, ys = zip(*batch)
	x = torch.stack(xs, dim=0)  # (B, N, 129)
	y = torch.stack(ys, dim=0)  # (B, 129)
	return x, y


def train(
	slot_dir: str,
	output_dir: str,
	window_size: int = 24,
	channels: Sequence[int] = (128, 128, 128),
	kernel_size: int = 3,
	dilations: Sequence[int] = (1, 4, 16),
	dropout: float = 0.1,
	batch_size: int = 64,
	epochs: int = 50,
	lr: float = 1e-3,
	val_ratio: float = 0.1,
	use_cosine: bool = False,
	cosine_alpha: float = 0.5,
	device: str = None,
) -> None:
	set_seed(42)

	dataset = SlotDataset(slot_dir, window_size)
	train_idx, val_idx = split_indices(len(dataset), val_ratio=val_ratio)
	train_subset = torch.utils.data.Subset(dataset, train_idx)
	val_subset = torch.utils.data.Subset(dataset, val_idx)

	train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
	val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

	device = device or ("cuda" if torch.cuda.is_available() else "cpu")

	model = NextVectorTCN(
		input_dim=129,
		tcn_channels=channels,
		kernel_size=kernel_size,
		dilations=dilations,
		dropout=dropout,
	).to(device)

	criterion = nn.SmoothL1Loss()  # Huber loss
	cosine = nn.CosineSimilarity(dim=1) if use_cosine else None
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.2)

	os.makedirs(output_dir, exist_ok=True)
	best_val = float("inf")
	best_path = os.path.join(output_dir, "next_vector_tcn.pt")

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)

			pred = model(xb)
			loss = criterion(pred, yb)
			if cosine is not None:
				cos_loss = 1 - cosine(pred[:, :128], yb[:, :128]).mean()
				loss = (1 - cosine_alpha) * loss + cosine_alpha * cos_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item() * xb.size(0)

		train_loss = total_loss / len(train_subset)

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				pred = model(xb)
				loss = criterion(pred, yb)
				if cosine is not None:
					cos_loss = 1 - cosine(pred[:, :128], yb[:, :128]).mean()
					loss = (1 - cosine_alpha) * loss + cosine_alpha * cos_loss
				val_loss += loss.item() * xb.size(0)

		val_loss /= max(1, len(val_subset))

		scheduler.step(val_loss)
		current_lr = optimizer.param_groups[0]["lr"]
		print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.6f}")

		if val_loss < best_val:
			best_val = val_loss
			torch.save({
				"model_state": model.state_dict(),
				"config": {
					"window_size": window_size,
					"channels": channels,
					"kernel_size": kernel_size,
					"dilations": dilations,
					"dropout": dropout,
				},
			}, best_path)
			print(f"Saved best model to {best_path} (val_loss={best_val:.4f})")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train TCN for next-slot vector prediction")
	parser.add_argument("--slot_dir", default="../data/slots/", help="Directory containing slot .npy files")
	parser.add_argument("--output_dir", default="../models/tcn/", help="Where to save the trained model")
	parser.add_argument("--window_size", type=int, default=12, help="Number of past slots to observe")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--use_cosine", action="store_true", help="Add cosine similarity term for semantic dims")
	parser.add_argument("--cosine_alpha", type=float, default=0.3, help="Weight for cosine term (0-1)")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(
		slot_dir=args.slot_dir,
		output_dir=args.output_dir,
		window_size=args.window_size,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		use_cosine=args.use_cosine,
		cosine_alpha=args.cosine_alpha,
	)

