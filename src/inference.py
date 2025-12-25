"""Inference script for finetuned TCN+Transformer model.

Loads finetuned weights, runs on a slice of slot features (.npy),
returns color prediction, probabilities, and attention-based highlights.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch

try:
	import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
	plt = None

from finetune import FineTuneNarrativeModel, load_backbone


COLOR_MAP = {0: "Green", 1: "Blue", 2: "Red", 3: "Purple"}


def load_model(finetune_ckpt: str, tcn_ckpt: str, device: torch.device) -> FineTuneNarrativeModel:
	backbone = load_backbone(tcn_ckpt, device).to(device)
	model = FineTuneNarrativeModel(backbone).to(device)
	ckpt = torch.load(finetune_ckpt, map_location=device)
	model.load_state_dict(ckpt["model_state"], strict=False)
	model.eval()
	return model


def infer_slice(
	model: FineTuneNarrativeModel,
	npy_path: str,
	start_time: int,
	end_time: int,
	slot_seconds: int = 5,
	device: torch.device = torch.device("cpu"),
) -> Tuple[int, torch.Tensor, torch.Tensor | None]:
	arr = np.load(npy_path)
	start_idx = start_time // slot_seconds
	end_idx = end_time // slot_seconds if end_time is not None else arr.shape[0]
	seq = arr[start_idx:end_idx]
	if seq.size == 0:
		raise ValueError("Empty slice; check start/end times")

	x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)  # (1, L, 129)
	pad_mask = torch.zeros((1, x.size(1)), dtype=torch.bool, device=device)

	with torch.no_grad():
		logits, _, attn = model(x, pad_mask, return_attn=True)
		probs = torch.softmax(logits, dim=-1)
		pred = int(torch.argmax(probs, dim=-1).item())

	return pred, probs.cpu(), attn


def summarize_attn(attn: torch.Tensor) -> np.ndarray:
	"""Reduce attention weights to per-time-step importance.

	Expects attn shape (batch, heads, L, L). Returns (L,) importance.
	"""
	if attn is None:
		return None
	# attn may be (B, heads, L, L) or (B, L, L) if averaged
	if attn.dim() == 4:
		imp = attn.mean(dim=1).mean(dim=1)  # (B, L)
	elif attn.dim() == 3:
		imp = attn.mean(dim=1)  # (B, L)
	else:
		return None
	imp = imp.squeeze(0)
	return imp.detach().cpu().numpy()


def plot_importance(importance: np.ndarray, output_path: str) -> None:
	if plt is None:
		print("matplotlib not installed; skipping plot")
		return
	plt.figure(figsize=(12, 4))
	plt.plot(importance, label="Attention importance", color="orange")
	plt.fill_between(range(len(importance)), importance, color="orange", alpha=0.3)
	plt.xlabel("Time slots (5s)")
	plt.ylabel("Importance")
	plt.title("Run highlights")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path)
	print(f"Saved attention plot to {output_path}")


def main():
	parser = argparse.ArgumentParser(description="Inference for finetuned TCN+Transformer")
	parser.add_argument("--npy_path", required=True, help="Path to slot .npy file")
	parser.add_argument("--finetune_ckpt", default="../models/tcn/finetune.pt", help="Finetuned checkpoint")
	parser.add_argument("--tcn_ckpt", default="../models/tcn/next_vector_tcn.pt", help="Pretrained TCN checkpoint")
	parser.add_argument("--start_time", type=int, default=0, help="Start time in seconds")
	parser.add_argument("--end_time", type=int, default=None, help="End time in seconds")
	parser.add_argument("--slot_seconds", type=int, default=5, help="Seconds per slot")
	parser.add_argument("--plot", default=None, help="Output path for attention plot (optional)")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_model(args.finetune_ckpt, args.tcn_ckpt, device)

	pred, probs, attn = infer_slice(
		model,
		args.npy_path,
		args.start_time,
		args.end_time,
		slot_seconds=args.slot_seconds,
		device=device,
	)

	print(f"Predicted color: {COLOR_MAP.get(pred, pred)}")
	print(f"Probabilities: {probs.squeeze().tolist()}")

	importance = summarize_attn(attn)
	if importance is not None and args.plot:
		plot_importance(importance, args.plot)


if __name__ == "__main__":
	main()

