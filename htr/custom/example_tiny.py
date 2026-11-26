"""A minimal worked example of a custom architecture (also used as a fixture by studio/tests/test_htr_custom.py): two convolutions
collapsing height to 1, a linear head, no recurrent layer at all - about as small as a reader can be and still follow the
HTRArchitecture contract (`forward(images, input_lengths=None) -> (logits, output_lengths)`). Copy this file's shape, not its
exact sizes, to start your own.

Height collapses with a plain `MaxPool2d((h, 1))` at a height known at construction time (48, the only height this lab ever
reads at, so `h` after two 2x2 pools is a constant 12), not `AdaptiveMaxPool2d((1, None))`: the legacy TorchScript ONNX exporter
(htr.export.onnx_backend, and htr.nn.check's dry run) cannot trace a dynamic adaptive-pool output size, so a real architecture
needs to collapse height the same fixed way `htr.nn.layers.collapse` does."""
import torch
import torch.nn as nn


class _Tiny(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.width_stride = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((12, 1)),   # 48 -> 24 -> 12 -> 1, width untouched
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, images, input_lengths=None):
        f = self.cnn(images).squeeze(2).permute(0, 2, 1)   # [B, T, C]
        logits = self.fc(f).permute(1, 0, 2)                # [T, B, V]
        if input_lengths is None: return logits, None
        return logits, (input_lengths // self.width_stride).clamp(max=logits.shape[0])


def build(n_classes):
    return _Tiny(n_classes)
