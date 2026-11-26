"""Torch builders for the schema-2 catalog (`htr/catalog.py`): one function per block type, called in sequence by
`compile.compile_schema2` to assemble an `HTRArchitecture`-shaped network (same `forward(images, input_lengths) -> (logits,
out_lengths)` contract as the legacy path in `compile.py`). Needs torch."""
import torch
import torch.nn as nn


def _pool_module(pool, h):
    """pool (catalog.py semantics) + the height going into this block -> (nn.Module or None, new_h). None when pool is a no-op [1, 1]."""
    if pool == "collapse": return nn.MaxPool2d((h, 1)), 1
    ph, pw = pool
    if ph == "collapse": return nn.MaxPool2d((h, pw)), 1
    if (ph, pw) == (1, 1): return None, h
    return nn.MaxPool2d((ph, pw)), max(1, h // ph)


def conv2d(b, c_in, h, last):
    out, k = b["out"], b.get("k", 3)
    layers = [nn.Conv2d(c_in, out, k, 1, k // 2)]
    if b.get("bn", True): layers.append(nn.BatchNorm2d(out))
    if b.get("act", "relu") == "relu": layers.append(nn.ReLU())
    pool = b.get("pool", "collapse" if last else [1, 1])
    pm, h2 = _pool_module(pool, h)
    if pm is not None: layers.append(pm)
    return nn.Sequential(*layers), out, h2


def dwsep(b, c_in, h, last):
    out, k = b["out"], b.get("k", 3)
    layers = [nn.Conv2d(c_in, c_in, k, 1, k // 2, groups=c_in), nn.BatchNorm2d(c_in), nn.ReLU(),
              nn.Conv2d(c_in, out, 1), nn.BatchNorm2d(out), nn.ReLU()]
    pool = b.get("pool", "collapse" if last else [1, 1])
    pm, h2 = _pool_module(pool, h)
    if pm is not None: layers.append(pm)
    return nn.Sequential(*layers), out, h2


def maxpool(b, c_in, h, last):
    pm, h2 = _pool_module(b.get("pool", "collapse" if last else [2, 2]), h)
    return (pm or nn.Identity()), c_in, h2


def avgpool(b, c_in, h, last):
    pool = b.get("pool", "collapse" if last else [2, 2])
    if pool == "collapse" or pool[0] == "collapse": return nn.AvgPool2d((h, pool[1] if pool != "collapse" else 1)), c_in, 1
    ph, pw = pool
    return (nn.AvgPool2d((ph, pw)) if (ph, pw) != (1, 1) else nn.Identity()), c_in, max(1, h // ph)


def collapse(b, c_in, h, last):
    return nn.MaxPool2d((h, 1)), c_in, 1


BACKBONE_BUILDERS = {"conv2d": conv2d, "dwsep": dwsep, "maxpool": maxpool, "avgpool": avgpool, "collapse": collapse}


class RNNBlock(nn.Module):
    def __init__(self, b, c_in):
        super().__init__()
        hidden, layers, bidir = b["hidden"], b.get("layers", 1), b.get("bidirectional", True)
        cell = nn.LSTM if b.get("cell", "lstm") == "lstm" else nn.GRU
        self.rnn = cell(c_in, hidden, num_layers=layers, bidirectional=bidir, batch_first=True, dropout=b.get("dropout", 0.0) if layers > 1 else 0.0)
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x, key_padding_mask=None):
        out, _ = self.rnn(x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, b, c_in):
        super().__init__()
        dim, heads, layers, ff_mult, dropout = b["dim"], b.get("heads", 4), b.get("layers", 1), b.get("ff_mult", 4), b.get("dropout", 0.1)
        self.proj = nn.Linear(c_in, dim) if c_in != dim else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_dim = dim

    def forward(self, x, key_padding_mask=None):
        return self.encoder(self.proj(x), src_key_padding_mask=key_padding_mask)


class DropoutBlock(nn.Module):
    def __init__(self, b, c_in):
        super().__init__()
        self.drop = nn.Dropout(b.get("p", 0.3)); self.out_dim = c_in

    def forward(self, x, key_padding_mask=None): return self.drop(x)


SEQUENCE_BUILDERS = {"rnn": RNNBlock, "transformer": TransformerBlock, "dropout": DropoutBlock}
