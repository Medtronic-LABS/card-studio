"""config -> nn.Module. Dispatches on shape: a legacy preset name or 4-key spec (conv/context/rnn/dropout) builds through
`compile_legacy` (the direct successor to `work_phaseA/archnet.py build()`, same layers in the same order, so a state dict trained
under the old module loads here unchanged - `htr.legacy.state_dict_matches`); a schema-2 preset name or config (catalog blocks,
`htr.presets`) builds through `compile_schema2`; `{"custom": "<name>"}` loads and dry-run validates `htr/custom/<name>.py` through
`htr.nn.custom.load`.

Either way, `HTRArchitecture.forward(images, input_lengths=None)` returns `(logits [T,B,V], output_lengths [B])`: every block's width
pool factor is fixed at build time (never input-dependent), so `output_lengths` is `input_lengths` divided by the network's fixed
width stride.
"""
from .. import spec as S
from .. import presets as P
from .. import catalog as K
from . import custom as CUSTOM
from . import layers as L


def _is_schema2(config):
    return config in P.PRESETS if isinstance(config, str) else isinstance(config, dict) and config.get("schema") == 2


def _is_custom(config):
    return CUSTOM.is_custom_config(config)


def compile(config, n_classes=129):
    """A preset name or a config (legacy 4-key, schema-2, or {"custom": name}) -> HTRArchitecture. n_classes defaults to the
    shipped pack's class count; pass the vocabulary's len() for a new one."""
    if _is_custom(config): return CUSTOM.load(config["custom"], n_classes)
    return compile_schema2(config, n_classes) if _is_schema2(config) else compile_legacy(config, n_classes)


def compile_schema2(config, n_classes=129):
    import torch
    import torch.nn as nn

    s = P.normalize(config)
    width_stride = K.IMG_W_REF // P.describe(s, n_classes)["time_steps"]

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            backbone, c, h = [], 1, K.IMG_H
            for i, b in enumerate(s["backbone"], 1):
                mod, c, h = L.BACKBONE_BUILDERS[b["type"]](b, c, h, last=(i == len(s["backbone"]))); backbone.append(mod)
            self.backbone = nn.ModuleList(backbone)
            seq = []
            for b in s["sequence"]:
                block = L.SEQUENCE_BUILDERS[b["type"]](b, c); c = block.out_dim; seq.append(block)
            self.sequence = nn.ModuleList(seq)
            self.dropout = nn.Dropout(s["head"]["dropout"]); self.fc = nn.Linear(c, n_classes)

        def forward(self, x, input_lengths=None):
            for mod in self.backbone: x = mod(x)
            f = x.squeeze(2).permute(0, 2, 1)   # [B, T, C]
            out_lengths = None if input_lengths is None else (input_lengths // width_stride).clamp(max=f.shape[1])
            mask = None
            if out_lengths is not None:
                mask = torch.arange(f.shape[1], device=f.device)[None, :] >= out_lengths[:, None]
            for block in self.sequence: f = block(f, key_padding_mask=mask)
            logits = self.fc(self.dropout(f)).permute(1, 0, 2)   # [T, B, V]
            return logits, out_lengths

    net = _Net(); net.spec = s; net.width_stride = width_stride; net.n_classes = n_classes
    return net


def compile_legacy(config, n_classes=129):
    """A legacy spec or preset name -> HTRArchitecture. n_classes defaults to the shipped pack's class count; pass the vocabulary's
    len() for a new one."""
    import torch.nn as nn

    s = S.normalize(config)
    width_stride = S.IMG_W // S.describe(s, n_classes)["time_steps"]

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c, h = [], 1, S.IMG_H
            for b in s["conv"]:
                layers += [nn.Conv2d(c, b["out"], b["k"], 1, b["k"] // 2), nn.BatchNorm2d(b["out"]), nn.ReLU()]; c = b["out"]
                if b["pool"] == "collapse" or isinstance(b["pool"], list) and b["pool"][0] == "collapse":
                    pw = b["pool"][1] if isinstance(b["pool"], list) else 1; layers.append(nn.MaxPool2d((h, pw))); h = 1
                elif tuple(b["pool"]) != (1, 1): layers.append(nn.MaxPool2d(tuple(b["pool"]))); h //= b["pool"][0]
            self.cnn = nn.Sequential(*layers)
            ctx = []
            for b in s["context"]: ctx += [nn.Conv2d(c, b["out"], (1, b["k"]), 1, (0, b["k"] // 2)), nn.BatchNorm2d(b["out"]), nn.ReLU()]; c = b["out"]
            self.ctx = nn.Sequential(*ctx)
            r = s["rnn"]; feat = c
            if r:
                cell = nn.LSTM if r["type"] == "lstm" else nn.GRU
                self.rnn = cell(c, r["hidden"], num_layers=r["layers"], bidirectional=r["bidirectional"], batch_first=True, dropout=r["dropout"] if r["layers"] > 1 else 0.0)
                feat = r["hidden"] * (2 if r["bidirectional"] else 1)
            self.dropout = nn.Dropout(s["dropout"]); self.fc = nn.Linear(feat, n_classes)

        def forward(self, x, input_lengths=None):
            f = self.ctx(self.cnn(x)).squeeze(2).permute(0, 2, 1)         # [B, T, C]
            if hasattr(self, "rnn"): f, _ = self.rnn(f)
            logits = self.fc(self.dropout(f)).permute(1, 0, 2)            # [T, B, V]
            if input_lengths is None: return logits, None
            out_lengths = (input_lengths // width_stride).clamp(max=logits.shape[0])
            return logits, out_lengths

    net = _Net(); net.spec = s; net.width_stride = width_stride; net.n_classes = n_classes
    return net


HTRArchitecture = compile       # alias: the plan's common interface is this function's result, not a separate base class
