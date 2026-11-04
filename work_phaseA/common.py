"""Shared pieces for Phase A: vocabulary/masks from model_meta.json, CRNN (same as train_crnn.ipynb),
image loading, decoding with the app's confidence, metrics. Read-only on ../work and ../model_meta.json."""
from pathlib import Path
import csv, json, math, random, unicodedata
import numpy as np, cv2, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work"
CROPS = WORK / "crops"
IMG_H, IMG_W = 48, 192

META = json.load(open(ROOT / "model_meta.json"))
CLASSES = META["classes"]
CHAR2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
BLANK_IDX = META["blank_idx"]
FIELD_ALLOWED = {k: sorted(v["allowed_idx"]) for k, v in META["fields"].items()}
FIELD_TYPE = {k: v["type"] for k, v in META["fields"].items()}
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize_label(s):
    return unicodedata.normalize("NFC", s.strip()).translate(BN2EN)


def encode_label(s):
    return [CHAR2IDX[c] for c in normalize_label(s)]


_masks = {}
def mask_np(field):
    if field not in _masks:
        m = np.zeros(len(CLASSES), bool)
        m[FIELD_ALLOWED[field]] = True
        _masks[field] = m
    return _masks[field]


def mask_t(field):
    return torch.from_numpy(mask_np(field))


def read_labels(path):
    rows = [r for r in csv.DictReader(open(path)) if r["label"].strip() != ""]
    for r in rows:
        r["label"] = normalize_label(r["label"])
    return rows


def load_image(path):  # identical to notebook load_image
    im = cv2.imread(str(path), 0)
    h, w = im.shape
    im = cv2.resize(im, (min(IMG_W, max(1, int(w * IMG_H / h))), IMG_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((IMG_H, IMG_W), 255, np.uint8)
    canvas[:, : im.shape[1]] = im
    return canvas


_cache = {}
def cell_image(r):
    k = (r["page"], r["row"], r["col"])
    if k not in _cache:
        _cache[k] = load_image(Path(r["crop"]) if r.get("crop") else CROPS / r["row"] / r["col"] / f"{r['page']}.png")
    return _cache[k]


def to_input(im):
    return (im.astype(np.float32) / 255.0 - 0.5) / 0.5


class CRNN(nn.Module):  # identical to notebook (1 BiLSTM layer, hidden 128)
    def __init__(self, n_classes, hidden=128, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((IMG_H // 8, 1)),
        )
        self.rnn = nn.LSTM(128, hidden, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        seq, _ = self.rnn(f)
        return self.fc(self.dropout(seq)).permute(1, 0, 2)  # (T,B,V)


def decode_app(logits_tv, field):
    """Port of AncReader.read: masked greedy CTC; confidence = min over emitted chars of
    masked-softmax prob at the emitting frame. logits_tv: (T,V) numpy."""
    m = mask_np(field)
    z = np.where(m[None, :], logits_tv, -np.inf)
    best = z.argmax(1)
    out, confs, prev = [], [], -1
    for t, b in enumerate(best):
        if b != BLANK_IDX and b != prev:
            p = 1.0 / np.exp(z[t][m] - z[t, b]).sum()
            out.append(IDX2CHAR[int(b)]); confs.append(float(p))
        prev = b
    return "".join(out), (min(confs) if confs else 0.0)


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[len(b)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@torch.no_grad()
def predict_torch(model, rows, device="cpu"):
    """-> list of (pred, conf) using app decoding."""
    model.eval()
    if not rows:
        return []
    x = torch.from_numpy(np.stack([to_input(cell_image(r)) for r in rows])).unsqueeze(1).to(device)
    lg = model(x).cpu().numpy()  # (T,B,V)
    return [decode_app(lg[:, i, :], r["row"]) for i, r in enumerate(rows)]
