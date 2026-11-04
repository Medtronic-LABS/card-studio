from train import *
t = time.time(); st = pretrain(); torch.save(st, HERE / "pretrained.pt"); print("pretrain total sec", time.time() - t)
