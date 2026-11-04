"""Choose fine-tune settings on VAL only (3 configs x 3 seeds). TEST is not touched."""
import json, sys
from multiprocessing import Pool
def job(a):
    import torch; from train import CONFIGS, finetune, read_labels, HERE, torch as _t
    _t.set_num_threads(3)
    name, seed = a
    S = json.load(open(HERE / "split.json")); rows = read_labels(HERE / "merged_labels.csv")
    pre = torch.load(HERE / "pretrained.pt")
    tr = [r for r in rows if r["page"] in S["train"]]; va = [r for r in rows if r["page"] in S["val"]]
    _, info = finetune(pre, tr, va, CONFIGS[name], seed=seed, log=lambda *_: None)
    info.pop("hist"); info.update(config=name, seed=seed); print(info, flush=True); return info
if __name__ == "__main__":
    from train import CONFIGS, HERE
    jobs = [(n, s) for n in CONFIGS for s in (0, 1, 2)]
    with Pool(4) as p: res = p.map(job, jobs)
    agg = {}
    for r in res: agg.setdefault(r["config"], []).append(r)
    summ = {n: dict(val_exact=sum(x["val_exact"] for x in v) / len(v), val_cer=sum(x["val_cer"] for x in v) / len(v),
                    epoch_sec=sum(x["mean_epoch_sec"] for x in v) / len(v), best_epochs=[x["best_epoch"] for x in v]) for n, v in agg.items()}
    best = max(summ, key=lambda n: (summ[n]["val_exact"], -summ[n]["val_cer"]))
    json.dump(dict(runs=res, summary=summ, chosen=best), open(HERE / "selection.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), "CHOSEN", best)
