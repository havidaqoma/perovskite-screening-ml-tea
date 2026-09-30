"""Stage 12 (WSL/GPU): Roost baseline on the SAME persisted folds as s11.

Run in the WSL GPU env (env/setup_wsl_gpu.sh):
    ~/venvs/mlperov-gpu/bin/python -m pipeline.s12_roost runs/v2 --task gap_semi --split chemsys
Roost (Goodall & Lee, Nat. Commun. 2020, 10.1038/s41467-020-19964-7) via the `aviary` package.
Writes runs/v2/metrics/roost_<split>_<task>.json and predictions/roost_<split>_<task>.csv.
Imports only numpy/pandas/torch/aviary so it does not need the Windows main env.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def run(run_dir: Path, task: str, split: str, epochs: int, folds: list[int] | None):
    import torch
    from aviary.roost.data import CompositionData, collate_batch
    from aviary.roost.model import Roost
    from torch.utils.data import DataLoader

    df = pd.read_csv(run_dir / "data/train_v2.csv")
    sp = json.loads((run_dir / "splits/v2_splits.json").read_text())[split]
    target = "formation_energy_per_atom" if task == "ef" else "band_gap"
    mask = (df.is_semi == 1).to_numpy() if task == "gap_semi" else np.ones(len(df), bool)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    preds = np.full(len(df), np.nan)
    fold_res = []
    for fi, f in enumerate(sp):
        if folds and fi not in folds:
            continue
        tr = np.array(f["train"]); te = np.array(f["test"])
        tr, te = tr[mask[tr]], te[mask[te]]
        rng = np.random.default_rng(42)
        perm = rng.permutation(tr)
        nv = max(500, int(0.1 * len(perm)))
        va, trn = perm[:nv], perm[nv:]

        def mk(idx):
            sub = df.iloc[idx][["row", "formula", target]].rename(columns={"row": "material_id", "formula": "composition"})
            return CompositionData(sub, task_dict={target: "regression"}, inputs="composition", identifiers=("material_id", "composition"))

        dl = lambda idx, sh: DataLoader(mk(idx), batch_size=256, shuffle=sh, collate_fn=collate_batch)
        model = Roost(robust=True, n_targets=[1], elem_embedding="matscholar200",
                      task_dict={target: "regression"}, device=dev).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
        t0 = time.time()
        best, best_state = np.inf, None
        y_mean, y_std = float(df[target].iloc[trn].mean()), float(df[target].iloc[trn].std())

        def evaluate(idx):
            model.eval(); out = []
            with torch.no_grad():
                for inputs, targets, *_ in dl(idx, False):  # ids unused
                    inputs = [x.to(dev) if hasattr(x, "to") else x for x in inputs]
                    out.append(model(*inputs)[0][:, 0].cpu().numpy() * y_std + y_mean)
            return np.concatenate(out)

        for ep in range(epochs):
            model.train()
            for inputs, targets, *_ in dl(trn, True):
                inputs = [x.to(dev) if hasattr(x, "to") else x for x in inputs]
                y = (targets[0].to(dev).float().view(-1) - y_mean) / y_std
                out = model(*inputs)[0]
                mu, logstd = out[:, 0], out[:, 1]
                loss = (np.sqrt(2.0) * torch.abs(mu - y) * torch.exp(-logstd) + logstd).mean()   # robust L1
                opt.zero_grad(); loss.backward(); opt.step()
            v = float(np.mean(np.abs(evaluate(va) - df[target].to_numpy()[va])))
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
            print(f"fold {fi} epoch {ep} val_mae {v:.4f} best {best:.4f} {time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best_state)
        p = evaluate(te)
        preds[te] = p
        fold_res.append({"fold": fi, "n_test": int(len(te)), "mae": float(np.mean(np.abs(p - df[target].to_numpy()[te]))),
                         "val_mae": best, "seconds": round(time.time() - t0, 1)})
        print(fold_res[-1], flush=True)
    ev = np.isfinite(preds)
    res = {"split": split, "task": task, "model": "roost", "epochs": epochs, "device": dev,
           "mae_pooled": float(np.mean(np.abs(preds[ev] - df[target].to_numpy()[ev]))), "folds": fold_res}
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / f"metrics/roost_{split}_{task}.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"row": df.row[ev], "pred": preds[ev]}).to_csv(run_dir / f"predictions/roost_{split}_{task}.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--task", default="gap_semi", choices=["gap_semi", "gap_all", "ef"])
    ap.add_argument("--split", default="chemsys")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, nargs="*")
    a = ap.parse_args()
    run(a.run_dir, a.task, a.split, a.epochs, a.folds)
