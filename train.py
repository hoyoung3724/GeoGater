"""Step 7. Train the GeoGater router and evaluate adaptive routing.

Trains on meta-features of the training set(s) (default: YFCC26K), evaluates
on the test set, and reports Recall@{1,25,200,750,2500} km averaged over 5
seeds, next to the single-path baselines, the backbone and the oracle, plus
the average number of downstream LVLM calls per query.

Examples (reproduce Table 1 from the released features):
    python train.py --work_dir precomputed --backbone g3 --rank_k 15 --zs_k 5 --eval_dataset im2gps3k
    python train.py --work_dir precomputed --backbone georanker --rank_k 12 --zs_k 3 --eval_dataset im2gps3k
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from geogater.paths import Workspace
from geogater.router import GeoGater, GeoLocLoss
from geogater.utils import METRIC_KEYS, recall_stats, set_seed

SEEDS = [42, 101, 7, 99, 12]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split(ws, datasets, database, rank_k, zs_k):
    """Concatenate (meta-features, target dists, binary labels, backbone scores)."""
    X, D, C, S = [], [], [], []
    for ds in [datasets] if isinstance(datasets, str) else datasets:
        d = ws.feature_dir(ds, database, rank_k, zs_k)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Feature directory not found: {d}")
        x = torch.load(os.path.join(d, "meta_feats.pt"))
        if ws.backbone == "g3":
            # G3 does not use reference images / metadata -> mask those features.
            x = x * torch.load(os.path.join(d, "meta_feats_mask.pt"))
        X.append(x)
        D.append(torch.load(os.path.join(d, "target_dists.pt")))
        C.append(torch.load(os.path.join(d, "target_binary.pt")))
        S.append(torch.load(os.path.join(d, "target_sims.pt")))
    return torch.cat(X), torch.cat(D), torch.cat(C), torch.cat(S)


def normalize(train, test, eps=1e-6):
    """Z-normalize meta-features with training-set statistics."""
    x_tr, x_te = train[0].float(), test[0].float()
    mean = x_tr.mean(dim=0, keepdim=True)
    std = x_tr.std(dim=0, keepdim=True)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return ((x_tr - mean) / std, *train[1:]), ((x_te - mean) / std, *test[1:])


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def build_model(args, meta_dim):
    return GeoGater(
        meta_dim=meta_dim,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)


def train_seed(args, seed, data, ckpt_path):
    set_seed(seed)
    X, D, C, _ = data
    dataset = TensorDataset(X, D, C)
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(args.split_seed),
    )
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, X.shape[1])
    criterion = GeoLocLoss(mode=args.loss, tau=args.tau, loss_weights=args.loss_w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, patience = -float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        for m, d, c in train_ldr:
            m, d, c = m.to(args.device), d.to(args.device), c.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(m), dist_targets=d[:, :2], class_targets=c)
            loss.backward()
            optimizer.step()

        # Model selection: geometric-mean error of the routed predictions.
        model.eval()
        errs = []
        with torch.no_grad():
            for m, d, _ in val_ldr:
                probs = torch.softmax(model(m.to(args.device)), dim=1)
                preds = (probs[:, 1] >= 0.5).long()
                d = d.to(args.device)
                errs.extend(d[torch.arange(len(preds)), preds].cpu().numpy())
        score = -(np.exp(np.mean(np.log1p(errs))) - 1)
        if args.verbose:
            print(f"  seed {seed} epoch {epoch + 1:03d} val_geo_err={-score:.2f}")
        if score > best:
            best, patience = score, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                break
    return ckpt_path


@torch.no_grad()
def evaluate(args, ckpt_path, data):
    X, D, C, _ = data
    model = build_model(args, X.shape[1])
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model.eval()
    preds, probs = [], []
    for (m,) in DataLoader(TensorDataset(X), batch_size=256, shuffle=False):
        p = torch.softmax(model(m.to(args.device)), dim=1).cpu().numpy()
        probs.append(p[:, 1])
        preds.append((p[:, 1] >= 0.5).astype(np.int64))
    preds, probs = np.concatenate(preds), np.concatenate(probs)
    D, C = D.numpy(), C.numpy()
    errs = D[np.arange(len(preds)), preds]
    return preds, probs, errs, float(np.mean(preds == C) * 100)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def baselines(D, S, backbone):
    d_zs, d_ret = D[:, 0], D[:, 1]
    # Backbone: the path whose selected candidate has the higher backbone score.
    d_backbone = np.where(S[:, 0] >= S[:, 1], d_zs, d_ret)
    name = "GeoRanker" if backbone == "georanker" else "G3"
    return {
        "Zero-shot": recall_stats(d_zs),
        "Retrieval": recall_stats(d_ret),
        name: recall_stats(d_backbone),
        "Oracle": recall_stats(np.minimum(d_zs, d_ret)),
    }


def lvlm_calls(backbone, frac_zs, rank_k, zs_k):
    """Average downstream LVLM calls / query (Table 2)."""
    if backbone == "g3":
        # N zero-shot samples + 15 RAG generations (5 samples x k in {5, 10, 15}).
        return 20.0, 5.0 + 15.0 * (1.0 - frac_zs)
    # One GeoRanker reward call per candidate on the selected path.
    return float(zs_k + rank_k), zs_k * frac_zs + rank_k * (1.0 - frac_zs)


def print_table(rows):
    header = f"{'Method':<20}" + "".join(f"{k:>9}" for k in METRIC_KEYS)
    print("\n" + header)
    print("-" * len(header))
    for name, s in rows.items():
        print(f"{name:<20}" + "".join(f"{s[k]:>9.2f}" for k in METRIC_KEYS))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True, choices=["g3", "georanker"])
    p.add_argument("--work_dir", default="./outputs")
    p.add_argument("--database", default="mp16")
    p.add_argument("--train_datasets", nargs="+", default=["yfcc26k"])
    p.add_argument("--eval_dataset", default="im2gps3k")
    p.add_argument("--rank_k", type=int, default=15, help="K retrieved candidates.")
    p.add_argument("--zs_k", type=int, default=5, help="N zero-shot hypotheses.")
    p.add_argument("--train_rank_k", type=int, default=None, help="Override K for training.")
    p.add_argument("--train_zs_k", type=int, default=None, help="Override N for training.")
    # Router / optimization (Supp. A)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--split_seed", type=int, default=123, help="Train/val split seed.")
    # Loss: L = lambda * L_dist + L_BCE
    p.add_argument("--loss", nargs="+", default=["ltce", "ce"], help="ltce (L_dist) | ce (L_BCE)")
    p.add_argument("--loss_w", nargs="+", type=float, default=[5.0, 1.0])
    p.add_argument("--tau", type=float, default=1.0, help="Soft-target temperature.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--save_predictions", default=None, help="CSV path for per-query routes.")
    p.add_argument("--confusion", action="store_true", help="Print confusion matrices.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ws = Workspace(args.work_dir, args.backbone)
    tr_k = args.train_rank_k or args.rank_k
    tr_n = args.train_zs_k or args.zs_k
    train = load_split(ws, args.train_datasets, args.database, tr_k, tr_n)
    test = load_split(ws, args.eval_dataset, args.database, args.rank_k, args.zs_k)
    train, test = normalize(train, test)
    print(
        f"Backbone={args.backbone} | train={args.train_datasets} ({len(train[0])}) "
        f"| eval={args.eval_dataset} ({len(test[0])}) | (K, N)=({args.rank_k}, {args.zs_k})"
    )

    os.makedirs(ws.checkpoint_dir, exist_ok=True)
    tag = f"{'_'.join(args.train_datasets)}_{args.database}_rk{tr_k}_zs{tr_n}"
    metrics, accs, all_preds, cms = defaultdict(list), [], [], []
    for seed in tqdm(args.seeds, desc="Seeds"):
        ckpt = os.path.join(ws.checkpoint_dir, f"geogater_{tag}_seed{seed}.pth")
        train_seed(args, seed, train, ckpt)
        preds, probs, errs, acc = evaluate(args, ckpt, test)
        for k, v in recall_stats(errs).items():
            metrics[k].append(v)
        accs.append(acc)
        all_preds.append(preds)
        cms.append(confusion_matrix(test[2].numpy(), preds, labels=[0, 1]))

    D, S = test[1].numpy(), test[3].numpy()
    rows = baselines(D, S, args.backbone)
    rows["GeoGater (ours)"] = {k: float(np.mean(v)) for k, v in metrics.items()}
    print_table(rows)
    print(
        "GeoGater std over seeds: "
        + ", ".join(f"{k}={np.std(metrics[k]):.2f}" for k in METRIC_KEYS[:5])
    )

    frac_zs = float(np.mean(np.stack(all_preds) == 0))
    base_calls, ours_calls = lvlm_calls(args.backbone, frac_zs, args.rank_k, args.zs_k)
    print(f"\nRouting accuracy vs. oracle label: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")
    print(f"Routed to zero-shot path: {100 * frac_zs:.1f}% of queries")
    print(
        f"Downstream LVLM calls / query: backbone {base_calls:.2f} -> GeoGater {ours_calls:.2f} "
        f"({100 * (ours_calls - base_calls) / base_calls:+.2f}%)"
    )
    if args.confusion:
        print("\nMean confusion matrix (rows: oracle ZS/Ret, cols: routed ZS/Ret)")
        print(np.mean(np.stack(cms), axis=0).round(1))

    if args.save_predictions:
        out = pd.DataFrame(
            {f"route_seed{s}": p for s, p in zip(args.seeds, all_preds)}
        )
        out["dist_zs_km"], out["dist_ret_km"] = D[:, 0], D[:, 1]
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        out.to_csv(args.save_predictions, index=False)
        print(f"Saved per-query routes to {args.save_predictions}")


if __name__ == "__main__":
    main()
