"""results/preds altındaki tahminlerden mid-project tablosunu üretir (Markdown).

Her model için MASSIVE tr-TR dev/test macro-F1 ve accuracy; DIET için seed ortalaması
± std. Ayrıca test'te Config A - TF-IDF+LR eşleştirilmiş bootstrap farkı.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import bootstrap_ci, paired_bootstrap, scores

PREDS = Path(__file__).parent / "results" / "preds"
TUNE = Path(__file__).parent / "results" / "tune"


def read(name, split):
    base = TUNE if name.startswith("final_") else PREDS
    return pd.read_csv(base / f"{name}_{split}.csv", keep_default_na=False)


def row(label, split, names):
    ms, accs = [], []
    for n in names:
        d = read(n, split)
        s = scores(list(d.y_true), list(d.y_pred))
        ms.append(s["macro_f1"] * 100)
        accs.append(s["accuracy"] * 100)
    if len(names) == 1:
        d = read(names[0], split)
        lo, hi = bootstrap_ci(list(d.y_true), list(d.y_pred))
        return f"| {label} | {split} | {ms[0]:.2f} (95% CI {lo*100:.1f}-{hi*100:.1f}) | {accs[0]:.2f} |"
    return (f"| {label} ({len(names)} seed) | {split} | {np.mean(ms):.2f} ± {np.std(ms, ddof=1):.2f} "
            f"| {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f} |")


def main():
    a_names = sorted({p.name.rsplit("_", 1)[0] for p in PREDS.glob("diet_a_seed*_test.csv")})
    tuned = {}  # ad -> seed başına tahmin dosyası adları (results/tune/final_<ad>_seedN)
    for p in sorted(TUNE.glob("final_*_seed*_test.csv")):
        n = p.name.rsplit("_", 1)[0]
        tuned.setdefault(n.rsplit("_seed", 1)[0], []).append(n)
    print("| Model | Split | Macro-F1 | Accuracy |\n|---|---|---|---|")
    for split in ("dev", "test"):
        print(row("Majority", split, ["majority"]))
        print(row("TF-IDF char1-4 + LR", split, ["tfidf_lr"]))
        if a_names:
            print(row("DIET Config A (varsayılan)", split, a_names))
        for tname, names in tuned.items():
            print(row(f"DIET Config A (ayarlı: {tname[6:]})", split, names))
    for tname, names in tuned.items():
        print(f"\nTest, ayarlı A ({tname[6:]}) - TF-IDF+LR (eşleştirilmiş bootstrap):\n")
        base = read("tfidf_lr", "test")
        for n in names:
            d = read(n, "test")
            r = paired_bootstrap(list(base.y_true), list(base.y_pred), list(d.y_pred))
            print(f"- {n}: {r['diff']*100:+.2f} puan, 95% [{r['ci_low']*100:+.2f}, "
                  f"{r['ci_high']*100:+.2f}], sıfırı dışlıyor: {r['excludes_zero']}")
    if a_names:
        print("\nTest, Config A - TF-IDF+LR (eşleştirilmiş bootstrap, 1000 örneklem):\n")
        base = read("tfidf_lr", "test")
        for n in a_names:
            d = read(n, "test")
            r = paired_bootstrap(list(base.y_true), list(base.y_pred), list(d.y_pred))
            print(f"- {n}: {r['diff']*100:+.2f} puan, 95% [{r['ci_low']*100:+.2f}, "
                  f"{r['ci_high']*100:+.2f}], sıfırı dışlıyor: {r['excludes_zero']}")


if __name__ == "__main__":
    main()
