"""Majority ve TF-IDF (char 1-4) + lojistik regresyon baseline'ları, MASSIVE tr-TR.

C hiperparametresi YALNIZCA dev'de seçilir; test'e dokunulmaz (Proposal 4).
Eğitim metni Rasa'nın gördüğü ile aynıdır (birebir aynı örnekler tekilleştirilir).
Çıktı: results/preds/<model>_<split>.csv ve results/baselines.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from metrics import bootstrap_ci, scores

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "massive"
PREDS = ROOT / "results" / "preds"
PREDS.mkdir(parents=True, exist_ok=True)


def load(split):
    return pd.read_csv(DATA / f"{split}.csv")


def report(name, split, df, pred, out):
    pd.DataFrame({"id": df["id"], "y_true": df["intent"], "y_pred": pred}).to_csv(
        PREDS / f"{name}_{split}.csv", index=False
    )
    s = scores(list(df["intent"]), list(pred))
    lo, hi = bootstrap_ci(list(df["intent"]), list(pred))
    out[f"{name}/{split}"] = {**s, "ci_low": lo, "ci_high": hi}
    print(f"{name:10s} {split:4s} macroF1={s['macro_f1']*100:5.2f} "
          f"[{lo*100:5.2f}, {hi*100:5.2f}]  acc={s['accuracy']*100:5.2f}")


def main():
    train = load("train").drop_duplicates(subset=["text", "intent"])
    dev, test = load("dev"), load("test")
    out = {}

    majority = train["intent"].value_counts().idxmax()
    for split, df in (("dev", dev), ("test", test)):
        report("majority", split, df, [majority] * len(df), out)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True)
    Xtr = vec.fit_transform(train["text"])
    best = None
    for C in (1, 3, 10, 30, 100):
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xtr, train["intent"])
        s = scores(list(dev["intent"]), list(clf.predict(vec.transform(dev["text"]))))
        print(f"  C={C:<4} dev macroF1={s['macro_f1']*100:.2f}")
        if best is None or s["macro_f1"] > best[0]:
            best = (s["macro_f1"], C, clf)
    _, C, clf = best
    out["tfidf_lr_C"] = C
    print("seçilen C (dev):", C)
    for split, df in (("dev", dev), ("test", test)):
        report("tfidf_lr", split, df, clf.predict(vec.transform(df["text"])), out)

    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
