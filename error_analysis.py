"""Config A (seed 42) MASSIVE tr-TR test hatalarından 40 rastgele örnek (random_state=42) alır,
elle atanmış kategorileri ekler ve errors_configA_seed42.csv yazar.
Kategoriler okuma ile atanmıştır (öznel); ekip gözden geçirmelidir.
Kullanım: PYTHONPATH=. .venv/bin/python error_analysis.py
"""
import pandas as pd

# örnek sırası (sample(40, random_state=42) sonrası index) -> (kategori, not)
A, B, C, D, E = ("A same-domain neighbour", "B open-domain overlap", "C very short / underspecified",
                 "D label or translation artefact", "E other cross-domain")
CATS = {0: A, 1: C, 2: A, 3: B, 4: B, 5: D, 6: A, 7: C, 8: A, 9: B, 10: E, 11: D, 12: A, 13: C, 14: B,
        15: E, 16: B, 17: E, 18: A, 19: B, 20: D, 21: E, 22: A, 23: D, 24: E, 25: C, 26: D, 27: A,
        28: E, 29: A, 30: A, 31: B, 32: C, 33: D, 34: D, 35: E, 36: B, 37: C, 38: B, 39: E}

p = pd.read_csv("results/preds/diet_a_seed42_test.csv")
t = pd.read_csv("data/massive/test.csv")
d = p.merge(t[["id", "text"]], on="id")
e = d[d.y_true != d.y_pred]
print(f"test={len(d)} errors={len(e)} acc={1-len(e)/len(d):.4f}")
s = e.sample(40, random_state=42).reset_index(drop=True)
s["category"] = [CATS[i] for i in s.index]
s = s.rename(columns={"y_true": "gold", "y_pred": "predicted"})
s[["id", "text", "gold", "predicted", "confidence", "category"]].to_csv("errors_configA_seed42.csv", index=False)
print(s.category.value_counts().sort_index())
