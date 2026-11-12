"""Ortak metrikler: macro-F1, accuracy ve eşleştirilmiş bootstrap.

Macro-F1 tanımı (tüm modeller için aynı): y_true içinde en az 1 örneği olan
sınıflar üzerinden F1 ortalaması. Test'te 59 intent var (cooking_query yok),
bu yüzden 59 üzerinden hesaplanır. 'nlu_fallback' gibi y_true'da olmayan
tahminler ilgili gerçek sınıf için FN sayılır.
"""
import numpy as np


def encode(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    t = np.array([idx[v] for v in y_true])
    p = np.array([idx[v] for v in y_pred])
    return t, p, len(labels)


def macro_f1_idx(t, p, k):
    tp = np.bincount(t[t == p], minlength=k).astype(float)
    support = np.bincount(t, minlength=k).astype(float)
    pred = np.bincount(p, minlength=k).astype(float)
    denom = support + pred
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f1[support > 0].mean()


def scores(y_true, y_pred):
    t, p, k = encode(y_true, y_pred)
    return {"macro_f1": macro_f1_idx(t, p, k), "accuracy": float((t == p).mean())}


def bootstrap_ci(y_true, y_pred, n=1000, seed=0, alpha=0.05):
    """Tek model için macro-F1 %95 aralığı."""
    t, p, k = encode(y_true, y_pred)
    rng = np.random.default_rng(seed)
    N = len(t)
    vals = np.empty(n)
    for b in range(n):
        s = rng.integers(0, N, N)
        vals[b] = macro_f1_idx(t[s], p[s], k)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, n=1000, seed=0, alpha=0.05):
    """B - A macro-F1 farkı için eşleştirilmiş bootstrap (aynı örnek indeksleri)."""
    labels = sorted(set(y_true) | set(pred_a) | set(pred_b))
    idx = {l: i for i, l in enumerate(labels)}
    t = np.array([idx[v] for v in y_true])
    a = np.array([idx[v] for v in pred_a])
    b = np.array([idx[v] for v in pred_b])
    k = len(labels)
    rng = np.random.default_rng(seed)
    N = len(t)
    diffs = np.empty(n)
    for i in range(n):
        s = rng.integers(0, N, N)
        diffs[i] = macro_f1_idx(t[s], b[s], k) - macro_f1_idx(t[s], a[s], k)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = macro_f1_idx(t, b, k) - macro_f1_idx(t, a, k)
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}
