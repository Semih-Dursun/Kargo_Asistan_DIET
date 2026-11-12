"""DIET hiperparametre taraması, YALNIZCA dev üzerinde (test'e hiç dokunulmaz).

Her aday: config_a.yml'in DIETClassifier bölümüne override uygulanır, seed ile eğitilir,
sadece dev tahmin edilir. Sonuç results/tune/<ad>.json'a yazılır; varsa atlanır.

Kullanım: PYTHONPATH=. python tune_diet.py screen        # tek seed (42) tarama
          PYTHONPATH=. python tune_diet.py confirm ADAY  # ADAY'ı seed 42/43/44 ile dev'de
          PYTHONPATH=. python tune_diet.py final  ADAY   # ADAY ile 3 seed, dev+test (config'e yazar)
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("RASA_TELEMETRY_ENABLED", "false")

import pandas as pd
import yaml

from metrics import scores

ROOT = Path(__file__).parent
TUNE = ROOT / "results" / "tune"
MODELS = ROOT / "models" / "tune"

# Varsayılan: epochs 100, batch [64,128] (GPU), drop_rate 0.2, regularization 0.002,
# transformer 2 katman x 256, weight_sparsity 0.8. Her aday TEK bir şeyi değiştirir.
GRID = {
    "epochs50": {"epochs": 50},
    "drop0.4": {"drop_rate": 0.4},
    "reg0.01": {"regularization_constant": 0.01},
    "notransformer": {"number_of_transformer_layers": 0},
    "sparsity0.9": {"weight_sparsity": 0.9},
    "hidden256": {"hidden_layer_sizes": {"text": [256, 128], "label": []}},
}


def build_config(overrides, seed, dataset="massive"):
    cfg = yaml.safe_load((ROOT / "config_a.yml").read_text(encoding="utf-8"))
    for comp in cfg["pipeline"]:
        if comp["name"] == "DIETClassifier":
            comp["random_seed"] = seed
            comp.update(overrides)
    return cfg


def run_one(name, overrides, seed, splits=("dev",)):
    tag = f"{name}_seed{seed}"
    TUNE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    out_json = TUNE / f"{tag}.json"
    if out_json.exists():
        return json.loads(out_json.read_text())
    cfg_path = TUNE / f"{tag}.yml"
    cfg_path.write_text(yaml.safe_dump(build_config(overrides, seed), sort_keys=False))
    model = MODELS / f"{tag}.tar.gz"
    if not model.exists():
        subprocess.run([str(ROOT / ".venv" / "bin" / "rasa"), "train", "nlu",
                        "--nlu", str(ROOT / "data" / "massive" / "train.yml"),
                        "--config", str(cfg_path), "--out", str(MODELS),
                        "--fixed-model-name", tag], check=True)

    async def predict():
        from rasa.core.agent import Agent
        agent = Agent.load(str(model))
        res = {}
        for split in splits:
            df = pd.read_csv(ROOT / "data" / "massive" / f"{split}.csv")
            preds = [(await agent.parse_message(str(t)))["intent"]["name"] for t in df["text"]]
            pd.DataFrame({"id": df["id"], "y_true": df["intent"], "y_pred": preds}).to_csv(
                TUNE / f"{tag}_{split}.csv", index=False)
            res[split] = scores(list(df["intent"]), preds)
        return res

    res = {"name": name, "overrides": overrides, "seed": seed, **asyncio.run(predict())}
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[{tag}] dev macroF1={res['dev']['macro_f1']*100:.2f}", flush=True)
    return res


def spawn(name, overrides_name, seed, splits):
    """Her adayı AYRI süreçte çalıştır: önceki tahmin süreci GPU belleğini bırakır."""
    subprocess.run([sys.executable, __file__, "one", name, overrides_name, str(seed),
                    ",".join(splits)], check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})


def overrides_of(key):
    return {} if key == "default" else GRID[key]


def main():
    mode = sys.argv[1]
    if mode == "one":  # iç kullanım: tek aday, tek seed
        name, key, seed, splits = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(",")
        run_one(name, overrides_of(key), seed, splits=tuple(splits))
        return
    if mode == "screen":
        for key in ["default", *GRID]:
            spawn(key, key, 42, ("dev",))
    elif mode == "confirm":
        key = sys.argv[2]
        for seed in (42, 43, 44):
            spawn(key, key, seed, ("dev",))
    elif mode == "final":
        key = sys.argv[2]
        for seed in (42, 43, 44):
            spawn(f"final_{key}", key, seed, ("dev", "test"))
    rows = []
    for p in sorted(TUNE.glob("*_seed*.json")):
        r = json.loads(p.read_text())
        rows.append((r["name"], r["seed"], r["dev"]["macro_f1"] * 100))
    print("\nad | seed | dev macro-F1")
    for n, s, v in rows:
        print(f"{n} | {s} | {v:.2f}")


if __name__ == "__main__":
    main()
