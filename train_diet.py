"""Bir DIET config'ini eğitir, dev+test üzerinde tahmin üretir, skor yazar.

Kullanım:
  PYTHONPATH=. .venv/bin/python train_diet.py massive a 42
  (veri seti: massive|cargo, config: a|b|c, seed)

Çıktı: models/<dataset>_<cfg>_seed<N>.tar.gz
       results/preds/diet_<cfg>_seed<N>_<split>.csv   (id, y_true, y_pred, confidence)
       results/diet_<cfg>_seed<N>.json                 (macro-F1, accuracy)

Değerlendirme CSV üzerinden TAM test setiyle yapılır (Rasa eğitimde birebir aynı
örnekleri tekilleştirir, değerlendirmede değil).
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("RASA_TELEMETRY_ENABLED", "false")

import pandas as pd

from metrics import scores

ROOT = Path(__file__).parent


def train(dataset, cfg, seed):
    name = f"{dataset}_{cfg}_seed{seed}"
    config = ROOT / "configs" / dataset / f"{cfg}_seed{seed}.yml"
    model = ROOT / "models" / f"{name}.tar.gz"
    if model.exists():
        print("model zaten var, eğitim atlandı:", model.name)
        return model
    cmd = [
        str(ROOT / ".venv" / "bin" / "rasa"), "train", "nlu",
        "--nlu", str(ROOT / "data" / dataset / "train.yml"),
        "--config", str(config),
        "--out", str(ROOT / "models"),
        "--fixed-model-name", name,
    ]
    subprocess.run(cmd, check=True)
    return model


async def predict(model, dataset, cfg, seed, splits=("dev", "test")):
    from rasa.core.agent import Agent

    agent = Agent.load(str(model))
    out = {}
    for split in splits:
        df = pd.read_csv(ROOT / "data" / dataset / f"{split}.csv")
        preds, confs = [], []
        for text in df["text"]:
            r = await agent.parse_message(str(text))
            preds.append(r["intent"]["name"])
            confs.append(r["intent"]["confidence"])
        pd.DataFrame({"id": df["id"], "y_true": df["intent"],
                      "y_pred": preds, "confidence": confs}).to_csv(
            ROOT / "results" / "preds" / f"diet_{cfg}_seed{seed}_{split}.csv", index=False)
        out[split] = scores(list(df["intent"]), preds)
        print(f"{cfg} seed{seed} {split}: macroF1={out[split]['macro_f1']*100:.2f} "
              f"acc={out[split]['accuracy']*100:.2f}")
    return out


def main():
    dataset, cfg, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])  # opsiyonel: --dev-only
    (ROOT / "results" / "preds").mkdir(parents=True, exist_ok=True)
    dev_only = "--dev-only" in sys.argv   # test setine dokunma (mid: Config B yalnızca dev)
    model = train(dataset, cfg, seed)
    res = asyncio.run(predict(model, dataset, cfg, seed, ("dev",) if dev_only else ("dev", "test")))
    suffix = "_devonly" if dev_only else ""
    (ROOT / "results" / f"diet_{cfg}_seed{seed}{suffix}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
