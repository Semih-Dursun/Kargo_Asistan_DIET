"""MASSIVE tr-TR parquet dosyalarını Rasa NLU YAML ve CSV'ye çevirir.

Girdi : data/raw/{train,validation,test}.parquet, data/raw/intent_names.json
Çıktı : data/massive/{train,dev,test}.yml  (Rasa, entity'li)
        data/massive/{train,dev,test}.csv  (text,intent; baseline'lar için)

Not: Veri HF'nin parquet dönüşümünden indirildi (betik çalıştırılmadan).
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "massive"
SPLITS = {"train": "train", "validation": "dev", "test": "test"}

# MASSIVE gösterimi: [slot_adı : değer]  ->  Rasa gösterimi: [değer](slot_adı)
SLOT_RE = re.compile(r"\[\s*([a-z_]+)\s*:\s*([^\]]+?)\s*\]")


def to_rasa(annot: str) -> str:
    return SLOT_RE.sub(lambda m: f"[{m.group(2)}]({m.group(1)})", annot)


def yaml_escape(line: str) -> str:
    # Satır "- " ile başladığı için çoğu karakter güvenli; blok skalerde sorun çıkarmaz.
    return line.replace("\n", " ").strip()


def main() -> None:
    names = json.loads((RAW / "intent_names.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for raw_name, name in SPLITS.items():
        df = pd.read_parquet(RAW / f"{raw_name}.parquet")
        df["intent_name"] = df["intent"].map(lambda i: names[i])
        df[["id", "utt", "intent_name"]].rename(
            columns={"utt": "text", "intent_name": "intent"}
        ).to_csv(OUT / f"{name}.csv", index=False)

        lines = ['version: "3.1"', "nlu:"]
        for intent, grp in df.groupby("intent_name", sort=True):
            lines.append(f"- intent: {intent}")
            lines.append("  examples: |")
            for annot in grp["annot_utt"]:
                lines.append(f"    - {yaml_escape(to_rasa(annot))}")
        (OUT / f"{name}.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(name, len(df), "örnek,", df["intent_name"].nunique(), "intent")


if __name__ == "__main__":
    main()
