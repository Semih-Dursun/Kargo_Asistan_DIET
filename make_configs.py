"""config_a/b/c.yml şablonlarından (veri seti, seed) başına config üretir.

Çıktı: configs/<dataset>/<a|b|c>_seed<N>.yml

- massive: FallbackClassifier YOK. MASSIVE'de out-of-scope sınıfı yok; fallback
  düşük güvenli örnekleri 'nlu_fallback' yapıp A-B farkını bozardı.
- cargo:   FallbackClassifier VAR (OOS yakalama ikincil metrik). Güven değerinin
  anlamlı olması için DIET'e constrain_similarities eklenir. Eşik dev'de seçilmeli.

Kullanım: python make_configs.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
SEEDS = [42, 43, 44]
TEMPLATES = {"a": "config_a.yml", "b": "config_b.yml", "c": "config_c.yml"}
FALLBACK_THRESHOLD = 0.3  # yer tutucu: cargo dev setinde ayarlanacak


def build(template: str, dataset: str, seed: int) -> dict:
    cfg = yaml.safe_load((ROOT / template).read_text(encoding="utf-8"))
    for comp in cfg["pipeline"]:
        if comp["name"] == "DIETClassifier":
            comp["random_seed"] = seed
            if dataset == "cargo":
                comp["constrain_similarities"] = True
    if dataset == "cargo":
        cfg["pipeline"].append(
            {"name": "FallbackClassifier", "threshold": FALLBACK_THRESHOLD}
        )
    return cfg


def main() -> None:
    for dataset in ("massive", "cargo"):
        out_dir = ROOT / "configs" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, template in TEMPLATES.items():
            for seed in SEEDS:
                cfg = build(template, dataset, seed)
                path = out_dir / f"{key}_seed{seed}.yml"
                path.write_text(
                    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
