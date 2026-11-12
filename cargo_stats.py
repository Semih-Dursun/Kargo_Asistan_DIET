"""Kargo verisi için doğrulama ve istatistik; train_diet.py'nin beklediği CSV'yi üretir.

Kullanım: PYTHONPATH=. .venv/bin/python cargo_stats.py [dosya.yml ...]
Varsayılan: data/cargo/train.yml (varsa), test.yml (varsa)

Kontroller: Rasa'nın dosyayı okuyabilmesi, intent başına örnek sayısı (hedef: ~15 kişi başı),
entity sayıları, birebir tekrarlar, bilinmeyen intent/entity adları, hizalama uyarıları.
Her yml için aynı adlı .csv (id,text,intent) yazılır.
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from rasa.shared.nlu.training_data.loading import load_data

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cargo"
INTENTS = {
    "kargo_takip", "teslimat_tarihi_sor", "kayip_kargo", "iade_talebi", "adres_degistir",
    "hasarli_kargo", "kargo_ucreti_sor", "sube_bul", "teslimat_randevu", "kargo_iptal",
    "gonderi_olustur", "kargo_gecikme_sikayet", "temsilci_talep",
}
ENTITIES = {"takip_no", "kargo_firmasi", "konum", "eticaret_sitesi", "desi"}
TARGET_PER_INTENT = 15


def check(path: Path) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        td = load_data(str(path))
    msgs = list(td.nlu_examples)
    print(f"\n== {path.name}: {len(msgs)} örnek")
    rows, seen, dups = [], set(), []
    intents, ents = Counter(), Counter()
    for i, m in enumerate(msgs):
        text, intent = m.get("text"), m.get("intent")
        intents[intent] += 1
        for e in m.get("entities", []):
            ents[e["entity"]] += 1
        if (text, intent) in seen:
            dups.append(text)
        seen.add((text, intent))
        rows.append({"id": i, "text": text, "intent": intent})
    if rows:
        pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    print("intent başına örnek (hedef >= %d):" % TARGET_PER_INTENT)
    for name in sorted(INTENTS):
        n = intents.get(name, 0)
        print(f"  {name:24s} {n:4d} {'' if n >= TARGET_PER_INTENT else '  <-- eksik'}")
    for name in sorted(set(intents) - INTENTS):
        print(f"  BİLİNMEYEN INTENT: {name} ({intents[name]})")
    print("entity:", dict(ents) or "yok")
    for name in sorted(set(ents) - ENTITIES):
        print(f"  BİLİNMEYEN ENTITY: {name}")
    if dups:
        print(f"  {len(dups)} birebir tekrar, örn: {dups[:3]}")
    mis = [str(x.message) for x in w if "Misaligned" in str(x.message)]
    if mis:
        print(f"  {len(mis)} hizalama uyarısı (entity sınırı/noktalama), ilki: {mis[0][:160]}")


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or [
        p for p in (DATA / "train.yml", DATA / "test.yml") if p.exists()]
    if not files:
        print("data/cargo/train.yml henüz yok; takımın cümleleri yazılınca çalıştırın.")
        return
    for f in files:
        check(f)


if __name__ == "__main__":
    main()
