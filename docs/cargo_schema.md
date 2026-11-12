# Turkish cargo NLU set: schema (DRAFT v0.1)

Status: draft for the team to review and edit. The proposal fixes 13 intents and 5 entity
types and names four intents (`kargo_takip`, `teslimat_tarihi_sor`, `kayip_kargo`,
`iade_talebi`). The other nine, all definitions and all boundary examples below were drafted
with Claude (Anthropic) and must be reviewed by the team before they count as the schema.

## Entity types (5)

| Entity | Meaning | Example span |
|---|---|---|
| `takip_no` | tracking / shipment number | `TR123456789`, `1234 5678 9012` |
| `kargo_firmasi` | carrier | `Yurtiçi Kargo`, `Aras`, `MNG`, `PTT`, `Sürat`, `HepsiJet` |
| `konum` | province or district | `Ankara`, `Kadıköy` |
| `eticaret_sitesi` | e-commerce site | `Trendyol`, `Hepsiburada`, `N11` |
| `desi` | volumetric weight / size | `5 desi`, `30x40x20` |

Annotation rules: annotate the value only (no suffix: `Ankara'ya` -> `[Ankara](konum)'ya`);
one span per mention; no overlapping spans.

## Intents (13): definition and two boundary examples each

A boundary example is a short utterance that sits close to a neighbouring intent and says
which side it belongs to.

| # | Intent | Definition | Boundary example 1 | Boundary example 2 |
|---|---|---|---|---|
| 1 | `kargo_takip` | User wants the current status or location of a shipment. | "[TR123456789](takip_no) nerede" | "kargom şu an hangi aşamada" |
| 2 | `teslimat_tarihi_sor` | User asks WHEN it will arrive (a date), not where it is now. | "kargom ne zaman gelir" | "[Trendyol](eticaret_sitesi) siparişim hangi gün teslim edilecek" |
| 3 | `kayip_kargo` | Shipment is lost / not found / marked delivered but not received. | "kargom kayboldu" | "teslim edildi yazıyor ama elime ulaşmadı" |
| 4 | `iade_talebi` | User wants to send an item back or start a return. | "aldığım ürünü iade etmek istiyorum" | "[Hepsiburada](eticaret_sitesi) siparişimi geri göndermem lazım" |
| 5 | `adres_degistir` | Change the delivery address of a shipment already sent. | "kargonun adresini değiştirmek istiyorum" | "teslimat adresi yanlış girilmiş, [Kadıköy](konum)'e yönlendirir misiniz" |
| 6 | `hasarli_kargo` | Item arrived damaged or the package was broken. | "kargom hasarlı geldi" | "paket ezilmiş, ürün kırık çıktı" |
| 7 | `kargo_ucreti_sor` | Ask the price of sending, possibly by size/weight. | "[5 desi](desi) kargo ne kadar tutar" | "[Ankara](konum)'dan [İzmir](konum)'e kargo ücreti nedir" |
| 8 | `sube_bul` | Find a branch / drop-off point near a place. | "[Kadıköy](konum)'de [Aras Kargo](kargo_firmasi) şubesi nerede" | "en yakın kargo şubesini öğrenmek istiyorum" |
| 9 | `teslimat_randevu` | Reschedule or choose the delivery time/day (not the address). | "kargoyu cumartesi teslim eder misiniz" | "evde olmayacağım, teslimatı yarına alabilir miyiz" |
| 10 | `kargo_iptal` | Cancel a shipment before it is delivered (not a return of a delivered item). | "gönderdiğim kargoyu iptal etmek istiyorum" | "kargo çıkmadan siparişi iptal edebilir miyim" |
| 11 | `gonderi_olustur` | User wants to send a parcel / call a courier pickup. | "kargo göndermek istiyorum" | "adresimden kurye çağırabilir miyim" |
| 12 | `kargo_gecikme_sikayet` | Complaint that delivery is late (it still exists, it is late). | "kargom çok gecikti" | "[Yurtiçi Kargo](kargo_firmasi) üç gündür getirmedi" |
| 13 | `temsilci_talep` | Asks for a human agent / customer service contact. | "müşteri temsilcisine bağlanmak istiyorum" | "yetkili biriyle konuşabilir miyim" |

## Confusable pairs (plan the error analysis around these)

- `kargo_takip` vs `teslimat_tarihi_sor`: where it is now vs when it arrives.
- `kayip_kargo` vs `kargo_gecikme_sikayet`: lost for good vs late.
- `iade_talebi` vs `kargo_iptal`: after delivery vs before delivery.
- `adres_degistir` vs `teslimat_randevu`: where vs when.
- `hasarli_kargo` vs `iade_talebi`: damage report vs return request ("hasarlı ürünü iade etmek istiyorum"
  needs a rule: label by the user's main ask; write the rule here once the team decides).

## Out-of-scope (needed for the secondary metric)

The proposal reports the share of out-of-scope inputs caught by `nlu_fallback`, but plans no OOS
data. Collect about 30-50 OOS utterances for the TEST set only (weather, jokes, other
banking/shopping questions). They get no intent label in training.

## Collection plan (from the proposal)

1. Fix this schema (the team agrees on definitions and boundaries).
2. Each member writes ~15 utterances per intent in four registers: formal, colloquial,
   word-order inverted, misspelled.
3. Annotate in Rasa YAML. A second member re-labels 10% -> Cohen's kappa.
4. Test set (~150): written by 5 people outside the team who never see the training data.

Note for the kappa: labels written to a prompt are fixed by the prompt. The meaningful kappa is
on the externally written test set, labelled blind by two people.

Note for tuning: the cargo set has no dev split. Pick hyperparameters by cross-validation on
train, never on test.
