| Model | Split | Macro-F1 | Accuracy |
|---|---|---|---|
| Majority | dev | 0.21 (95% CI 0.2-0.2) | 6.44 |
| TF-IDF char1-4 + LR | dev | 82.65 (95% CI 79.5-84.9) | 83.28 |
| DIET Config A (varsayılan) (3 seed) | dev | 78.27 ± 1.28 | 80.90 ± 0.27 |
| Majority | test | 0.22 (95% CI 0.2-0.3) | 7.03 |
| TF-IDF char1-4 + LR | test | 81.72 (95% CI 79.3-83.7) | 83.42 |
| DIET Config A (varsayılan) (3 seed) | test | 77.32 ± 0.76 | 80.37 ± 0.05 |

Test, Config A - TF-IDF+LR (eşleştirilmiş bootstrap, 1000 örneklem):

- diet_a_seed42: -5.10 puan, 95% [-6.74, -3.10], sıfırı dışlıyor: True
- diet_a_seed43: -4.50 puan, 95% [-6.39, -2.41], sıfırı dışlıyor: True
- diet_a_seed44: -3.60 puan, 95% [-5.48, -1.66], sıfırı dışlıyor: True
