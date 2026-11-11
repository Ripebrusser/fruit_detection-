# Per-fruit evaluation

Test split: 480 images, overall accuracy 0.9688, fresh-vs-rotten accuracy 0.9875.

| Fruit | Images | Fruit correct | Fruit id. rate | Condition acc. | Fresh acc. | Rotten acc. | Precision | Recall | F1 | Main confusions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple | 60 | 54 | 0.900 | 1.000 | 1.000 | 1.000 | 0.982 | 0.900 | 0.939 | FreshPomegranate (4), RottenGuava (2) |
| Banana | 60 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — |
| Grape | 60 | 59 | 0.983 | 0.967 | 1.000 | 0.933 | 1.000 | 0.983 | 0.992 | RottenJujube (1) |
| Guava | 60 | 59 | 0.983 | 0.983 | 1.000 | 0.967 | 0.967 | 0.983 | 0.975 | FreshPomegranate (1) |
| Jujube | 60 | 59 | 0.983 | 1.000 | 1.000 | 1.000 | 0.952 | 0.983 | 0.967 | RottenOrange (1) |
| Orange | 60 | 58 | 0.967 | 0.967 | 0.933 | 1.000 | 0.983 | 0.967 | 0.975 | RottenJujube (2) |
| Pomegranate | 60 | 59 | 0.983 | 0.983 | 1.000 | 0.967 | 0.922 | 0.983 | 0.952 | RottenApple (1) |
| Strawberry | 60 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — |

## Focus classes

- **FreshPomegranate** — 30/30 correct, recall 1.000, precision 0.833, F1 0.909; confused with: none
- **RottenPomegranate** — 28/30 correct, recall 0.933, precision 1.000, F1 0.966; confused with: RottenApple (1), FreshPomegranate (1)
- **FreshApple** — 26/30 correct, recall 0.867, precision 1.000, F1 0.929; confused with: FreshPomegranate (4)
- **RottenApple** — 28/30 correct, recall 0.933, precision 0.966, F1 0.949; confused with: RottenGuava (2)
- **FreshOrange** — 28/30 correct, recall 0.933, precision 1.000, F1 0.966; confused with: RottenJujube (2)
- **RottenOrange** — 30/30 correct, recall 1.000, precision 0.968, F1 0.984; confused with: none

Misclassified images: 15 (listed in `Evaluation/test_results/misclassified.csv`).
