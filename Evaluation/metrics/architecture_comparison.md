# Architecture comparison: 16-class softmax vs two heads

Same data, same backbone, same schedule (8 frozen epochs + 8 fine-tune epochs), one
evaluation pass each on the untouched test split.

| Metric | 16-class single head | Multi-task (8 + 2) | Difference |
| --- | --- | --- | --- |
| 16-class accuracy | 0.9688 | 0.9646 | -0.0042 |
| fresh vs rotten accuracy | 0.9875 | 0.9854 | -0.0021 |
| fruit identity accuracy | 0.9750 | 0.9792 | +0.0042 |
| macro F1 (16-class) | 0.9689 | — | — |

Multi-task training time: 537 s on CPU.

The shipped model keeps the 16-class head unless the multi-task model is measurably
better here; see `Admin/FINAL_MODEL_EVALUATION_REPORT.md` for the decision.
