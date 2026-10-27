# Holdout (3) audit for EXP-020 — 27/10/2026

Local archive inspected: `RL_Cashew_holdout (3).zip`.

- SHA-256: `134072def34a6d766068b7c2de68085e3a6876d8544450d51b9428d0dfa6f6c9`
- Images: **126** — anthracnose 46, healthy 19, leaf_miner 36, red_rust 25.
- Exact byte overlap with all 7,045 images in `CashewData_Split (9).zip`: **0**.
- Exact byte duplicates within Holdout (3): **0**.

Compared with Holdout (2), `healthy/real_healthy_028.jpg` was removed and
eight other healthy images retained their paths but changed bytes. This is a
new evaluation dataset version; results from EXP-019 and EXP-020 should retain
their separate archive fingerprints. Byte checks do not establish that
different photos come from different leaves, trees or capture sessions.

EXP-020 has no executed model metrics in this repository. The notebook reads
the five EXP-018 Validation-locked checkpoints and writes only Holdout (3)
evaluation outputs when run in Colab.
