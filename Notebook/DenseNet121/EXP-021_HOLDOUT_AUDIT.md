# Holdout (4) audit for EXP-021 — 27/10/2026

Local archive inspected: `RL_Cashew_holdout (4).zip`.

- SHA-256: `3b340ea830418638fa610fb2176dcc171739cc7fe0bd8e8430c8ed2a614717cc`
- Images: **120** — anthracnose 46, healthy 13, leaf_miner 36, red_rust 25.
- Exact byte overlap with all 7,051 images in `CashewData_Split (10).zip`: **0**.
- Exact byte duplicates within Holdout (4): **0**.

Compared with Holdout (3), these six healthy images were removed; no retained
image changed bytes: `real_healthy_035.jpg`, `real_healthy_037.jpg`,
`real_healthy_038.jpg`, `real_healthy_039.jpg`, `real_healthy_040.jpg`, and
`real_healthy_043.jpg`.

EXP-021 is an evaluation-only notebook. It reads the five Validation-locked
EXP-018 checkpoints and does not run training or the internal Final Test.
EXP-018 was trained on Split (9). Split (10), SHA-256
`b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac`,
is used here only as the overlap reference. Compared with Split (9), it adds
six `train/healthy` images: IDs 035, 037, 038, 039, 040 and 043; all 7,045
existing image paths retain identical bytes.
Exact-byte checks do not establish that different photographs come from
different leaves, trees, or capture sessions.
