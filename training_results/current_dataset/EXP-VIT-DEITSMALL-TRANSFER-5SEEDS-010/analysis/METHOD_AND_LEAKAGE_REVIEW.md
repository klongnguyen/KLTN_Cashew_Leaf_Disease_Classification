# Method and leakage review — EXP-010

## Source inheritance and controlled differences

The EXP-007 ViT training notebook, EXP-008 Validation classifier notebook/artifacts and EXP-009 inference-only source notebook were inspected before writing EXP-010. EXP-007 uses the distilled DeiT-Tiny KerasHub preset, RGB pixel values 0–255, bilinear 224×224 resize, one preset preprocessor, and the same mild train augmentation recipe. EXP-010 keeps the image pipeline, split and class order. The intended differences are DeiT-Small's 384-D backbone and partial fine-tuning of the last two encoder blocks. Stage 2 uses a verified common LR because differential LR was not established in this Keras stack.

EXP-008 Tiny A Validation metrics and per-class reports are preserved in this folder. The EXP-007 full archive exists on the user's Drive but was not available as a local checkpoint for independent inference parity in this run. EXP-009's source protocol was reviewed; its Test/Holdout results were not available for review and cannot influence EXP-010.

## Split and leakage audit

The V08 ZIP and manifest SHA256 match the frozen source. Counts are Train 4,790; Validation 1,430; Test 686. Every ZIP member passed CRC and image decode verification. SHA256 was computed per image; no exact duplicate was found within or across splits. There are 226 Train/Val/Test cross-split pHash candidates at Hamming ≤6. pHash can group visually similar but distinct leaves and can miss different photos of the same plant. These pairs need source metadata or human inspection. Until then, independence by leaf/tree/photo session is unverified.

The 35 `real_` Validation images are a tagged subset, not an independent holdout. Metrics for that subset will be descriptive only. Test members are accessed only in the read-only dataset audit, never extracted to the training work directory or passed to the model. Historical Holdout is not opened.

## Selection discipline

The frozen JSON config specifies seeds, trainable layers, optimizers, schedules, Validation checkpoint rule, stage/seed tie-breaks and promotion guardrails before any five-seed run. Both Stage 1 and Stage 2 are evaluated on the same Validation set. Five seed outcomes measure initialization/training variability on one fixed split; they are not five independent datasets. Comparisons to EXP-008 Tiny A remain observational because model capacity and fine-tuning strategy change together. No Test-based threshold or post-Test seed switch is allowed.
