# Read saved EXP-023 Validation artifacts only. No dataset ZIP, fit, Test or Holdout.
if VIEW_ONLY:
    from IPython.display import Image as DisplayImage
    for name in ("validation_results_5seeds.csv", "comparison_exp022_exp023_by_seed.csv",
                 "comparison_exp022_exp023_summary.csv", "per_class_validation_by_seed.csv",
                 "real_validation_by_seed.csv"):
        display(pd.read_csv(AGGREGATE_DIR / name))
    figures = [OUT_DIR / "figures" / "augmentation" / "photometric_qa_5classes.png"]
    figures += [PANEL_DIR / name for name in (
        "accuracy_loss_curves_5seeds.png", "validation_confusion_matrices_5seeds.png",
        "validation_confusion_matrices_5seeds_normalized.png",
        "validation_macro_f1_ci95.png", "paired_delta_macro_f1_ci95.png",
        "paired_delta_healthy_recall_ci95.png", "paired_delta_disease_to_healthy_ci95.png",
        "healthy_precision_recall_by_seed.png", "per_class_recall_ci95.png",
        "validation_errors_seed_selected.png")]
    for path in figures:
        if not path.is_file(): raise FileNotFoundError(path)
        print(path.relative_to(OUT_DIR))
        display(DisplayImage(filename=str(path)))
