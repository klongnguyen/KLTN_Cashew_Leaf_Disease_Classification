"""Execute EXP-034 on its explicit synthetic fixture; not a real experiment."""
import json
import os
from pathlib import Path

os.environ["EXP034_DRY_RUN"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

notebook = json.loads(Path(__file__).with_name(
    "densenet121_yolo_combined_training_034.ipynb").read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    print(f"START CELL {index}: {source.splitlines()[0][:70]}", flush=True)
    exec(compile(source, f"cell_{index}", "exec"), namespace)
    print(f"PASS CELL {index}", flush=True)
