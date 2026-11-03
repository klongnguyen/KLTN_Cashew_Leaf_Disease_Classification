"""Execute EXP-033 cells sequentially on its explicit synthetic dry-run fixture."""
import json
import os
from pathlib import Path

os.environ["EXP033_DRY_RUN"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

notebook = json.loads(Path(__file__).with_name(
    "densenet121_yolo_combined_033.ipynb").read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    print(f"START CELL {index}: {source.splitlines()[0][:70]}", flush=True)
    exec(compile(source, f"cell_{index}", "exec"), namespace)
    print(f"PASS CELL {index}", flush=True)
