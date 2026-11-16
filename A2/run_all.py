"""Train the A2 variants one after another (proposal A2.1, section 5.1).

    cd A2
    python run_all.py

Each run is a separate process with its log in results/<run>.log; finished runs
(results/<run>.json exists) are skipped, so the script can be restarted.
"""
import subprocess, sys
from pathlib import Path

A2 = Path(__file__).resolve().parent

RUNS = [  # name, arch, init, epochs, lr drop after epoch
    ("V1_frcnn_coco", "frcnn", "coco", 10, 7),
    ("V2_detr_coco", "detr", "coco", 20, 15),
    ("V3_detr_imagenet", "detr", "imagenet", 20, 15),
    ("V5_frcnn_imagenet", "frcnn", "imagenet", 10, 7),
]

(A2 / "results").mkdir(exist_ok=True)
for run, arch, init, epochs, drop in RUNS:
    if (A2 / "results" / f"{run}.json").exists():
        print("skip", run, flush=True)
        continue
    cmd = [sys.executable, "-m", "src.train", "--run", run, "--arch", arch, "--init", init,
           "--epochs", str(epochs), "--lr_drop", str(drop)]
    print("run", " ".join(cmd[2:]), flush=True)
    with open(A2 / "results" / f"{run}.log", "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, cwd=A2, stdout=log, stderr=subprocess.STDOUT)
    print(run, "exit code", code, flush=True)
