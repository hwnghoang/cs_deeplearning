"""Run the 8 main configurations one after another (protocol A1.1, section 4.1).

    cd A1
    python run_all.py

Each run is a separate process (a crash or out-of-memory error does not stop the others);
its console output goes to results/<arch>_<mode>.log. Finished runs (results/<run>.json
exists) are skipped, so the script can be restarted.
"""
import subprocess, sys
from pathlib import Path

A1 = Path(__file__).resolve().parent

# Pretrained modes first (cheap, most important), from-scratch last.
EPOCHS = {"linear": 5, "partial": 5, "full": 5, "scratch": 10}
ORDER = [(arch, mode) for mode in ("linear", "partial", "full", "scratch") for arch in ("resnet50", "deit_s")]

for arch, mode in ORDER:
    run = f"{arch}_{mode}"
    if (A1 / "results" / f"{run}.json").exists():
        print("skip", run, flush=True)
        continue
    cmd = [sys.executable, "-m", "src.train", "--arch", arch, "--mode", mode, "--epochs", str(EPOCHS[mode])]
    print("run", " ".join(cmd[1:]), flush=True)
    (A1 / "results").mkdir(exist_ok=True)
    with open(A1 / "results" / f"{run}.log", "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, cwd=A1, stdout=log, stderr=subprocess.STDOUT)
    print(run, "exit code", code, flush=True)
