"""Run the whole pipeline: python src/run_all.py   (data_collection.py needs internet + yfinance first)"""
import subprocess, sys
from pathlib import Path
here = Path(__file__).parent
for step in ["filtering", "features", "labels", "train", "evaluate", "detect"]:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(here / f"{step}.py")], check=True, cwd=here)
