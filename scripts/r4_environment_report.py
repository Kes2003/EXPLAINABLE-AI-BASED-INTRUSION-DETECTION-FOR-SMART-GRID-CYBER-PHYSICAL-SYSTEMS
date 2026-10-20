# scripts/r4_environment_report.py
"""Records the single computational environment used for every result of
Revision 4 (manuscript Section 3.9): interpreter, platform, CPU, RAM and the
versions of all scientific libraries; writes environment/requirements-lock.txt
(pip freeze) and results/r4/environment.json."""
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "catboost", "shap", "tensorflow",
        "matplotlib", "seaborn", "joblib", "yaml", "fastapi", "pydantic", "psutil", "py7zr"]


def main():
    import psutil
    info = {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
                              platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor(),
            "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "packages": {}}
    for p in PKGS:
        try:
            info["packages"][p] = importlib.import_module(p).__version__
        except Exception as e:  # noqa: BLE001
            info["packages"][p] = f"not importable ({type(e).__name__})"
    (ROOT / "environment").mkdir(exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (ROOT / "environment" / "requirements-lock.txt").write_text(
        f"# pip freeze of the environment that produced every Revision-4 result\n# Python {info['python']}\n" + freeze)
    (ROOT / "results" / "r4").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "r4" / "environment.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
