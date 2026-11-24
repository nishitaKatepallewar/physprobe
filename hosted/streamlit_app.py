"""Entry point for the hosted, read-only explorer (for example on Streamlit Community Cloud).

It shows the results bundle committed under results/ (created with
`python -m physprobe export --runs runs --out results`). Visitors can add their own
bundles with the upload box on the Overview page. Running the model is not possible here.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PHYSPROBE_HOSTED"] = "1"
sys.argv = [sys.argv[0], "--runs", str(ROOT / "results"), "--data", str(ROOT / "results" / "datasets"), "--hosted"]
runpy.run_path(str(ROOT / "physprobe" / "gui" / "app.py"), run_name="__main__")
