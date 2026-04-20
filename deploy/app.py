"""Hugging Face Spaces entry point.

HF Spaces auto-runs `streamlit run app.py` at the repo root. This shim keeps
the real Streamlit code in `app/streamlit_app.py` (so the local dev path
`streamlit run app/streamlit_app.py` keeps working) and re-exports it here.
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Execute the real app module in-place — Streamlit picks up all st.* calls.
runpy.run_path(str(ROOT / "app" / "streamlit_app.py"), run_name="__main__")
