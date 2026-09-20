"""Streamlit UI wrapper for SceneMotion-LLM."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / 'app.py'


def run_ui():
    """Launch the Streamlit web application."""
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Streamlit app not found at {APP_PATH}")

    subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(APP_PATH)])
