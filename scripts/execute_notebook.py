"""Regenerate the shareable results notebook from the canonical clean demo."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = root / "artifacts/jupyter-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IPYTHONDIR", str(runtime / "ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime / "connections"))
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    source = root / "notebooks/01_pattern_aware_training_demo.ipynb"
    destination = root / "notebooks/02_pattern_aware_training_results.ipynb"
    notebook = nbformat.read(source, as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    for cell in notebook.cells:
        # Remove timing/kernel metadata; retain verified synthetic displays.
        cell.metadata = {}
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise RuntimeError("Refusing to save an errored notebook")
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": ".".join(map(str, sys.version_info[:3]))},
    }
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "# Multimodal pattern-aware training", "# Multimodal pattern-aware training — executed results", 1
    ) + "\n\nThis output was regenerated from the public package. Run details appear below."
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    print(f"Wrote {destination.name}")


if __name__ == "__main__":
    main()
