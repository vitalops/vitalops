"""Copies what the docs site shares with the package into docs/: vitalops's Python source (docs/py/vitalops/,
which the page loads into Pyodide) and the planner's JavaScript (vitalops/ui/ is the source of truth).
Run after changing either; the tests fail while docs/ is out of date.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = ["__init__.py", "hw.py", "models.py", "physics.py", "cli.py", "web.py"]
JS = ["hw.js", "lab-shell.js", "lab-planner.js"]


def pairs():
    for f in PY:
        yield os.path.join(ROOT, "vitalops", f), os.path.join(ROOT, "docs", "py", "vitalops", f)
    for f in JS:
        yield os.path.join(ROOT, "vitalops", "ui", f), os.path.join(ROOT, "docs", f)


def main() -> None:
    os.makedirs(os.path.join(ROOT, "docs", "py", "vitalops"), exist_ok=True)
    for src, dst in pairs():
        shutil.copyfile(src, dst)
        print("copied", os.path.relpath(src, ROOT), "->", os.path.relpath(dst, ROOT))


if __name__ == "__main__":
    main()
