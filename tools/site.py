"""Copies what the docs site shares with the package into docs/: tokplan's Python source (docs/py/tokplan/,
which the page loads into Pyodide) and the planner's JavaScript (tokplan/ui/ is the source of truth).
Run after changing either; the tests fail while docs/ is out of date.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = ["__init__.py", "hw.py", "models.py", "physics.py", "cli.py", "web.py"]
JS = ["hw.js", "lab-shell.js", "lab-planner.js"]


def pairs():
    for f in PY:
        yield os.path.join(ROOT, "tokplan", f), os.path.join(ROOT, "docs", "py", "tokplan", f)
    for f in JS:
        yield os.path.join(ROOT, "tokplan", "ui", f), os.path.join(ROOT, "docs", f)


def main() -> None:
    os.makedirs(os.path.join(ROOT, "docs", "py", "tokplan"), exist_ok=True)
    for src, dst in pairs():
        shutil.copyfile(src, dst)
        print("copied", os.path.relpath(src, ROOT), "->", os.path.relpath(dst, ROOT))


if __name__ == "__main__":
    main()
