"""Validate chapter 5 behavior and the same checks used by the browser."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "public" / "downloads"
CHECKS = ROOT / "public" / "python" / "checks.py"
sys.path.insert(0, str(DOWNLOADS))
data = json.loads((ROOT / "lib" / "ml-chapter-five.json").read_text(encoding="utf8"))
assert len(data["lessons"]) == len(data["courses"]) == len(data["program"]) == 10
assert [c["number"] for c in data["courses"].values()] == list(range(1, 11))
for course in data["courses"].values():
    assert all(0 <= q["answer"] < len(q["options"]) for q in course["quiz"])
    assert "import numpy as npimport" not in course["text"]
for script in DOWNLOADS.glob("ch05_cours*.py"):
    ast.parse(script.read_text(encoding="utf8"), filename=str(script))
for number in range(1, 11):
    prefix = f"ch05_cours{number:02d}"
    assert (DOWNLOADS / (prefix + "_lab.py")).is_file()
    assert (DOWNLOADS / (prefix + "_starter.py")).is_file()
    assert (DOWNLOADS / (prefix + "_solution.py")).is_file()

count = 0
for exercise in ["ch02_cours01", "ch03_cours01", "ch05_cours03_browser"] + [f"ch05_cours{n:02d}" for n in range(4, 11)]:
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = runpy.run_path(str(DOWNLOADS / (exercise + "_solution.py")), run_name="grading")
        checks = runpy.run_path(str(CHECKS), init_globals={"namespace": namespace, "exercise": exercise})["results"]
    failed = [c for c in checks if not c["passed"]]
    assert checks and not failed, (exercise, failed)
    count += len(checks)
    with contextlib.redirect_stdout(io.StringIO()):
        starter = runpy.run_path(str(DOWNLOADS / (exercise + "_starter.py")), run_name="grading")
        checks = runpy.run_path(str(CHECKS), init_globals={"namespace": starter, "exercise": exercise})["results"]
    assert any(not c["passed"] for c in checks), exercise + ": unfinished starter must fail"
    print("PASS", exercise)

for number in [1, 2, 4, 5, 6, 7, 8, 9, 10]:
    prefix = f"ch05_cours{number:02d}"
    namespace = runpy.run_path(str(DOWNLOADS / (prefix + "_solution.py")), run_name="verification")
    with contextlib.redirect_stdout(io.StringIO()):
        result = namespace["run"]()
    assert result is not None or number in [1, 2]
    print("PASS full lab", prefix)
if all(importlib.util.find_spec(name) for name in ("shap", "lime", "xgboost")):
    namespace = runpy.run_path(str(DOWNLOADS / "ch05_cours03_solution.py"), run_name="verification")
    with contextlib.redirect_stdout(io.StringIO()):
        importance, stability = namespace["run"]()
    assert len(importance) == 8 and len(stability) == 9
    print("PASS full XAI lab")
else:
    print("SKIP full XAI runtime: install shap, lime, xgboost in the local Python environment.")
print(f"PASS {count} behavioral checks; starters rejected; 10 courses and script syntax verified.")
