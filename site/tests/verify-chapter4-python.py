import contextlib
import io
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.loads(Path("lib/ml-chapter-four.json").read_text(encoding="utf-8"))
for course in data["courses"].values():
    blocks = re.findall(r"~~~python\n([\s\S]*?)~~~", course["text"])
    namespace = {"__name__": "course_example"}
    for index, block in enumerate(blocks, 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(block, f"cours{course['number']}_bloc{index}", "exec"), namespace)
        except Exception as error:
            raise RuntimeError(f"Cours {course['number']}, bloc {index}: {error}") from error
        plt.close("all")
    print(f"Cours {course['number']}: {len(blocks)} blocs Python executes.")
