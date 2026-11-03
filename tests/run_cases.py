"""Build cases in this process and print their digests as one JSON line.

    python tests/run_cases.py [--reverse] [CASE ...]

Each case starts from an empty scene. The test runs this twice, in two fresh
processes, the second in reverse order: equal digests mean every verb gives the
same mesh whatever ran before it. Exits through `os._exit`: bpy 4.5 hangs at
interpreter shutdown once a boolean has run.
"""
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

import sand3d as s3d  # noqa: E402
import cases  # noqa: E402,F401 — registers the cases
from harness import CASES, digest  # noqa: E402

MARK = "@@sand3d-digests "


def main(argv):
    reverse = "--reverse" in argv
    names = [a for a in argv if not a.startswith("--")] or list(CASES)
    if reverse:
        names = names[::-1]
    out = {}
    for name in names:
        fn, _ = CASES[name]
        s3d.empty_scene()
        s3d.units("mm")   # a case may have set another
        try:
            out[name] = digest(fn())
        except Exception:
            out[name] = "error: " + traceback.format_exc(limit=-3).strip().splitlines()[-1]
            traceback.print_exc()
    print(MARK + json.dumps({"digests": out, "env": cases.environment()}), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
