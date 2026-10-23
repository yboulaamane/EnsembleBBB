"""Check the app still starts when RDKit's drawing backend is unavailable.

The deployment failed on `from rdkit.Chem.Draw import rdMolDraw2D`, so the
import is now optional. That fallback is only useful if it actually works, and
it cannot be tested by running the app locally where the import succeeds -
so the import is blocked deliberately here.
"""
from __future__ import annotations

import builtins
import subprocess
import sys
import textwrap

HARNESS = textwrap.dedent("""
    import builtins
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("rdkit.Chem.Draw") or (
                name == "rdkit.Chem.Draw" or "rdMolDraw2D" in (fromlist or ())
                and name.endswith("Draw")):
            raise ImportError("libXrender.so.1: cannot open shared object file")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked

    import runpy, sys
    sys.argv = ["BBBScope.py"]
    try:
        runpy.run_path("BBBScope.py", run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:
        # Streamlit calls outside a script run context raise; anything from
        # rdkit.Chem.Draw would mean the fallback failed.
        if "rdkit" in repr(exc) or "Draw" in repr(exc):
            print(f"FAILED: {exc!r}")
            raise SystemExit(1)
    print("module-level import survived without the drawing backend")
""")

result = subprocess.run(
    [sys.executable, "-c", HARNESS],
    cwd="repo", capture_output=True, text=True, timeout=300)

print(result.stdout.strip() or "(no stdout)")
tail = [line for line in result.stderr.splitlines()
        if "rdkit" in line.lower() or "Draw" in line or "Error" in line]
if tail:
    print("\nstderr of interest:")
    for line in tail[-8:]:
        print(" ", line)
sys.exit(result.returncode)
