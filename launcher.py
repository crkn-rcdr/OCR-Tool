import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def _get_project_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _venv_candidates(project_dir: Path) -> list[Path]:
    if os.name == "nt":
        return [
            project_dir / "venv" / "Scripts" / "python.exe",
            project_dir / ".venv" / "Scripts" / "python.exe",
        ]

    return [
        project_dir / "venv" / "bin" / "python",
        project_dir / ".venv" / "bin" / "python",
    ]


def _resolve_python_command(project_dir: Path) -> list[str]:
    for candidate in _venv_candidates(project_dir):
        if candidate.exists():
            return [str(candidate)]

    if not getattr(sys, "frozen", False) and Path(sys.executable).exists():
        return [sys.executable]

    if os.name == "nt":
        py_launcher = shutil.which("py")
        if py_launcher is not None:
            return [py_launcher, "-3"]

    for name in ("python3", "python"):
        interpreter = shutil.which(name)
        if interpreter is not None:
            return [interpreter]

    raise FileNotFoundError(
        "No Python interpreter was found. Create a virtual environment in "
        "`venv` or `.venv`, or install Python 3."
    )


def main() -> None:
    project_dir = _get_project_dir()
    app_path = project_dir / "app.py"

    if not app_path.exists():
        raise FileNotFoundError(f"Could not find app.py in {project_dir}")

    python_cmd = _resolve_python_command(project_dir)
    env = os.environ.copy()
    env.setdefault("OCR_TOOL_PROJECT_DIR", str(project_dir))

    subprocess.run(
        [*python_cmd, str(app_path)],
        cwd=project_dir,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        if os.name == "nt" and sys.stdin.isatty():
            input("Press Enter to close...")
        raise SystemExit(1)
