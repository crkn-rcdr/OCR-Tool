import os
import subprocess
import traceback

try:
    project_dir = r"C:\\CRKN Projects\\paddle_pipeline"
    os.chdir(project_dir)

    python_exe = r"C:\\CRKN Projects\\paddle_pipeline\\venv\\Scripts\\python.exe"

    subprocess.run([python_exe, "app.py"], check=True)

except Exception:
    traceback.print_exc()

input("Press Enter to close...")