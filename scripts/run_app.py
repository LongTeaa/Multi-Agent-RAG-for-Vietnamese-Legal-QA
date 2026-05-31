"""
scripts/run_app.py
Run both the FastAPI backend and the Vite frontend concurrently.
"""

import os
import subprocess
import sys
import time


def run_app():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_dir = root_dir
    frontend_dir = os.path.join(root_dir, "frontend")

    frontend_url = "http://127.0.0.1:5173"
    backend_url = "http://127.0.0.1:8000"

    print("Starting Vietnamese Legal AI (Multi-Agent RAG)...")

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    print(f"Starting Backend on {backend_url} (with auto-reload)...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=api_dir)

    time.sleep(2)

    print(f"Starting Frontend in {frontend_dir}...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1"],
        cwd=frontend_dir,
    )

    print("\nBoth services are running!")
    print(f"Frontend: {frontend_url}")
    print(f"Backend API: {backend_url}")
    print("\nPress Ctrl+C to stop both services.")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")


if __name__ == "__main__":
    run_app()
