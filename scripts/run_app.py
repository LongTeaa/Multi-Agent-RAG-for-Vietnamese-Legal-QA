"""
scripts/run_app.py
Script to run both the FastAPI backend and the Vite frontend concurrently.
"""
import subprocess
import os
import sys
import signal
import time

def run_app():
    # Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_dir = root_dir
    frontend_dir = os.path.join(root_dir, "frontend")

    print("🚀 Starting Vietnamese Legal AI (Multi-Agent RAG)...")

    # Start Backend
    backend_cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    print(f"📡 Starting Backend on http://localhost:8000 (with auto-reload)...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=api_dir)

    # Give backend a moment to start
    time.sleep(2)

    # Start Frontend
    print(f"🎨 Starting Frontend in {frontend_dir}...")
    # On Windows, use 'npm.cmd' for npm
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    frontend_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=frontend_dir)

    print("\n✅ Both services are running!")
    print("👉 Frontend: http://localhost:5173")
    print("👉 Backend API: http://localhost:8000")
    print("\nPress Ctrl+C to stop both services.")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("❌ Backend stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("❌ Frontend stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    run_app()
