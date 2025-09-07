from flask import Flask, render_template
import subprocess
import os
import signal
import atexit
import platform
from pathlib import Path

app = Flask(__name__, template_folder="templates")
streamlit_process = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/model")
def model():
    # simple client redirect to Streamlit
    return """
    <script>window.location.href = 'http://localhost:8501';</script>
    <p>Redirecting to AutVid AI...</p>
    """

def start_streamlit():
    """Launch Streamlit as subprocess (works on Windows/macOS/Linux)."""
    global streamlit_process
    if streamlit_process is not None:
        return
    # ensure model.py exists
    if not Path("model.py").exists():
        raise FileNotFoundError("model.py not found next to app.py")
    cmd = ["streamlit", "run", "model.py", "--server.port", "8501", "--server.headless", "true"]
    if platform.system() == "Windows":
        streamlit_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        # setsid to create a new process group so we can kill gracefully
        streamlit_process = subprocess.Popen(cmd, preexec_fn=os.setsid)

def stop_streamlit():
    """Kill Streamlit process when Flask exits."""
    global streamlit_process
    if streamlit_process:
        try:
            if platform.system() == "Windows":
                streamlit_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
        except Exception:
            try:
                streamlit_process.terminate()
            except Exception:
                pass
        streamlit_process = None

atexit.register(stop_streamlit)

if __name__ == "__main__":
    start_streamlit()
    app.run(debug=True, port=5000)
