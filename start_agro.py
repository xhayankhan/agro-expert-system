# launch.py  →  Fixed to use .venv Python (port 7860)
import subprocess
import time
import sys
import socket

PORT = 7860
SCRIPT = "app_agro.py"

def check_port(port, max_wait=30):
    """Wait until Flask is actually listening"""
    print(f"Waiting for Flask to start on port {port}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                print(f"✓ Flask is ready on port {port}!")
                return True
        except:
            pass
        time.sleep(1)
    return False

print(f"Starting AgroExpert Vision on http://127.0.0.1:{PORT}")

# Use sys.executable to run with the SAME Python (inside .venv)
flask_process = subprocess.Popen([
    sys.executable, SCRIPT  # ← Fixed: uses your .venv Python
])

# Wait for Flask to actually be ready
if not check_port(PORT):
    print(f"\n❌ Flask didn't start on port {PORT}!")
    flask_process.terminate()
    sys.exit(1)

print("\nLaunching public Cloudflare Tunnel...")
print("Your AgroExpert Vision is going live!\n")

# Launch Cloudflare tunnel
tunnel_process = subprocess.Popen([
    "cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"
])

print("Public link will appear below in a few seconds")
print("Share this link with farmers!\n")

try:
    tunnel_process.wait()
except KeyboardInterrupt:
    print("\n\nShutting down AgroExpert Vision...")
    flask_process.terminate()
    tunnel_process.terminate()
    print("Bye!")
    sys.exit(0)