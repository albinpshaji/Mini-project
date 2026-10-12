#!/usr/bin/env python3
"""
OmniSpec-Triage Web Server Runner.
Launches the FastAPI server on 0.0.0.0 so you can open it on your smartphone browser.
"""

import os
import sys
import socket
import argparse
import subprocess
from pathlib import Path

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import uvicorn


def get_lan_ip() -> str:
    """Discovers host LAN IP for phone connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def ensure_ssl_certs(cert_dir: Path) -> tuple[Path, Path]:
    """Generates self-signed SSL certs for HTTPS if requested."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if not cert_path.exists() or not key_path.exists():
        print("🔐 Generating self-signed SSL certificate for HTTPS microphone access...")
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-days", "365", "-nodes",
            "-subj", "/CN=omnispec-triage"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("✅ SSL certificate generated successfully in certs/")
        except Exception as e:
            print(f"⚠️ Failed to generate SSL cert via openssl: {e}")
            return None, None

    return cert_path, key_path


def print_banner(host: str, port: int, lan_ip: str, is_ssl: bool):
    proto = "https" if is_ssl else "http"
    print("\n" + "=" * 74)
    print(" 🏥 OMNISPEC-TRIAGE LIVE CLINICAL WEB & MOBILE INTERFACE")
    print("=" * 74)
    print(f" 💻 Laptop Browser:")
    print(f"    👉 {proto}://localhost:{port}")
    print()
    print(f" 📱 Phone Browser (Same Wi-Fi network):")
    print(f"    👉 {proto}://{lan_ip}:{port}")
    print()
    print(" 🎙️ Phone Talking Guide:")
    if is_ssl:
        print("    1. Open the URL above on Chrome / Safari on your smartphone.")
        print("    2. If you see 'Your connection is not private', tap 'Advanced' -> 'Proceed'.")
        print("    3. Tap the glowing Microphone button and speak symptoms directly!")
    else:
        print("    1. Open the URL above on Chrome / Safari on your smartphone.")
        print("    2. Tap the Microphone button to record speech, or tap the Upload button")
        print("       to use your phone's native audio recorder over Wi-Fi.")
        print("    3. You can also type in live chat or tap the quick symptom chips!")
    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="OmniSpec-Triage Web Kiosk Runner")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000)")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS with self-signed certificate")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    lan_ip = get_lan_ip()
    ssl_certfile = None
    ssl_keyfile = None

    if args.ssl:
        cert_p, key_p = ensure_ssl_certs(ROOT_DIR / "certs")
        if cert_p and key_p:
            ssl_certfile = str(cert_p)
            ssl_keyfile = str(key_p)
        else:
            print("⚠️ Reverting to plain HTTP due to SSL generation issue.")
            args.ssl = False

    print_banner(args.host, args.port, lan_ip, args.ssl)

    uvicorn.run(
        "src.web.server:app",
        host=args.host,
        port=args.port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
