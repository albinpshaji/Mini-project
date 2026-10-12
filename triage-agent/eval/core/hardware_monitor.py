"""
Hardware & Edge Memory Monitor.
Tracks resident GPU VRAM, temperature, and CPU core affinity load on the GTX 1650 budget.
"""

import os
import time
import threading
import subprocess
from typing import Dict, Any, List, Optional


def get_gpu_telemetry() -> Dict[str, Any]:
    """Reads VRAM and GPU utilization via nvidia-smi."""
    telemetry = {"vram_used_mb": 0.0, "vram_total_mb": 4096.0, "gpu_util_pct": 0.0, "temp_c": 0.0}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,nounits,noheader"],
            encoding="utf-8",
            timeout=1.0
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 4:
            telemetry["vram_used_mb"] = float(parts[0])
            telemetry["vram_total_mb"] = float(parts[1])
            telemetry["gpu_util_pct"] = float(parts[2])
            telemetry["temp_c"] = float(parts[3])
    except Exception:
        pass
    return telemetry


class HardwareSampler:
    """
    Background thread that samples hardware stats every `interval_s` and dumps to CSV.
    """

    def __init__(self, output_csv_path: Optional[str] = None, interval_s: float = 0.2):
        self.output_csv_path = output_csv_path
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []
        self.peak_vram_mb: float = 0.0

    def start(self):
        self._stop_event.clear()
        self.samples = []
        self.peak_vram_mb = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t0 = time.time()
        while not self._stop_event.is_set():
            data = get_gpu_telemetry()
            elapsed = time.time() - t0
            sample = {
                "elapsed_s": round(elapsed, 2),
                "vram_used_mb": data["vram_used_mb"],
                "gpu_util_pct": data["gpu_util_pct"],
                "temp_c": data["temp_c"]
            }
            if data["vram_used_mb"] > self.peak_vram_mb:
                self.peak_vram_mb = data["vram_used_mb"]
            self.samples.append(sample)
            time.sleep(self.interval_s)

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        # Write to CSV if path was provided
        if self.output_csv_path and self.samples:
            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
            try:
                import csv
                with open(self.output_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["elapsed_s", "vram_used_mb", "gpu_util_pct", "temp_c"])
                    writer.writeheader()
                    writer.writerows(self.samples)
            except Exception:
                pass

        return {
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "samples_count": len(self.samples)
        }
