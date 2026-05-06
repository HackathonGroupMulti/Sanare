from __future__ import annotations

import os
import subprocess

from sanare.schemas import NvidiaRuntimeReport


def nvidia_runtime_report() -> NvidiaRuntimeReport:
    gpus: list[dict[str, str]] = []
    driver_version: str | None = None
    notes: list[str] = []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                name, memory_total, driver = parts[:3]
                driver_version = driver_version or driver
                gpus.append({"name": name, "memory_total_mb": memory_total, "driver_version": driver})
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        notes.append("nvidia-smi is unavailable; use hosted NVIDIA NIM or run on an NVIDIA GPU machine.")

    if not os.getenv("NVIDIA_API_KEY") and not os.getenv("NVIDIA_NIM_BASE_URL"):
        notes.append("Set NVIDIA_API_KEY for hosted NVIDIA endpoints or NVIDIA_NIM_BASE_URL for local NIM.")

    return NvidiaRuntimeReport(
        nvidia_smi_available=bool(gpus),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        driver_version=driver_version,
        gpus=gpus,
        recommended_provider="nvidia",
        nim_base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        notes=notes,
    )
