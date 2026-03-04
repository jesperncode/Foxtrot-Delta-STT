from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from platformdirs import user_data_dir

APP_NAME = "Foxtrot-Delta-Pilot"
APP_AUTHOR = "Foxtrot-Delta"


# =========================
# Lokal config (kun policy)
# =========================

def get_config_dir() -> Path:
    base = Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
    path = base / "config"
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================
# USB helpers
# =========================

def _normalize_volume_guid(s: str) -> str:
    s = s.strip()
    while s.endswith("\\"):
        s = s[:-1]
    return s + "\\"


def _get_removable_driveletters() -> set[str]:
    ps = r"""
    (Get-CimInstance Win32_LogicalDisk |
      Where-Object {$_.DriveType -eq 2} |
      Select-Object -ExpandProperty DeviceID)
    """
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", ps],
        text=True,
    )
    return {line.strip() for line in out.splitlines() if line.strip()}


def _get_volume_uniqueids() -> dict[str, str]:
    ps = r"""
    Get-Volume | Where-Object {$_.DriveLetter} |
      ForEach-Object { "$($_.DriveLetter):|$($_.UniqueId)" }
    """
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", ps],
        text=True,
    )

    result = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        dl, uid = line.split("|", 1)
        result[dl.strip()] = _normalize_volume_guid(uid)
    return result


def _load_usb_policy() -> list[dict]:
    policy_path = get_config_dir() / "usb_policy.json"
    if not policy_path.exists():
        raise RuntimeError("usb_policy.json mangler")

    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            p = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"usb_policy.json er ugyldig JSON: {e}")

    return p.get("approved_usb", [])


def _find_usb_base_dir() -> Optional[Path]:
    approved = _load_usb_policy()
    removable = _get_removable_driveletters()
    uid_map = _get_volume_uniqueids()

    for drive in removable:
        uid = uid_map.get(drive)
        if not uid:
            continue

        for item in approved:
            allowed = _normalize_volume_guid(item["volume_guid"])
            if _normalize_volume_guid(uid).lower() == allowed.lower():
                # VIKTIG: bruk Volume GUID path, ikke D:\
                return Path(uid) / APP_NAME

    return None


# =========================
# Public API
# =========================

def get_meetings_dir() -> Path:
    base = _find_usb_base_dir()
    if not base:
        raise RuntimeError("Ingen godkjent USB-minnebrikke funnet.")

    path = base / "meetings"
    path.mkdir(parents=True, exist_ok=True)
    return path

