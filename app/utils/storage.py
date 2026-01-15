from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from platformdirs import user_data_dir

APP_NAME = "Foxtrot-Delta-Pilot"
APP_AUTHOR = "Foxtrot-Delta"



# ===== Lokal base (kun config/logs) =====

def _get_local_base_dir() -> Path:
    base = Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_config_dir() -> Path:
    path = _get_local_base_dir() / "config"
    path.mkdir(exist_ok=True)
    return path


def get_logs_dir() -> Path:
    path = _get_local_base_dir() / "logs"
    path.mkdir(exist_ok=True)
    return path


# ===== USB-detektering =====

def _normalize_volume_guid(s: str) -> str:
    s = s.strip()

    # Fjern alle trailing backslashes og legg på én
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
        stderr=subprocess.STDOUT,
    )
    return {line.strip() for line in out.splitlines() if line.strip()}


def _get_volume_uniqueids() -> dict[str, str]:
    ps = r"""
    Get-Volume | Where-Object {$_.DriveLetter} |
      Select-Object DriveLetter, UniqueId |
      ForEach-Object { "$($_.DriveLetter):|$($_.UniqueId)" }
    """
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", ps],
        text=True,
        stderr=subprocess.STDOUT,
    )

    result: dict[str, str] = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        dl, uid = line.split("|", 1)
        result[dl.strip()] = _normalize_volume_guid(uid)

    return result

def _load_usb_policy() -> list[dict]:
    policy_path = get_config_dir() / "usb_policy.json"
    if not policy_path.exists():
        return []
    with open(policy_path, "r", encoding="utf-8") as f:
        p = json.load(f)
    if isinstance(p, dict):
        return p.get("approved_usb", [])
    return []



def _find_usb_base_dir() -> Optional[Path]:
    approved = _load_usb_policy()
    if not approved:
        return None

    removable = _get_removable_driveletters()
    uid_map = _get_volume_uniqueids()

    for drive in removable:
        uid = uid_map.get(drive)
        if not uid:
            continue

        for item in approved:
            allowed = _normalize_volume_guid(item.get("volume_guid", ""))
            if allowed and uid.lower() == allowed.lower():
                return Path(f"{drive}\\") / APP_NAME

    return None


# ===== Public API (brukes av resten av appen) =====

def get_meetings_dir() -> Path:
    print("Approved policy:", _load_usb_policy())
    print("Removable drives:", _get_removable_driveletters())
    print("UID map:", _get_volume_uniqueids())
    """
    Resultater lagres KUN her.
    Krever godkjent USB.
    """
    base = _find_usb_base_dir()
    if not base:
        raise RuntimeError("Ingen godkjent USB-minnebrikke funnet.")

    path = base / "meetings"
    path.mkdir(parents=True, exist_ok=True)
    

    return path
