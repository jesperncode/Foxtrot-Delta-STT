from pathlib import Path
from platformdirs import user_data_dir

APP_NAME = "Foxtrot-Delta-Pilot"
APP_AUTHOR = "Foxtrot-Delta"

def get_base_dir() -> Path:
    base = Path(
        user_data_dir(
            appname=APP_NAME,
            appauthor=APP_AUTHOR
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_meetings_dir() -> Path:
    path = get_base_dir() / "meetings"
    path.mkdir(exist_ok=True)
    return path


def get_logs_dir() -> Path:
    path = get_base_dir() / "logs"
    path.mkdir(exist_ok=True)
    return path


def get_config_dir() -> Path:
    path = get_base_dir() / "config"
    path.mkdir(exist_ok=True)
    return path
