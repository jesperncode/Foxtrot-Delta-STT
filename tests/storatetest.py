from app.utils.storage import (
    get_base_dir,
    get_meetings_dir,
    get_logs_dir,
    get_config_dir
)
from pathlib import Path
from datetime import datetime
import uuid

def run_manual_storage_test():
    print("=== STORAGE TEST START ===")

    base = get_base_dir()
    meetings = get_meetings_dir()
    logs = get_logs_dir()
    config = get_config_dir()

    print(f"Base dir:     {base}")
    print(f"Meetings:     {meetings}")
    print(f"Logs:         {logs}")
    print(f"Config:       {config}")

    # Lag et fake møte
    meeting_id = str(uuid.uuid4())
    meeting_dir = meetings / meeting_id
    meeting_dir.mkdir()

    print(f"Created meeting dir: {meeting_dir}")

    # Skriv testfiler
    (meeting_dir / "created_at.txt").write_text(
        datetime.now().isoformat(),
        encoding="utf-8"
    )

    (meeting_dir / "summary.txt").write_text(
        "Dette er et test-referat.",
        encoding="utf-8"
    )

    print("Wrote test files")

    # Les tilbake
    summary = (meeting_dir / "summary.txt").read_text(encoding="utf-8")
    print("Read back summary:", summary)

    print("=== STORAGE TEST OK ===")

if __name__ == "__main__":
    run_manual_storage_test()
