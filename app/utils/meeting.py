import uuid
from datetime import datetime
from pathlib import Path

from app.utils.storage import get_meetings_dir


def create_meeting(meeting_name: str | None = None):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    short_id = uuid.uuid4().hex[:8]

    safe = ""
    if meeting_name:
        cleaned = "".join(
            c for c in meeting_name.strip()
            if c.isalnum() or c in (" ", "-", "_")
        )
        cleaned = "_".join(cleaned.split())
        safe = cleaned[:60]

    if safe:
        meeting_id = f"{ts}_{safe}_{short_id}"
    else:
        meeting_id = f"{ts}_{short_id}"

    meeting_dir = get_meetings_dir() / meeting_id
    meeting_dir.mkdir(parents = True, exist_ok = False)

    (meeting_dir / "created_at.txt").write_text(
        datetime.now().isoformat(),
        encoding = "utf-8"
    )

    if meeting_name:
        (meeting_dir / "meeting_name.txt").write_text(
            meeting_name.strip(),
            encoding = "utf-8"
        )
    return meeting_id, meeting_dir