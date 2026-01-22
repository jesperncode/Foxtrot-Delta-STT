import uuid
from datetime import datetime
from pathlib import Path

from app.utils.storage import get_meetings_dir


def create_meeting():
    meeting_id = str(uuid.uuid4())
    meeting_dir: Path = get_meetings_dir() / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=False)

    (meeting_dir / "created_at.txt").write_text(
        datetime.now().isoformat(),
        encoding="utf-8",
    )

    return meeting_id, meeting_dir

