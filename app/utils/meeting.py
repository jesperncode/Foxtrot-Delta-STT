import uuid
from datetime import datetime
from app.utils.storage import get_meetings_dir

def create_meeting():
    meeting_id = str(uuid.uuid4())
    meeting_dir = get_meetings_dir() / meeting_id
    meeting_dir.mkdir()

    (meeting_dir / "created_at.txt").write_text(
        datetime.now().isoformat(),
        encoding="utf-8"
    )

    return meeting_id, meeting_dir
