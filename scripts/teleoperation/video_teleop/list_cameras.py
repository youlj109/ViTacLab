"""List camera indices (delegates to ``video_teleop.tools.list_cameras``)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
source_dir = project_root / "source"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from video_teleop.tools.list_cameras import main

if __name__ == "__main__":
    main()
