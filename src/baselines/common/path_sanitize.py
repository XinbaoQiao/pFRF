from __future__ import annotations

import os
import re


_ABS_USER_PREFIX = re.compile(r"^/(data|home)/[^/]+")


def sanitize_path_for_log(path: str, project_root: str | None = None) -> str:
    raw = str(path)
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    if project_root:
        pr = os.path.abspath(project_root).replace("\\", "/")
        if normalized == pr:
            return "${PROJECT_ROOT}"
        prefix = f"{pr}/"
        if normalized.startswith(prefix):
            rel = normalized[len(prefix) :]
            return f"${{PROJECT_ROOT}}/{rel}"
    if os.path.isabs(raw):
        return _ABS_USER_PREFIX.sub(r"/\1/<user>", normalized)
    return normalized
