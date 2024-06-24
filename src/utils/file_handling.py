import os
from typing import Any


def create_open(path: str, mode: str = "r", **kwargs: Any) -> Any:

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode, **kwargs)
