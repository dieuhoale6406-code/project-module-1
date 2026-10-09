from pathlib import Path
from typing import Any, Dict, List


class JSONStorage:
    def __init__(self, file_path: str = "data/database.json") -> None:
        self.file_path = Path(file_path)

    def load_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError
