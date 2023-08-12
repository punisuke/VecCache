import time
from dataclasses import dataclass, field


@dataclass
class StoredText:
    text: str
    timestamp: float = field(default_factory=time.time)
