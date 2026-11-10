"""Provider-independent file content."""

import base64
from dataclasses import dataclass


@dataclass(frozen=True)
class Content:
    data: bytes
    media_type: str

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_uri(self) -> str:
        return f"data:{self.media_type};base64,{self.base64}"

    @property
    def is_text(self) -> bool:
        return self.media_type.startswith("text/") or self.media_type in {
            "application/json",
            "application/xml",
            "application/javascript",
        }

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")
