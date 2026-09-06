from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str | None
    result: Any

