from typing import TypedDict
from pydantic import BaseModel

class Reason(TypedDict):
    requirements: str

class Accept(BaseModel):
    logic: str
    accept: bool
