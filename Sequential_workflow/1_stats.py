from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic: str
    score: int
    summary: str = ""

    @field_validator
    def score_positive(cls, value):
        if value < 0:
            raise ValueError("Score must be positive")