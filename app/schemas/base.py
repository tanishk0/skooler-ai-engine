from pydantic import BaseModel, ConfigDict


def to_camel(value: str) -> str:
    """Accept Pythonic names internally while matching JavaScript JSON conventions."""
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class APIModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
