
from typing import TypedDict, Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages



class EvalResult(BaseModel):
    company: str
    aspect: str
    reason: str
    candidates: list[str]


class AgentState(TypedDict):
  user_input: str
  messages: Annotated[list, add_messages]
  result: EvalResult