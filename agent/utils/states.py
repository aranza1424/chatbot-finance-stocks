""" Module to define states to the agent"""

from langgraph.graph import MessagesState
import operator
from typing import Annotated, List, Literal
from pydantic import BaseModel
from typing_extensions import TypedDict

MessagesState



"""
-------------------------------------------
STATES for main agent
"""
class OverallQueryState(MessagesState):
    question: str
    context: list[str]
    answers: Annotated[List[List[str]], operator.add]


"""
--------------------------------------------
"""





"""
-------------------------------------------
STATES for query agent
"""
class Context(BaseModel):
    context: list[str]

class FinalAnswerQuery(BaseModel):
    answer: str

class QueryState(TypedDict):
    subquestion: str

"""
--------------------------------------------
"""