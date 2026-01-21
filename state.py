from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class NegotiationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_role: str
    ai_role:str
    ai_scenario: str
    user_scenario: str
    ai_priority: str
    user_priority: str
    
    summary: str
    
    final_result: str
    buyer_score: int
    seller_score: int
    mediator_feedback: str
    is_finished: bool

    model: str