from typing import TypedDict, Annotated, List, Dict, Optional, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class NegotiationState(TypedDict):
    # Trajectory (STM)
    messages: Annotated[List[BaseMessage], add_messages] 
    # Experience (LTM)
    reflections: Annotated[List[str], operator.add]

    # Basic MetaData
    user_role: str
    ai_role:str
    ai_scenario: str
    user_scenario: str
    model: str
    
    # Priority
    ai_priority: str
    user_priority: str
    ai_goals: dict[str, int] # for evaluate node
    user_goals: dict[str, int]

    user_priority_inputs: Optional[Dict[str, int]]
    ai_priority_inputs: Optional[Dict[str, int]]

    # In-Context
    summary: str
    
    # Evaluate
    final_result: str
    buyer_score: int
    seller_score: int
    is_finished: bool

    

    