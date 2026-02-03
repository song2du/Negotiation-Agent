from typing import TypedDict, Annotated, List, Dict, Optional, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class NegotiationState(TypedDict):
    # history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # meta data
    user_role: str         
    ai_role: str
    model: str
    mode: str             
    
    # scenario and summary
    ai_scenario: str
    user_scenario: str
    summary: str            
    
    # priority
    ai_priority: str        
    user_priority: str
    
    # goals
    ai_goals: Optional[Dict[str, int]]
    user_goals: Optional[Dict[str, int]]
    
    # inputs
    user_priority_inputs: Optional[Dict[str, int]]
    ai_priority_inputs: Optional[Dict[str, int]]
    
    # for React+Reflexion mode
    reflections: Optional[Annotated[List[str], operator.add]]
    buyer_reward: float
    seller_reward: float
    max_retries: int
    
    # evaluate result
    final_result: str
    buyer_score: float
    seller_score: float
    is_finished: bool