from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, model_validator
from typing import Sequence
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union

from langchain_core.runnables import (
    chain as as_runnable,
)


class FinalResponse(BaseModel):
    """The final response/answer."""

    response: Union[str,Dict]


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]

    @model_validator(mode="before")
    @classmethod
    def _coerce_loose_outputs(cls, value):
        """Handle loosely formatted model outputs (e.g., bare 'Replan')."""
        if isinstance(value, str):
            # If the model only emitted a string, treat it as a replan request.
            return {"thought": "", "action": {"feedback": value}}
        if isinstance(value, dict):
            action_val = value.get("action")
            if isinstance(action_val, str):
                # Map plain string action to Replan or FinalResponse depending on content.
                lowered = action_val.lower()
                if "replan" in lowered:
                    value["action"] = {"feedback": action_val}
                else:
                    value["action"] = {"response": action_val}
        return value

def _strip_content(content: Any) -> Any:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        sanitized = []
        for item in content:
            if isinstance(item, dict):
                new_item = dict(item)
                if "text" in new_item and isinstance(new_item["text"], str):
                    new_item["text"] = new_item["text"].strip()
                sanitized.append(new_item)
            else:
                sanitized.append(_strip_content(item))
        return sanitized
    if isinstance(content, dict):
        return {
            key: _strip_content(value) if key == "text" else value
            for key, value in content.items()
        }
    return content


def parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=_strip_content(f"Thought: {decision.thought}"))]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=_strip_content(
                    f"Context from last attempt: {decision.action.feedback}"
                )
            )
        ]
    else:
        return response + [
            AIMessage(content=_strip_content(str(decision.action.response)))
        ]


def select_recent_messages(messages: list) -> dict:
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    sanitized = []
    for msg in selected[::-1]:
        clone = msg.copy()
        clone.content = _strip_content(clone.content)
        sanitized.append(clone)
    return {"messages": sanitized}
