import ast
import re
import time
from typing import List, Optional, Union
import json
from pathlib import Path
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from src.llm_factory import build_structured_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.language_models import BaseChatModel
from PIL import Image
import base64
from pathlib import Path

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_experimental.tools import PythonAstREPLTool



_DESCRIPTION = (
    " data_plotting (question:str, context: Union[str, List[str],dict])-> str\n"
    " This tools is a data plotting task. For given data and a question, it analysis the data and plot a proper chart to answer a user query. \n"
    " - Minimize the number of `data_plotting` actions as much as possible."
    " if you want this tools does its job properly, you should include all required information from the user query in previous tasks."
    
    # Context specific rules below"
)

# " Plotting or any other visualization request should be done after each analysis.\n"
_SYSTEM_PROMPT = """You are a data plotting assistant. Plot the the provided data from the previous steps to answer the question.
- Analyze the user's request and input data to determine the most suitable type of visualization/plot that also can be understood by the simple user.
- If the required information has not found in the provided data, ask for replaning and ask from previous tools to include the missing information.
- Dont create any sample data in order to answer to the user question.
- You should save the generated plot at the specified path with the proper filename and .png extension.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
    Use it to substitute into any ${{#}} variables or other words in the problem.\
    \n\n${context}\n\nNote that context variables are not defined in code yet.\
You must extract the relevant data and directly put them in code.
"""

  
class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the answer, including how context is included, if applicable.",
    )

    code: str = Field(
        "",
        description="The simple code expression to execute by python_executor.",
    )

def extract_code_from_block(response):
    if '```' not in response:
        return response
    if '```python' in response:
        code_regex = r'```python(.+?)```'
    else:
        code_regex = r'```(.+?)```'
    code_matches = re.findall(code_regex, response, re.DOTALL)
    code_matches = [item for item in code_matches]
    return  "\n".join(code_matches)

class PythonREPL:
    def __init__(self):
        self.local_vars = {}
        self.python_tool = PythonAstREPLTool()
    def run(self, code: str) -> str:
        code = extract_code_from_block(code) 
        # print(code)
        # output = str(self.python_tool.run(code))
        
        # if output == "":
        #     return "Your code is executed successfully"
        # else:
        #     return output
        try:
            result = self.python_tool.run(code)
        except Exception as e:
            print(f"Failed to execute. Error: {repr(e)}")
            return f"Failed to execute. Error: {repr(e)}"
        return f"Plot created successfully!:\n```python\n{code}\n```\nStdout: {result}"
        
python_repl = PythonREPL() 
      
def _extract_data_from_context(context) -> Optional[list]:
    try:
        payload = context
        if isinstance(payload, str):
            payload = ast.literal_eval(payload)
        if isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]
        if isinstance(payload, dict) and "data" in payload:
            return payload.get("data")
    except Exception:
        return None
    return None

def _invoke_with_retry(extractor, chain_input, attempts: int = 2):
    last_err = None
    for _ in range(attempts):
        try:
            return extractor.invoke(chain_input)
        except Exception as exc:
            if "max_tokens" in str(exc) or "ValidationError" in str(exc):
                return {
                    "status": "error",
                    "message": "LLM truncated structured output (max_tokens); replan or retry with higher max_tokens.",
                    "truncated": True,
                }
            last_err = exc
    return {
        "status": "error",
        "message": f"data_plotting failed: {last_err}",
    }


def get_plotting_tools(llm: BaseChatModel, log_path):
    """
   
    Args:
        question (str): The question.
        context list(str)
    Returns:
        python code: the python code that is needed in the plot genration task.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            ("user", "{context}"),
            
        ]
    )
    
    extractor = build_structured_runnable(llm, prompt, ExecuteCode)


    def data_plotting(
        question: Optional[str] = None,
        context: Union[str, List[str],dict] = None,
        config: Optional[RunnableConfig] = None,
    ):
        if not question:
            return {"status": "error", "message": "data_plotting called without question"}
        #test
        
        #context="[{'study_id': 56222792, 'image_id': '3c7d71b0-383da7fc-80f78f8c-6be2da46-3614e059'}]"
       # data= [{'week': '48', 'male_patient_count': 6}, {'week': '49', 'male_patient_count': 2}, {'week': '50', 'male_patient_count': 2}, {'week': '51', 'male_patient_count': 1}, {'week': '52', 'male_patient_count': 7}]
       
        print("context-first:", context,type(context))
        context_str= str(context).strip()
        # context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
        #     context= context_str.strip()
        # )
        # if 'data' in context:
        #     context=context['data']
        parsed_data = _extract_data_from_context(context_str)
        if parsed_data is not None and len(parsed_data) == 0:
            return {"status": "error", "message": "No data returned from previous step; replan to retrieve rows before plotting."}
        if parsed_data:
            context_str = str(parsed_data)
        try:
            data_len = len(parsed_data) if parsed_data is not None else "n/a"
            print(f"[debug:data_plotting] context_len={len(context_str)}, parsed_data_len={data_len}")
        except Exception:
            pass
        if len(context_str) > 72000:
            time.sleep(60)
        context_str += f"Save the generated plot to the following directory: {log_path}"
        chain_input = {"question": question,"context":context_str}
        # chain_input["context"] = [SystemMessage(content=context)]
                       
        try:
            code_model = _invoke_with_retry(extractor, chain_input, attempts=2)
        except Exception as e:
            return {"status": "error", "message": f"data_plotting failed: {e}"}

        if code_model.code=='':
            return code_model.reasoning 
        # Ensure log directory exists and track before/after files
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        before = {p.name for p in log_dir.glob("*.png")}

        codeExecution_result = python_repl.run(code_model.code)

        after = {p.name for p in log_dir.glob("*.png")}
        new_files = sorted(list(after - before))

        if "Error" in codeExecution_result:
            _error_handiling_prompt=f"Something went wrong on executing Code: `{code_model.code}`. This is the error I got: `{codeExecution_result}`. \\ Can you fixed the problem and write the fixed python code?"
            chain_input["info"] =[HumanMessage(content= _error_handiling_prompt)]
            code_model = extractor.invoke(chain_input)
            try:
                codeExecution_result = python_repl.run(code_model.code)
                # refresh new files
                after = {p.name for p in log_dir.glob("*.png")}
                new_files = sorted(list(after - before))
            except Exception as e:
                return repr(e)

        if new_files:
            return {"status": "success", "plot_path": str(log_dir / new_files[-1]), "note": codeExecution_result}
        return {
            "status": "error",
            "message": "Plot code executed but no PNG was saved; check the generated code to ensure plt.savefig() writes to the provided log directory.",
            "details": codeExecution_result,
        }

    return StructuredTool.from_function(
        name = "data_plotting",
        func = data_plotting,
        description=_DESCRIPTION,
    )
