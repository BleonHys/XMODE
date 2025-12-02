import ast
import re
from typing import List, Optional, Union
import json
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
    " data_preparation (question:str, context: Union[str, List[str],dict])-> str\n"
    " This tools is a data preparation task. For given data and question, it porcess the data and prepare the proper data structure for a request. \n"
    " - Minimize the number of `data_preparation` actions as much as possible."
    " if you want this tools does its job properly, you should include all required information from the user query in previous tasks."
  
    
    # Context specific rules below"
)

# " Plotting or any other visualization request should be done after each analysis.\n"
_SYSTEM_PROMPT = """You are a data preparation and processing assistant. Create a proper structure for the provided data from the previous steps to answer the request.
- If the required information has not found in the provided data, ask for replaning and ask from previous tools to include the missing information.
- You should include all the input data in the code, and prevent of ignoring them by  `# ... (rest of the data)`.
- You should provide a name or caption for each value in the final output considering the question and the input context."
- Dont create any sample data in order to answer to the user question.
- You should print the final data structure.
- You should save the final data structure at the specified path with a proper filename.
- You should output the final data structure as a final output.
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
    
    data: str = Field(
        ...,
        description="The final data structure as a final output.",
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
            return f"Failed to execute. Error: {repr(e)}"
        return result
        
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

def _compress_data(parsed_data: list, max_rows: int = 10):
    """
    Reduce oversized datasets passed to the LLM by aggregating and sampling.
    """
    if not parsed_data:
        return parsed_data
    # Group by year-like fields to keep plotting-friendly info small.
    year_counts = {}
    for row in parsed_data:
        year = (
            row.get("inception_year")
            or row.get("year")
            or row.get("studydatetime")
        )
        if year is None:
            continue
        year_str = str(year)
        year_counts[year_str] = year_counts.get(year_str, 0) + (
            row.get("count_of_paintings")
            or row.get("painting_count")
            or 1
        )
    sample = parsed_data[:max_rows]
    return {"summary": {"total_rows": len(parsed_data), "year_counts": year_counts}, "sample": sample}

def _invoke_with_retry(extractor, chain_input, attempts: int = 2):
    last_err = None
    for _ in range(attempts):
        try:
            return extractor.invoke(chain_input)
        except Exception as exc:
            # If the model truncated structured output, surface a clear signal
            if "max_tokens" in str(exc) or "ValidationError" in str(exc):
                return {
                    "status": "error",
                    "message": "LLM truncated structured output (max_tokens); replan or retry with higher max_tokens.",
                }
            last_err = exc
    raise last_err


def get_data_preparation_tools(llm: BaseChatModel, log_path):
    """
   
    Args:
        question (str): The question.
        context list(str)
    Returns:
        dataframe: the dataframe that is needed for a plot genration task.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            ("user", "{context}"),
            
        ]
    )
    
    extractor = build_structured_runnable(llm, prompt, ExecuteCode)


    def data_preparation(
        question: Optional[str] = None,
        context: Union[str, List[str],dict] = None,
        config: Optional[RunnableConfig] = None,
    ):
        if not question:
            return {"status": "error", "message": "data_preparation called without question"}
        print("context-first:", context,type(context))
        context_str= str(context).strip()
        # context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
        #     context= context_str.strip()
        # )
        # if 'data' in context:
        #     context=context['data']
        parsed_data = _extract_data_from_context(context_str)
        if parsed_data is not None and len(parsed_data) == 0:
            return {"status": "error", "message": "No data returned from previous step; replan to retrieve rows before preparing data."}
        if parsed_data:
            parsed_data = _compress_data(parsed_data)
            # If we already have aggregated data, return it directly to avoid long LLM calls
            if isinstance(parsed_data, dict) and "summary" in parsed_data:
                year_counts = parsed_data["summary"].get("year_counts", {})
                aggregated = [{"year": y, "count": c} for y, c in sorted(year_counts.items())]
                return {"status": "success", "data": aggregated, "note": "Aggregated locally to reduce prompt size."}
            context_str = str(parsed_data)
        try:
            data_len = len(parsed_data) if parsed_data is not None else "n/a"
            print(f"[debug:data_preparation] context_len={len(context_str)}, parsed_data_len={data_len}")
        except Exception:
            pass
        context_str += f"Save the generated data to the following directory: {log_path} and output the final data structure in data filed"
        chain_input = {"question": question,"context":context_str}
        # chain_input["context"] = [SystemMessage(content=context)]
                       
        try:
            code_model = _invoke_with_retry(extractor, chain_input, attempts=2)
        except Exception as e:
            return {"status": "error", "message": f"data_preparation failed: {e}"}

        if code_model.code=='':
            return code_model.reasoning 
        codeExecution_result = python_repl.run(code_model.code)
        if "Error" in codeExecution_result:
            _error_handiling_prompt=f"Something went wrong on executing Code: `{code_model.code}`. This is the error I got: `{codeExecution_result}`. \\ Can you fixed the problem and write the fixed python code?"
            chain_input["info"] =[HumanMessage(content= _error_handiling_prompt)]
            code_model = extractor.invoke(chain_input)
            try:
                return code_model.data
            except Exception as e:
                return repr(e)
        else:
            # extract data from the code
           
            return code_model.data


    return StructuredTool.from_function(
        name = "data_preparation",
        func = data_preparation,
        description=_DESCRIPTION,
    )
