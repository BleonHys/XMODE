import os
import base64
import json
from pathlib import Path
from typing import List, Optional, Union
import ast

from src.llm_factory import build_structured_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from PIL import Image
from src.settings import get_settings
from tools.vqa_m3ae import post_vqa_m3ae_with_url
from utils import correct_malformed_json

SETTINGS = get_settings()
IMAGE_ROOT = Path(
    os.environ.get("ARTWORK_IMAGE_DIR", SETTINGS.base_dir / "ArtWork" / "data")
)
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

# 'For example, 1. text2SQL("given the last study of patient 13859433 this year") and then 2. image_analysis("are there any anatomicalfinding that are still no in the left hilar structures in $1") is NEVER allowed. '
#'Use 2. image_analysis("are there any anatomicalfinding that are still no in the left hilar structures", context=["$1"]) instead.\n'
_DESCRIPTION = (
    " image_select(question:str, context: Union[str, List[str]])-> str\n"
    " This tools is a image retrieval task. For given question, it analysis the images and provide answer to the question. \n"
    " - Minimize the number of `image_select` actions as much as possible."
    # Context specific rules below
    " - You should provide either list of strings or string as `context` from previous agent to help the `image_select` agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `image_select` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do image_select on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract image_path which `image_select` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `image_select` action. "
)


def _load_image(image_path: Path):
    try:
        image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{image}"
    except FileNotFoundError:
        raise FileNotFoundError(f"Image_path <{image_path}> not found")
    except base64.binascii.Error:
        raise ValueError(f"Image_path <{image_url}> is not a valid image")
    except Exception as e:
        raise e

def extract_data(context):
    import re
    # Define the regex pattern to match the 'data' key and its value
    pattern = r"'data': (\[.*?\])\}"

    # Use re.search to find the 'data' value in the string
    match = re.search(pattern, context)

    if match:
        # Extract the matched value
        data_value = match.group(1)
        return data_value
    else:
        return None


def get_image_select_tool(db_path: str):
    """
   
    Args:
        question (str): The question about the image.
        context Union[str, List[str]]
    Returns:
        str: the answer to the question about the image.
    """

    def image_analysis(
        question: str,
        context: Union[str, List[str]],
    ):
        print("context-first:", context, type(context))

        if isinstance(context, str):
            context = correct_malformed_json(context)
            context = [ast.literal_eval(context)]
            if 'status' in context[0]:
                context = context[0]
        else:
            print("context-2", context)
            context = ast.literal_eval(context[0])

        print("context-2", context)
            # If the context contains 'data' key, use its value
        if 'data' in context:
            #["{'status': 'success', 'data': [{'studydatetime': '2105-09-06 18:18:18'}]}"]
            context = context['data']

        print("context-after:", context)
        if not isinstance(context, list):
            print("context-after in not list", list(context))
            context = list(context)
        try:
            image_answers = []

            for ctx in context:
                raw_path = ctx.get('image_path') or ctx.get('img_path')
                if not raw_path:
                    continue
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = IMAGE_ROOT / candidate
                images_encoded = _load_image(candidate)
                vqa_answer = post_vqa_m3ae_with_url(question, images_encoded)
                ctx[question] = vqa_answer.get('vqa_answers')
                image_answers.append(ctx)

            if len(image_answers) > 1:
                return image_answers
            if image_answers:
                return image_answers[0].get(question)
            return None
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="image_analysis",
        func=image_analysis,
        description=_DESCRIPTION,
    )
