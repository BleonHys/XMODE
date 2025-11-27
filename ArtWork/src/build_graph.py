import os

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from src.joiner import Replan, JoinOutputs
from src.joiner import *


from langchain import hub
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate


import itertools
from typing import Dict

from langgraph.graph import END, MessageGraph, START


from langchain_core.runnables import (
    chain as as_runnable,
)
from src.planner import create_planner
from src.task_fetching_unit import schedule_tasks
from src.llm_factory import build_chat_model, build_structured_runnable
from src.vqa_factory import build_vqa
from tools.SQL import get_text2SQL_tools
from tools.visual_qa import get_image_analysis_tools
from tools.plot import get_plotting_tools
from tools.data import get_data_preparation_tools
from langgraph.checkpoint.memory import MemorySaver



def graph_construction(
    model,
    temperature,
    db_path,
    log_path,
    saver=None,
    vqa_provider: str = "blip",
    settings=None,
):

    ## Tools
    base_kwargs = {"model": model, "temperature": temperature}
    if settings is None:
        from src.settings import get_settings

        settings = get_settings()
    vqa_model = build_vqa(vqa_provider, settings)
    translate = get_text2SQL_tools(build_chat_model(**base_kwargs), db_path)
    image_analysis = get_image_analysis_tools(vqa_model)
    data_preparation = get_data_preparation_tools(build_chat_model(**base_kwargs), log_path=log_path)
    plotting = get_plotting_tools(build_chat_model(**base_kwargs), log_path=log_path)

    tools = [translate, image_analysis, data_preparation, plotting]
    llm = build_chat_model(**base_kwargs)

# In sub_question related to text2SQL include all requested information to be retrieved at once.
 # For each image analysis task, generate three distinct questions that each convey the same idea in different wording.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict planner. Produce a task list using the available tools.
Rules:
- NEVER emit an action without all required args. Every text2SQL action MUST include problem=<full user question>. If you omit it, the run fails.
- Each action must have a unique, increasing idx.
- Inputs from previous actions use $<idx> in args.
- Plan for maximal parallelism but correctness first.
- Use data_preparation before data_plotting; include plotting only if the user asked for a plot/visualization.
- Use image_analysis only when visual inspection is needed (the DB has no depiction content).
- In any text2SQL retrieval, include all relevant columns needed for downstream steps (e.g., inception for plotting/century).
- Each sub-question is textual (no code). Keep the plan minimal if unsure—do not emit empty/incomplete actions.
Available actions ({num_tools} total):
{tool_descriptions}
{num_tools}. join(): Collects and combines results from prior actions.
join is always last; append <END_OF_PLAN> after join.
""",
            ),
            ("user", "{messages}"),
            ("assistant", "Remember, ONLY respond with the task list in the correct format! E.g.:\nidx. tool(arg_name=args)"),
        ]
    )
    # This is the primary "agent" in our application
    planner = create_planner(llm, tools, prompt)
    #example_question = "is there evidence in the last study for patient 13859433 this year of any anatomical findings in the left hilar structures still absent compared to the previous study?"

    
    @as_runnable
    def plan_and_schedule(messages: List[BaseMessage], config):
        tasks = planner.stream(messages, config)
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        scheduled_tasks = schedule_tasks.invoke(
            {
                "messages": messages,
                "tasks": tasks,
            },
            config,
        )
        
        return scheduled_tasks
    
    joiner_prompt=ChatPromptTemplate.from_messages(
        [("system",'''Solve a question answering task. Here are some guidelines:
    - In the Assistant Scratchpad, you will be given results of a plan you have executed to answer the user's question.
    - Thought needs to reason about the question based on the Observations in 1-2 sentences.
    - Ignore irrelevant action results.
    - If the required information is present, give a concise but complete and helpful answer to the user's question.
    - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:
    Thought: <reason about the task results and whether you have sufficient information to answer the question>
    Action: <action to take>
    - If an error occurs during previous actions, replan and take corrective measures to obtain the required information.
    - Ensure that you consider errors in all the previous steps, and tries to replan accordingly.
    - Ensure the final answer is provided in a structured format as JSON as follows:
        {{'Summary': <concise summary of the answer>,
         'details': <detailed explanation and supporting information>,
         'source': <source of the information or how it was obtained>,
         'inference':<your final inference as YES, No, or list of requested information without any extra information which you can take from the `labels` as given below>,
         'extra explanation':<put here the extra information that you dont provide in inference >,
         }}
         In the `inferencer` do not provide additinal explanation or description. Put them in `extra explanation`.

    Available actions:
    (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
    (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan

    Using the above previous actions, decide whether to replan or finish. 
    If all the required information is present, you may finish. 
    If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you. 
    ''' ),
        ("user", '{messages}'),
        ]
    ).partial(
        examples=""
    )  
    
    runnable = build_structured_runnable(llm, joiner_prompt, JoinOutputs)
    
    joiner = select_recent_messages | runnable | parse_joiner_output
    
    graph_builder = MessageGraph()

    # 1.  Define vertices
    # We defined plan_and_schedule above already
    # Assign each node to a state variable to update
    graph_builder.add_node("plan_and_schedule", plan_and_schedule)
    graph_builder.add_node("join", joiner)


    ## Define edges
    graph_builder.add_edge("plan_and_schedule", "join")

    ### This condition determines looping logic


    def should_continue(state: List[BaseMessage]):
        if isinstance(state[-1], AIMessage):
            return END
        # Prevent infinite replanning loops
        max_replans = int(os.environ.get("XMODE_MAX_REPLANS", "6"))
        replan_count = sum(1 for msg in state if isinstance(msg, SystemMessage))
        if replan_count >= max_replans:
            return END
        return "plan_and_schedule"

    # Set up memory
    #memory = MemorySaver()  
    graph_builder.add_conditional_edges(
            "join",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            #{"plan_and_schedule": "plan_and_schedule", "__end__": "__end__"},
        )
    graph_builder.add_edge(START, "plan_and_schedule")
    chain = graph_builder.compile()
    
    return chain
        
