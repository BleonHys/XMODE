import ast
import json
import os

from langchain_core.messages import HumanMessage

from src.build_graph import graph_construction_m3ae
from src.settings import get_settings


def main() -> None:
    settings = get_settings()
    settings.require_env("OPENAI_API_KEY")

    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain.project)

    model = settings.models.default_chat_model
    chain = graph_construction_m3ae(model, db_path=settings.ehrxqa.db_path)

    example_question = (
        "Is there any evidence in the most recent study for patient 13859433 this year "
        "indicating the continued absence of anatomical findings in the left hilar structure "
        "compared to the previous study?"
    )

    inputs = {"question": example_question, "database_schema": None}
    payload = [HumanMessage(content=[inputs])]

    for output in chain.stream(payload, stream_mode="values"):
        print(output)

    to_json = [msg.to_json()["kwargs"] for msg in output]
    ast.literal_eval(output[-1].content)

    with open("steps_dict.json", "w", encoding="utf-8") as handle:
        json.dump(to_json, handle, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
