from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.query_engine import CustomQueryEngine
from pydantic import Field
from config import *



import os

with open("API_KEY", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

llm = GoogleGenAI(model="models/gemini-2.0-flash")

class LlmQueryEngine(CustomQueryEngine):
    """Custom query engine for direct calls to the LLM model."""
    llm_gemini: GoogleGenAI | None = Field(default=None)
    prompt: str

    def custom_query(self, query_str: str):
        llm = self.llm_gemini
        llm_prompt = self.prompt.format(query=query_str)
        llm_response = llm.complete(llm_prompt)
        return str(llm_response)

llm_query_engine = LlmQueryEngine(llm_gemini=llm, prompt=DEFUALT_DIRECT_LLM_PROMPT)

llm_tool = QueryEngineTool.from_defaults(
    query_engine=llm_query_engine,
    name="llm_query_tool",
    description=DEFAULT_LLM_QUERY_TOOL_DESCRIPTION,
)

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm),
    query_engine_tools=[
        llm_tool
    ],
    llm=llm
)


chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    chat_history.append(f"User: {user_input}")

    conversation_context = "\n".join(chat_history) + "\nBot:"

    response = router_query_engine.query(conversation_context)

    print(f"Bot: {response}\n")

    chat_history.append(f"Bot: {response}")
