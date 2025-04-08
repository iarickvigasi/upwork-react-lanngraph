import os

from typing import List
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

system_prompt = """
Answer the following questions as best you can.
"""

tools = [TavilySearchResults(max_results=5)]

llm = ChatOpenAI(model="gpt-4o")

react_agent_runnable = create_react_agent(llm, tools=tools, prompt=system_prompt)

# Run the agent with a sample query
inputs = {"messages": [("user", "What are the latest developments in AI?")]}
for s in react_agent_runnable.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

