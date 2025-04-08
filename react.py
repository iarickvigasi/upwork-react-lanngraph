import os

from typing import List
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

react_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


class SERPItem(BaseModel):
    """Represents a single item in the Search Engine Results Page (SERP)."""
    url: str = Field(..., description="The URL of the search result.")
    position: int = Field(..., description="The rank/position of the result in the SERP (1-based).")
    title: str = Field(None, description="The title of the search result page.")
    content: str = Field(None, description="The content keys value from the tavily search result.")
    is_editorial: bool = Field(..., description="Indicates if the result is editorial content (True) or not (False).")

class SERPResults(BaseModel):
    """Structured list of the SERP items."""
    results: List[SERPItem] = Field(..., description="A list containing the identified SERP items based on the search results.")

tools = [TavilySearchResults(max_results=5)]

llm = ChatOpenAI(model="gpt-4o")

react_agent_runnable = create_react_agent(llm, tools=tools, prompt=react_prompt)

# Example of how to run the agent
if __name__ == "__main__":
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        exit(1)
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in environment variables")
        exit(1)
        
    # Run the agent with a sample query
    query = "What are the latest developments in AI?"
    print(f"Running query: {query}")
    result = react_agent_runnable.invoke({"input": query})
    
    # Print the response
    print("\nAgent Response:")
    print(result)
    
    if isinstance(result, SERPResults):
        print("\nStructured Results:")
        print(result.model_dump_json(indent=2))
    else:
        print(f"\nOutput type: {type(result)}")
