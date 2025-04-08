import os
import textwrap

from typing import List
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

class SERPItem(BaseModel):
    """Represents a single item in the Search Engine Results Page (SERP)."""
    url: str = Field(..., description="The URL of the search result.")
    position: int = Field(..., description="The rank/position of the result in the SERP (1-based).")
    title: str = Field(None, description="The title of the search result page.")
    content: str = Field(None, description="The content keys value from the tavily search result.")
    is_editorial: bool = Field(description="Indicates if the result is editorial content (True) or not (False).")

class SERPResults(BaseModel):
    """Structured list of the SERP items."""
    results: List[SERPItem] = Field(..., description="A list containing the identified SERP items based on the search results.")

# Initialize the tool with include_raw_content to get full search results
tools = [TavilySearchResults(max_results=5, include_raw_content=True)]

llm = ChatOpenAI(model="gpt-4o")

prompt = """
You are a helpful assistant that can search the web for information.
"""

# Create the ReAct agent with structured_output using Pydantic model's schema
react_agent_runnable = create_react_agent(
    llm, 
    tools=tools, 
    prompt=prompt,
    response_format=SERPResults
)

# Run the agent with a sample query
if __name__ == "__main__":
    query = "What are the latest developments in AI?"
    inputs = {"messages": [("user", query)]}
    
    print(f"Running ReAct agent query: '{query}'\n")

    result = react_agent_runnable.invoke(inputs)
    
    print("\n" + "="*50)
    print("SEARCH RESULTS FOR: '{}'".format(query))
    print("="*50 + "\n")
    
    # The structured response is directly available in the result object
    if hasattr(result, 'structured_response'):
        serp_results = result.structured_response
    else:
        # Print all available keys to help debug
        print(f"Result type: {type(result)}")
        
        # Check for alternative locations of the results
        if isinstance(result, dict) and 'structured_response' in result:
            serp_results = result['structured_response']
        else:
            print("Search results not found in expected format.")
            print(f"Available data structure: {result}")
            exit(1)
    
    # Display each search result in a formatted way
    for i, item in enumerate(serp_results.results, 1):
        print(f"RESULT #{i} [Position: {item.position}]")
        print(f"Title: {item.title}")
        print(f"URL: {item.url}")
        
        if item.is_editorial:
            print("Type: Editorial Content")
            
        # Format content for readability
        if item.content:
            print("\nContent Summary:")
            # Wrap text at 80 characters
            wrapped_content = textwrap.fill(item.content, width=80) if item.content else "No content available"
            print(wrapped_content)
        
        print("\n" + "-"*50 + "\n")
    
    # Print total number of results
    print(f"Total results found: {len(serp_results.results)}")
    