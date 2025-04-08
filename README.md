# React LangGraph Project

This project uses LangChain and LangGraph to implement a ReAct agent that searches for and extracts editorial content using Tavily search.

## Setup

### Prerequisites
- Python 3.9+
- Poetry (for dependency management)
- OpenAI API key
- Tavily API key

### Installation

1. Clone the repository
2. Install dependencies with Poetry:

```bash
poetry install
```

3. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

Run the main application:

```bash
poetry run python main.py
```

By default, the application searches for "running shoes" and returns the top editorial pages from search results.

You can modify the search query in `main.py`. 