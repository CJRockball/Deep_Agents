# Minimal Academic Research Agent

A barebone LangGraph ReAct agent for testing academic paper processing tools.

## Project Structure
```
minimal_research_agent/
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables
├── minimal_agent.py         # Main agent implementation
├── tool_integration.py      # Integration with academic_paper_tool
├── test_runner.py          # Test script
└── README.md               # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   - Copy `.env` and add your OpenAI API key
   - Or use Anthropic by setting `ANTHROPIC_API_KEY`

3. **Run the agent:**
   ```bash
   python test_runner.py
   ```

## How it works

The agent follows a simple linear ReAct flow:

1. **Input**: Takes a subject (for paper search) and a query (specific question)
2. **Search**: Uses `search_and_select_papers` to find relevant papers
3. **Process**: Uses `download_and_process_papers` to run the 9-stage pipeline
4. **Query**: Uses `query_processed_data` to answer the specific question
5. **Output**: Returns a comprehensive answer with citations

## Integration with Academic Paper Tool

To connect with your actual `academic_paper_tool`:

1. Ensure your tool is installed and accessible
2. Update the imports in `tool_integration.py` to match your actual package structure
3. Replace the mock tools in `minimal_agent.py` with imports from `tool_integration.py`

Example integration:
```python
from tool_integration import (
    search_and_select_papers,
    download_and_process_papers, 
    query_processed_data
)
```

## Agent Thinking Output

The agent provides detailed logging of its thinking process:
- 🤖 Agent nodes and decisions
- 🔍 Tool executions and reasoning
- 📊 Tool results and processing stages
- ➡️ Routing decisions between nodes

## Testing

Run `python test_runner.py` to test with predefined research scenarios or create your own by calling:

```python
from minimal_agent import run_research_agent

result = run_research_agent(
    subject="your research topic",
    query="your specific question"
)
```