# Basic Deep Agent Implementation

A minimal, educational implementation of LangGraph's `create_deep_agent` function demonstrating the core concepts of Deep Agents for complex, long-horizon tasks.

## What are Deep Agents?

Deep Agents represent an evolution beyond simple "shallow" agents that just call tools in a loop. Inspired by systems like Claude Code, Deep Research, and Manus, Deep Agents implement four key architectural components:

1. **📋 Planning Tool** - Built-in TODO list management for structured task planning
2. **🤖 Sub-Agents** - Specialized agents for context quarantine and domain expertise  
3. **💾 Virtual File System** - Persistent memory using LangGraph's state system
4. **📝 Detailed System Prompt** - Comprehensive instructions for sophisticated reasoning

This implementation showcases these concepts in their simplest form, making it ideal for learning and experimentation.

## About LangGraph's create_deep_agent Function

The `create_deep_agent` function is part of the official [deepagents](https://github.com/langchain-ai/deepagents) package, designed to make building deep agents accessible to developers. Key features include:

- **Built-in Tools**: Automatic TODO management (`write_todos`) and virtual file system (`ls`, `read_file`, `write_file`, `edit_file`)
- **Sub-Agent Support**: Context isolation and specialized task delegation
- **Claude Code Inspiration**: System prompt and architecture based on proven production systems
- **LangGraph Integration**: Seamless state management and execution flow

## What This Implementation Demonstrates

This basic Deep Agent includes:

✅ **Required Parameters**:
- `tools`: Custom web search using Tavily API
- `instructions`: Comprehensive workflow for research and reflection

✅ **Optional Features**:
- `subagents`: Two specialized agents (researcher and reflector)
- `model`: Rate-limited Google Gemini 2.0 Flash
- `recursion_limit`: Proper execution controls

✅ **Built-in Capabilities**:
- Planning workflow with `write_to_dos`
- Virtual file system for persistent memory
- Reflection and iteration patterns
- Context management across sub-agents

## Prerequisites

### Required Libraries

```bash
pip install deepagents
pip install langchain
pip install langchain-google-genai
pip install python-dotenv
pip install tavily-python
```

### API Keys

You'll need API keys for:
- **Google Gemini**: Get from [Google AI Studio](https://aistudio.google.com/)
- **Tavily Search**: Get from [Tavily](https://tavily.com/)
- **OpenAI** (optional): For summarization model

## Setup

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd Basic_deepagent
   ```

2. **Create a `.env` file** with your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional
   RECURSION_LIMIT=50
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Agent

### Basic Usage

```python
from deepagent import agent

# Simple research query
result = agent.invoke({
    'messages': [{'role': 'user', 'content': 'Research AI agents in finance and provide analysis'}]
}, {'recursion_limit': 50})

print(result['messages'][-1].content)
```

### With Conversation Formatter

The repository includes a conversation formatter to view the complete agent dialogue:

```python
from pretty_print import pretty_print_conversation

result = agent.invoke({...})
pretty_print_conversation(result)
```

This displays:
- User messages and agent responses
- Tool calls and results  
- Sub-agent handoffs
- Planning and file operations

## Architecture Overview

### Main Components

1. **🔍 Web Search Tool**
   - Tavily-powered internet search
   - Returns formatted results for research

2. **🤖 Researcher Sub-Agent**
   - Specializes in information gathering
   - Uses built-in planning tools
   - Saves findings to virtual file system

3. **🧠 Reflector Sub-Agent**  
   - Critical evaluation and analysis
   - Iteration limits to prevent loops
   - Quality assessment and improvement suggestions

4. **📋 Built-in Workflow**
   - PLAN → RESEARCH → REFLECT → ITERATE → FINALIZE
   - Automatic TODO creation and tracking
   - Context preservation across steps

### Key Features Demonstrated

- **Context Quarantine**: Sub-agents work in isolated contexts
- **Persistent Memory**: Virtual file system maintains state
- **Iterative Refinement**: Reflection-based improvement cycles  
- **Tool Integration**: Seamless custom tool usage
- **Rate Limiting**: Production-ready API management

## Example Output

```
🤖 AGENT CONVERSATION TRANSCRIPT
============================================================

 1 | 👤 USER [HumanMessage]
    Research AI agents in finance and provide analysis

 2 | 🤖 ASSISTANT [AIMessage]
    📞 Tool Calls:
       • write_to_dos({'todos': [...]})
    💭 Reasoning:
       I'll create a research plan and then conduct thorough research...

 3 | 🔧 TOOL [ToolMessage]
    🔧 [write_to_dos] Result:
       Created TODO list with 4 items for financial AI research

 4 | 🤖 ASSISTANT [AIMessage]
    📞 Tool Calls:
       • search_web({'query': 'AI agents financial services 2024'})
    💭 Reasoning:
       Starting research on AI agents in financial services...
```

## Learning Objectives

This implementation teaches:

1. **Deep Agent Architecture**: Understanding the four core components
2. **Sub-Agent Patterns**: Context isolation and specialization
3. **Built-in Tool Usage**: Planning and file system integration
4. **LangGraph State Management**: How state flows between components
5. **Production Patterns**: Rate limiting, error handling, recursion control

## Extending This Example

Consider adding:
- Additional specialized sub-agents (writer, critic, validator)
- Enhanced error handling and recovery
- Middleware integration for logging/monitoring
- Custom state schema for domain-specific tracking
- Integration with external APIs and databases

## Resources

- [Official Deep Agents Documentation](https://docs.langchain.com/labs/deep-agents/overview)
- [Deep Agents GitHub Repository](https://github.com/langchain-ai/deepagents)
- [LangGraph Documentation](https://docs.langchain.com/langgraph)
- [Complete Deep Research Example](https://github.com/langchain-ai/deepagents/tree/main/examples/research)

---

**Note**: This is an educational implementation focused on demonstrating core concepts. For production use, consider the full `deepagents` package with additional features like middleware, advanced state management, and enterprise-grade error handling.