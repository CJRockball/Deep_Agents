
# Deep_Agents

**Explore and develop deep agents with LangGraph**
_Last updated: October 2025_

***

## About This Repository

This repo is dedicated to hands-on experimentation with the [LangGraph](https://github.com/langchain-ai/langgraph) agent framework during October–November 2025.
Projects are split into folders, each representing an independent agent implementation, demo, or prototype.

- All code is posted in its current state—raw, refactored, or exploratory.
- The focus is accelerated agent development, tool integration, and solutioning, rather than polished deliverables.

> **Background**:
> Building on my experience as a production data scientist, ML engineer, and agent system experimenter ([see main profile](https://github.com/cjrockball)), I'm using this space for rapid prototyping and conceptual deep dives.
> My wider machine learning journey and skill development are documented in [learning_journey](https://github.com/CJRockball/learning_journey), which shows my evolution from research scientist to full-stack ML system builder.

***

## Current Folder Structure

**Basic_deepagent Project**
This is a minimal, educational implementation of LangGraph's `create_deep_agent` function, designed to showcase the core ideas behind deep agents for complex, multi-step tasks. It demonstrates:

- Planning via automatic TODO lists
- Use of sub-agents for research and reflection (context specialization)
- A virtual file system for persistent memory and state
- Comprehensive system prompts for advanced reasoning
- The agent can search the web (using the Tavily API), plan, research, reflect, and iterate on results, demonstrating architecture and tool integration in a simple, learnable form. This project is best for learning or experimentation with deep agent patterns and LangGraph tooling.

**Agent Project**
This folder contains multiple agent implementations, including the **minimal_research_agent** and **academic_paper_tool** subprojects:

- **minimal_research_agent**: A streamlined LangGraph ReAct agent that tests academic paper search and processing. It takes a research topic and question, finds and processes relevant academic papers, and uses a 9-stage pipeline to answer user queries with citations. It integrates with an academic paper tool for searching, downloading, processing, and querying papers, and provides detailed step-by-step agent reasoning.
- **academic_paper_tool**: (based on folder structure) Appears to be a modular set of scripts and configs for automated academic paper handling, likely supporting downloading, processing, and querying of academic literature, possibly for reuse by research-focused agents.

In summary, the **Agent** folder offers more specialized and modular agent implementations aimed at academic research and workflow automation.


_Folders will expand as additional agent demos and experiments are added. Each represents a standalone investigation into agentic workflows, checkpointing, tool use, or persistent state logic according to the LangGraph paradigm._

***

## Repository Philosophy

- **Growth mindset**: Expect incomplete or unrefined code as a record of practical skill progression and iterative experimentation.
- **Transparency**: Projects are as-is, with evolutionary improvements over time.
- **Documentation**: Inline comments and markdown notes provided where possible; major learnings and design decisions will be chronicled in folder-specific `README`s.

***

## Intended Audience

- ML engineers, agent system developers, and LangGraph enthusiasts seeking code examples, patterns, or inspiration.
- Anyone interested in the application and evolution of agentic AI architectures in Python.

***

## Related Resources

- **Main Profile:** [CJRockball](https://github.com/cjrockball) — background, project links, portfolio
- **Learning/Machine Learning Timeline:** [learning_journey](https://github.com/CJRockball/learning_journey) — showcases Python/ML/AI skill progression, technical milestones, and project phases ([see TIMELINE.md](https://github.com/CJRockball/learning_journey/blob/main/TIMELINE.md) for details)
- **Agent-Lab Companion Repo:** [agent-lab](https://github.com/CJRockball/agent-lab) — reusable patterns and micro-services for agent systems

***

## Contact

For inquiries, collaborations, or feedback:

- [GitHub profile](https://github.com/cjrockball)
- [LinkedIn/in/patrick-cj-carlberg](https://www.linkedin.com/in/patrick-cj-carlberg/)

***

_You are welcome to use, contribute to, fork, or reference the materials herein in accordance with the repository license._
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: https://github.com/CJRockball/Deep_Agents


