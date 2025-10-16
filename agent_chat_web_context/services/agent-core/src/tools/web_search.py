# services/agent-core/src/tools/web_search.py

import logging
import os
import uuid
import requests
from typing import List, Dict, Any

from markdownify import markdownify as md
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from config import TAVILY_KEY

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to offload full content
OFFLOAD_DIR = "offloaded_pages"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Lightweight summarization model
summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def run_tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform the web search using Tavily API.
    Returns a list of result dicts with 'title' and 'url'.
    """
    logger.info("Running Tavily search for: %s", query)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": max_results}
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        simplified = [{"title": r.get("title"), "url": r.get("url")} for r in results]
        logger.info("Found %d results", len(simplified))
        return simplified
    except Exception as e:
        logger.error('Error in Tavily Search: %s', e)
        return []
    
    
def summarize_webpage_content(html: str) -> Dict[str, str]:
    """
    Generate structured summary of webpage content.
    Returns dict with 'filename' and 'summary'.
    """
    try:
        # Convert HTML to Markdown
        markdown = md(html)
        
        # Build prompt as string
        prompt = (
            "You are a concise summarizer. "
            "Given the following Markdown content, produce:\n"
            "1. A descriptive filename (no extension, max 50 chars).\n"
            "2. A brief 'Key Learnings' summary of 3-5 bullet points.\n\n"
            f"Content:\n{markdown[:2000]}\n\n"
            "Format your response as:\n"
            "Filename: your-filename\n"
            "Key Learnings:\n- Point 1\n- Point 2\n..."
        )
        
        logger.info("Summarizing webpage content")
        
        # Invoke LLM
        response = summary_llm.invoke([{"role": "user", "content": prompt}])
        
        # Extract content - handle different response types
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Ensure content is a string
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        
        # Parse the response
        parts = content.split("Key Learnings:")
        filename_line = parts[0].replace("Filename:", "").strip()
        summary = parts[1].strip() if len(parts) > 1 else "No summary provided"
        
        # Sanitize filename
        filename_line = filename_line[:50]  # Limit length
        if not filename_line or filename_line == "":
            filename_line = str(uuid.uuid4())[:8]
        
        return {"filename": filename_line, "summary": summary}
        
    except Exception as e:
        logger.error("Error summarizing content: %s", e)
        return {
            "filename": str(uuid.uuid4())[:8], 
            "summary": "Summary unavailable due to error"
        }

def process_search_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    For each search result:
    1. Fetch full HTML
    2. Convert to Markdown
    3. Summarize content
    4. Save full Markdown to disk
    Returns a list of minimal summaries.
    """
    summaries = []
    for idx, item in enumerate(results, 1):
        url = item["url"]
        try:
            logger.info("Fetching URL %d: %s", idx, url)
            html = requests.get(url, timeout=10).text
            info = summarize_webpage_content(html)
            # Save full Markdown to file
            filename = f"{info['filename'][:50] or str(uuid.uuid4())}.md"
            filename = filename.replace("/", "-").replace(" ", "_")  # Sanitize
            
            path = os.path.join(OFFLOAD_DIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(md(html))
                
            logger.info("Offloaded full content to %s", path)
            summaries.append({
                "title": item["title"],
                "url": url,
                "snippet": info["summary"],
                "offload_path": path
            })
        except Exception as e:
            logger.error("Error processing %s: %s", url, e)
            summaries.append({
                "title": item["title"],
                "url": url,
                "snippet": "Content unavailable",
                "offload_path": None
            })
    return summaries

@tool
def tavily_search_tool(query: str, max_results: int = 5) -> Command:
    """
    Main Tavily search tool for LangGraph ReAct agent.
    - Executes search
    - Processes results (fetch + summarize + offload)
    - Returns minimal summaries to agent
    - Emits commands to record offloaded file paths in agent state
    """
    logger.info("Executing tavily_search_tool for query: %s", query)
    
    # 1. Run search
    raw_results = run_tavily_search(query, max_results)
    
    if not raw_results:
        return "No search results found."
    
    # 2. Process results (fetch, summarize, offload)
    processed = process_search_results(raw_results)
    
    # 3. Format as string
    summary_texts = []
    for i, item in enumerate(processed, 1):
        summary_texts.append(
            f"{i}. **{item['title']}**\n"
            f"   {item['snippet']}\n"
            f"   URL: {item['url']}"
        )
    
    agent_message = "\n\n".join(summary_texts) or "No results found."
    
    return agent_message
