# tools/web_search/processors.py
"""
Content processors for web search results - FULL IMPLEMENTATION
Handles HTML fetching, markdown conversion, LLM summarization, and content offloading
"""

import os
import uuid
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from markdownify import markdownify as md
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def process_search_results(
    results: List[Dict[str, Any]],
    enable_summarization: bool = True,
    enable_offload: bool = True,
    offload_dir: str = "./data/offloaded_pages",
    llm_model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process raw search results: fetch full content, convert to markdown, summarize, offload.
    
    Args:
        results: Raw search results from Tavily
        enable_summarization: Whether to use LLM to summarize content
        enable_offload: Whether to save full content to disk
        offload_dir: Directory to save offloaded content
        llm_model: LLM model to use for summarization
        openai_api_key: OpenAI API key for summarization
        
    Returns:
        List of processed results with summaries and offload paths
    """
    processed = []
    
    # Create offload directory
    if enable_offload:
        Path(offload_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Offload directory ready: {offload_dir}")
    
    # Initialize LLM if summarization enabled
    llm = None
    if enable_summarization and openai_api_key:
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0.2,
            api_key=openai_api_key
        )
    
    for idx, result in enumerate(results, 1):
        url = result.get("url", "")
        title = result.get("title", "No title")
        tavily_content = result.get("content", "")
        
        logger.info(f"Processing result {idx}/{len(results)}: {url}")
        
        try:
            # Try to fetch full HTML content
            html_content = fetch_page_content(url)
            
            # Convert to markdown
            markdown_content = html_to_markdown(html_content)
            
            # Generate summary
            if llm and enable_summarization:
                summary_info = summarize_with_llm(
                    markdown_content,
                    title,
                    llm
                )
                snippet = summary_info["summary"]
                suggested_filename = summary_info["filename"]
            else:
                # Fallback to Tavily content or truncated markdown
                snippet = tavily_content if tavily_content else markdown_content[:300] + "..."
                suggested_filename = sanitize_filename(title)
            
            # Offload full content
            offload_path = None
            if enable_offload and markdown_content:
                offload_path = offload_content(
                    markdown_content,
                    suggested_filename,
                    offload_dir,
                    url
                )
            
            processed.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "content": markdown_content,
                "offload_path": offload_path,
                "score": result.get("score", 0.0)
            })
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            # Fallback to Tavily data
            processed.append({
                "title": title,
                "url": url,
                "snippet": tavily_content or "Content unavailable",
                "content": tavily_content or "",
                "offload_path": None,
                "score": result.get("score", 0.0)
            })
    
    logger.info(f"✓ Processed {len(processed)} results successfully")
    return processed


def fetch_page_content(url: str, timeout: int = 10) -> str:
    """
    Fetch HTML content from a URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        HTML content as string
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    logger.debug(f"Fetched {len(response.text)} chars from {url}")
    return response.text


def html_to_markdown(html: str) -> str:
    """
    Convert HTML to markdown format.
    
    Args:
        html: HTML content
        
    Returns:
        Markdown formatted content
    """
    # Convert with markdownify
    markdown = md(html, heading_style="ATX")
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in markdown.split('\n')]
    markdown = '\n'.join(line for line in lines if line)
    
    return markdown


def summarize_with_llm(
    content: str,
    title: str,
    llm: ChatOpenAI
) -> Dict[str, str]:
    """
    Summarize webpage content using LLM.
    
    Args:
        content: Full markdown content
        title: Page title
        llm: ChatOpenAI instance
        
    Returns:
        Dictionary with 'summary' and 'filename' keys
    """
    # Truncate content for LLM (first 3000 chars)
    truncated = content[:3000]
    
    prompt = f"""You are an expert content summarizer. Analyze this webpage content and provide:

1. A concise summary (2-4 sentences) highlighting the most important information
2. A descriptive filename (max 50 characters, alphanumeric and dashes only, no extension)

Title: {title}

Content:
{truncated}

Respond in this exact format:
FILENAME: your-suggested-filename
SUMMARY: Your concise summary here"""
    
    try:
        response = llm.invoke(prompt)
        content_str = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response
        lines = content_str.split('\n')
        filename = ""
        summary = ""
        
        for line in lines:
            if line.startswith("FILENAME:"):
                filename = line.replace("FILENAME:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
        
        # Fallback if parsing fails
        if not filename:
            filename = sanitize_filename(title)
        if not summary:
            summary = truncated[:200] + "..."
        
        logger.debug(f"LLM summary generated for: {title}")
        
        return {
            "filename": filename[:50],  # Enforce max length
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        return {
            "filename": sanitize_filename(title),
            "summary": content[:200] + "..."
        }


def offload_content(
    content: str,
    filename: str,
    directory: str,
    url: str
) -> str:
    """
    Save full content to disk with metadata.
    
    Args:
        content: Content to save
        filename: Base filename
        directory: Directory to save in
        url: Source URL
        
    Returns:
        Path to saved file
    """
    # Ensure filename is safe
    safe_filename = sanitize_filename(filename)
    
    # Add hash to prevent collisions
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    final_filename = f"{safe_filename}_{url_hash}.md"
    
    filepath = os.path.join(directory, final_filename)
    
    # Create content with metadata header
    full_content = f"""---
source: {url}
filename: {filename}
saved: {__import__('datetime').datetime.utcnow().isoformat()}
---

{content}
"""
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    logger.info(f"✓ Offloaded content to: {filepath}")
    return filepath


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be filesystem-safe.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Keep only alphanumeric, spaces, hyphens, underscores
    safe = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_'))
    
    # Replace spaces with hyphens
    safe = safe.replace(' ', '-')
    
    # Remove consecutive hyphens
    while '--' in safe:
        safe = safe.replace('--', '-')
    
    # Truncate to 50 chars
    safe = safe[:50].strip('-')
    
    # Fallback if empty
    if not safe:
        safe = str(uuid.uuid4())[:8]
    
    return safe
