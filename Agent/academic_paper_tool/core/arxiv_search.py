#%%

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import urllib.parse
import re

@dataclass
class ArxivPaper:
    """Data structure for arXiv paper information"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    abstract_url: str
    html_url: Optional[str] = None  # URL to the actual HTML paper (if available)

def get_html_paper_url(arxiv_id: str, abstract_url: str) -> Optional[str]:
    """
    Check if HTML version exists for a paper by scraping the abstract page
    
    Args:
        arxiv_id: arXiv ID of the paper
        abstract_url: URL to the abstract page
    
    Returns:
        URL to HTML version if available, None otherwise
    """
    try:
        # Request the abstract page
        response = requests.get(abstract_url, timeout=10)
        response.raise_for_status()
        
        # Look for HTML link pattern in the response
        # arXiv HTML links follow the pattern: https://arxiv.org/html/{arxiv_id}
        html_pattern = rf'https://arxiv\.org/html/{re.escape(arxiv_id)}(?:v\d+)?'
        html_match = re.search(html_pattern, response.text)
        
        if html_match:
            return html_match.group(0)
        
        # Also check for the direct HTML link text pattern
        if 'HTML</a>' in response.text or 'view html' in response.text.lower():
            # Construct the HTML URL based on arXiv's standard pattern
            return f"https://arxiv.org/html/{arxiv_id}"
        
        return None
        
    except Exception as e:
        print(f"Error checking for HTML version of {arxiv_id}: {e}")
        return None

def arxiv_search(query: str, max_results: int = 10, start: int = 0, fetch_html: bool = True) -> List[ArxivPaper]:
    """
    Search arXiv using their API and optionally fetch HTML links
    
    Args:
        query: Search query (e.g., "machine learning", "ti:neural networks")
        max_results: Maximum number of results to return (default: 10)
        start: Start index for pagination (default: 0)
        fetch_html: Whether to check for HTML versions (default: True, adds latency)
    
    Returns:
        List of ArxivPaper objects containing paper information
    """
    
    # arXiv API base URL
    base_url = "http://export.arxiv.org/api/query"
    
    # URL encode the query
    encoded_query = urllib.parse.quote_plus(query)
    
    # Build the API request URL
    url = f"{base_url}?search_query={encoded_query}&start={start}&max_results={max_results}"
    
    try:
        # Make the request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the XML response
        root = ET.fromstring(response.content)
        
        # Define namespaces for XML parsing
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        
        # Extract entries from the feed
        for entry in root.findall('atom:entry', namespaces):
            # Extract basic information
            arxiv_id = entry.find('atom:id', namespaces).text.split('/')[-1]
            title = entry.find('atom:title', namespaces).text.strip()
            abstract = entry.find('atom:summary', namespaces).text.strip()
            
            # Parse dates
            published = datetime.fromisoformat(entry.find('atom:published', namespaces).text.replace('Z', '+00:00'))
            updated = datetime.fromisoformat(entry.find('atom:updated', namespaces).text.replace('Z', '+00:00'))
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = author.find('atom:name', namespaces).text
                authors.append(name)
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                categories.append(category.get('term'))
            
            # Find PDF URL and abstract URL
            pdf_url = None
            abstract_url = None
            
            for link in entry.findall('atom:link', namespaces):
                link_rel = link.get('rel')
                link_type = link.get('type')
                link_title = link.get('title')
                
                if link_title == 'pdf':
                    pdf_url = link.get('href')
                elif link_rel == 'alternate' and link_type == 'text/html':
                    abstract_url = link.get('href')
            
            # Get HTML paper URL if requested
            html_url = None
            if fetch_html and abstract_url:
                html_url = get_html_paper_url(arxiv_id, abstract_url)
            
            # Create paper object
            paper = ArxivPaper(
                id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                updated=updated,
                categories=categories,
                pdf_url=pdf_url,
                abstract_url=abstract_url,
                html_url=html_url
            )
            
            papers.append(paper)
        
        return papers
        
    except requests.RequestException as e:
        print(f"Error making request to arXiv API: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Search with HTML link fetching
    results = arxiv_search("fintech laggraph agent", max_results=20, fetch_html=True)
    
    for paper in results:
        if paper.html_url != None:
            print(f"Title: {paper.title}")
            print(f"PDF: {paper.pdf_url}")
            print(f"Abstract: {paper.abstract_url}")
            print(f"HTML: {paper.html_url if paper.html_url else 'Not available'}")
            print("-" * 50)
