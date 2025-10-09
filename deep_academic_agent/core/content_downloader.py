#%%
# content_downloader_fixed.py
import requests
import os
import time
import hashlib
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import sqlite3
from bs4 import BeautifulSoup, Comment  # Fixed import
import re

# Using the existing ArxivPaper from your code
from .arxiv_search import ArxivPaper
from .database_setup import DatabaseManager

@dataclass
class DownloadResult:
    """Result of content download operation"""
    arxiv_id: str
    success: bool
    content_type: str  # 'html', 'pdf', 'error'
    file_path: Optional[str] = None
    content_size: int = 0
    download_time: float = 0.0
    error_message: Optional[str] = None
    content_hash: Optional[str] = None
    raw_content: Optional[str] = None  # Store raw HTML content


class ContentDownloader:
    """
    Handles downloading and storing HTML/PDF content from ArXiv papers
    with robust error handling, caching, and format validation
    """
    
    def __init__(self, 
                 storage_path: str = "data/papers",
                 db_manager: Optional[DatabaseManager] = None,
                 rate_limit_delay: float = 1.0):
        
        self.storage_path = Path(storage_path)
        self.db_manager = db_manager or DatabaseManager()
        self.rate_limit_delay = rate_limit_delay
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "html").mkdir(exist_ok=True)
        (self.storage_path / "pdf").mkdir(exist_ok=True)
        (self.storage_path / "raw").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure requests session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ArXiv-Academic-Analyzer/1.0 (research-paper-analysis)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def download_html_content(self, paper: ArxivPaper) -> DownloadResult:
        """
        Download HTML content for a paper with comprehensive error handling
        
        Args:
            paper: ArxivPaper object with html_url populated
            
        Returns:
            DownloadResult with success status and file information
        """
        start_time = time.time()
        
        if not paper.html_url:
            return DownloadResult(
                arxiv_id=paper.id,
                success=False,
                content_type='error',
                error_message='No HTML URL available'
            )
        
        try:
            self.logger.info(f"Downloading HTML content for paper {paper.id}")
            
            # Check if already downloaded and cached
            cached_result = self._check_cache(paper.id, 'html')
            if cached_result:
                self.logger.info(f"Using cached HTML content for {paper.id}")
                return cached_result
            
            # Apply rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Download the HTML content
            response = self.session.get(paper.html_url, timeout=30)
            response.raise_for_status()
            
            # Validate that we got HTML content
            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type and 'xml' not in content_type:
                return DownloadResult(
                    arxiv_id=paper.id,
                    success=False,
                    content_type='error',
                    error_message=f'Expected HTML, got {content_type}'
                )
            
            # Clean and validate HTML content
            html_content = response.text
            if not self._validate_arxiv_html(html_content):
                return DownloadResult(
                    arxiv_id=paper.id,
                    success=False,
                    content_type='error',
                    error_message='Invalid or incomplete ArXiv HTML content'
                )
            
            # Generate content hash for integrity checking
            content_hash = hashlib.sha256(html_content.encode('utf-8')).hexdigest()
            
            # Save raw HTML content
            raw_file_path = self.storage_path / "raw" / f"{paper.id}_raw.html"
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Clean and structure the HTML
            cleaned_html = self._clean_html_content(html_content)
            
            # Save cleaned HTML content
            html_file_path = self.storage_path / "html" / f"{paper.id}.html"
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_html)
            
            download_time = time.time() - start_time
            
            result = DownloadResult(
                arxiv_id=paper.id,
                success=True,
                content_type='html',
                file_path=str(html_file_path),
                content_size=len(html_content),
                download_time=download_time,
                content_hash=content_hash,
                raw_content=html_content
            )
            
            # Save download metadata to database
            self._save_download_metadata(result)
            
            self.logger.info(f"Successfully downloaded HTML for {paper.id} ({len(html_content)} bytes)")
            return result
            
        except requests.RequestException as e:
            error_msg = f"Network error downloading {paper.id}: {str(e)}"
            self.logger.error(error_msg)
            return DownloadResult(
                arxiv_id=paper.id,
                success=False,
                content_type='error',
                download_time=time.time() - start_time,
                error_message=error_msg
            )
        
        except Exception as e:
            error_msg = f"Unexpected error downloading {paper.id}: {str(e)}"
            self.logger.error(error_msg)
            return DownloadResult(
                arxiv_id=paper.id,
                success=False,
                content_type='error',
                download_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def _validate_arxiv_html(self, html_content: str) -> bool:
        """
        Validate that HTML content is a proper ArXiv paper
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            True if valid ArXiv HTML content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check for ArXiv-specific elements
            arxiv_indicators = [
                soup.find('meta', {'name': 'citation_arxiv_id'}),
                soup.find('div', class_='ltx_page_main'),
                soup.find('article', class_='ltx_document'),
                'arxiv.org' in html_content.lower(),
                soup.find('title') and len(soup.find('title').get_text().strip()) > 0
            ]
            
            # Must have at least 2 indicators for valid content
            valid_indicators = sum(1 for indicator in arxiv_indicators if indicator)
            
            # Check minimum content length
            text_content = soup.get_text().strip()
            min_content_length = 1000  # Reasonable minimum for academic paper
            
            return valid_indicators >= 2 and len(text_content) >= min_content_length
            
        except Exception as e:
            self.logger.warning(f"HTML validation error: {e}")
            return False
    
    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean and standardize HTML content for processing
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned HTML content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unnecessary elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Remove comments - FIXED: Use proper Comment import
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Clean up whitespace
            cleaned_html = str(soup.prettify())
            
            # Normalize whitespace but preserve structure
            cleaned_html = re.sub(r'\n\s*\n', '\n\n', cleaned_html)
            cleaned_html = re.sub(r'[ \t]+', ' ', cleaned_html)
            
            return cleaned_html
            
        except Exception as e:
            self.logger.warning(f"HTML cleaning error: {e}")
            return html_content
    
    def _check_cache(self, arxiv_id: str, content_type: str) -> Optional[DownloadResult]:
        """
        Check if content is already cached and valid
        
        Args:
            arxiv_id: ArXiv paper ID
            content_type: Type of content ('html' or 'pdf')
            
        Returns:
            DownloadResult if cached, None otherwise
        """
        file_extension = '.html' if content_type == 'html' else '.pdf'
        file_path = self.storage_path / content_type / f"{arxiv_id}{file_extension}"
        
        if file_path.exists():
            try:
                file_size = file_path.stat().st_size
                
                # Check database for metadata
                conn = self.db_manager.get_sqlite_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT content_hash, download_time FROM content_downloads 
                    WHERE arxiv_id = ? AND content_type = ? AND success = 1
                ''', (arxiv_id, content_type))
                
                result = cursor.fetchone()
                if result:
                    content_hash, download_time = result
                    return DownloadResult(
                        arxiv_id=arxiv_id,
                        success=True,
                        content_type=content_type,
                        file_path=str(file_path),
                        content_size=file_size,
                        download_time=download_time,
                        content_hash=content_hash
                    )
                
            except Exception as e:
                self.logger.warning(f"Cache check error for {arxiv_id}: {e}")
        
        return None
    
    def _save_download_metadata(self, result: DownloadResult):
        """
        Save download metadata to database - FIXED: Handle datetime properly
        
        Args:
            result: DownloadResult to save
        """
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            # Create content_downloads table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_downloads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id VARCHAR(50),
                    content_type VARCHAR(10),
                    success BOOLEAN,
                    file_path TEXT,
                    content_size INTEGER,
                    download_time REAL,
                    content_hash TEXT,
                    error_message TEXT,
                    timestamp TEXT,  -- Changed to TEXT for explicit control
                    UNIQUE(arxiv_id, content_type)
                )
            ''')
            
            # Insert or update download record - FIXED: Convert datetime to string
            timestamp_str = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO content_downloads
                (arxiv_id, content_type, success, file_path, content_size, 
                 download_time, content_hash, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.arxiv_id,
                result.content_type,
                result.success,
                result.file_path,
                result.content_size,
                result.download_time,
                result.content_hash,
                result.error_message,
                timestamp_str  # Use string instead of datetime object
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving download metadata: {e}")
    
    def download_papers_batch(self, papers: List[ArxivPaper]) -> List[DownloadResult]:
        """
        Download multiple papers with batch processing and error handling
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of DownloadResult objects
        """
        results = []
        total_papers = len(papers)
        
        self.logger.info(f"Starting batch download of {total_papers} papers")
        
        for i, paper in enumerate(papers, 1):
            self.logger.info(f"Processing paper {i}/{total_papers}: {paper.id}")
            
            if paper.html_url:
                result = self.download_html_content(paper)
                results.append(result)
                
                # Update processing status in database
                self._update_processing_status(paper.id, 'content_downloaded', result.success)
            else:
                result = DownloadResult(
                    arxiv_id=paper.id,
                    success=False,
                    content_type='error',
                    error_message='No HTML URL available'
                )
                results.append(result)
                self._update_processing_status(paper.id, 'content_download_failed', False)
        
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"Batch download completed: {success_count}/{total_papers} successful")
        
        return results
    
    def _update_processing_status(self, arxiv_id: str, stage: str, success: bool):
        """Update processing status in database - FIXED: Handle datetime properly"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            # Convert datetime to string to avoid deprecation warning
            timestamp_str = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO processing_log (paper_id, processing_stage, status, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (arxiv_id, stage, 'success' if success else 'failed', timestamp_str))
            
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating processing status: {e}")
    
    def get_download_stats(self) -> Dict[str, any]:
        """Get download statistics from database"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    content_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(download_time) as avg_download_time,
                    SUM(content_size) as total_size
                FROM content_downloads 
                GROUP BY content_type
            ''')
            
            stats = {}
            for row in cursor.fetchall():
                content_type, total, successful, avg_time, total_size = row
                stats[content_type] = {
                    'total': total,
                    'successful': successful,
                    'success_rate': (successful / total * 100) if total > 0 else 0,
                    'avg_download_time': round(avg_time or 0, 2),
                    'total_size_mb': round((total_size or 0) / 1024 / 1024, 2)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting download stats: {e}")
            return {}


# Usage example
if __name__ == "__main__":
    # Import your existing modules
    from arxiv_search import arxiv_search
    from deep_academic_agent.core.paper_selection import select_papers_for_analysis, SelectionCriteria, SelectionStrategy
    
    # Initialize downloader
    downloader = ContentDownloader()
    
    # Example with mock data (replace with your actual search)
    search_results = arxiv_search("fintech laggraph agent", max_results=20, start=20, fetch_html=True)
    criteria = SelectionCriteria(max_papers=3, strategy=SelectionStrategy.HTML_ONLY)
    selected_papers = select_papers_for_analysis(search_results, criteria)
    
    # Download content
    download_results = downloader.download_papers_batch(selected_papers)

# %%
