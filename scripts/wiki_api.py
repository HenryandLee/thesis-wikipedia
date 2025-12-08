"""
Utility module for interacting with Wikipedia/MediaWiki API
"""
import requests
import time
import json
from typing import Dict, List, Optional, Generator
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaAPI:
    """Interface for MediaWiki API"""
    
    def __init__(self, language='zh', user_agent='WikiResearch/1.0 (Educational Research; Contact: your@email.edu)'):
        """
        Initialize API client
        
        Args:
            language: Wikipedia language code (e.g., 'zh' for Chinese)
            user_agent: User agent string (recommended to customize with your contact info)
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.rate_limit_delay = 0.1  # Conservative: 0.1 second between requests
        
    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting
        
        Args:
            params: API parameters
            
        Returns:
            JSON response
        """
        params['format'] = 'json'
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_page_info(self, page_title: str) -> Optional[Dict]:
        """
        Get basic information about a page
        
        Args:
            page_title: Title of the Wikipedia page
            
        Returns:
            Dictionary with page information or None if page doesn't exist
        """
        params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'info|pageprops',
            'inprop': 'url|displaytitle',
        }
        
        data = self._make_request(params)
        pages = data.get('query', {}).get('pages', {})
        
        for page_id, page_data in pages.items():
            if page_id == '-1':
                logger.warning(f"Page '{page_title}' does not exist")
                return None
            return {
                'pageid': page_data.get('pageid'),
                'title': page_data.get('title'),
                'url': page_data.get('fullurl'),
                'exists': True
            }
        return None
    
    def get_revisions(self, page_title: str, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, limit: int = 500) -> Generator[Dict, None, None]:
        """
        Get revision history for a page (with continuation support)
        
        Args:
            page_title: Title of the Wikipedia page
            start_date: Start date in ISO format (e.g., '2020-01-01T00:00:00Z')
            end_date: End date in ISO format
            limit: Maximum revisions per request (max 500)
            
        Yields:
            Dictionary for each revision
        """
        params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'revisions',
            'rvprop': 'ids|timestamp|user|userid|size|comment|flags|tags',
            'rvlimit': min(limit, 500),  # API maximum is 500
            'rvdir': 'newer',  # Oldest first
        }
        
        if start_date:
            params['rvstart'] = start_date
        if end_date:
            params['rvend'] = end_date
        
        continue_params = {}
        
        while True:
            current_params = {**params, **continue_params}
            data = self._make_request(current_params)
            
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    logger.warning(f"Page '{page_title}' does not exist")
                    return
                
                revisions = page_data.get('revisions', [])
                for rev in revisions:
                    yield {
                        'revid': rev.get('revid'),
                        'parentid': rev.get('parentid'),
                        'timestamp': rev.get('timestamp'),
                        'user': rev.get('user', 'Anonymous'),
                        'userid': rev.get('userid', 0),
                        'size': rev.get('size', 0),
                        'comment': rev.get('comment', ''),
                        'minor': 'minor' in rev,
                        'tags': rev.get('tags', [])
                    }
            
            # Check for continuation
            if 'continue' not in data:
                break
            
            continue_params = data['continue']
            logger.info(f"Continuing pagination for '{page_title}'...")
    
    def get_page_views(self, page_title: str, start_date: str, end_date: str, 
                       granularity: str = 'daily') -> List[Dict]:
        """
        Get page view statistics (uses Wikimedia REST API)
        
        Args:
            page_title: Title of the Wikipedia page
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            granularity: 'daily' or 'monthly'
            
        Returns:
            List of page view data
        """
        # Encode title for URL
        encoded_title = requests.utils.quote(page_title.replace(' ', '_'))
        
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"{self.language}.wikipedia/all-access/all-agents/{encoded_title}/"
               f"{granularity}/{start_date}/{end_date}")
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get page views for '{page_title}': {e}")
            return []
    
    def search_pages(self, search_term: str, limit: int = 50) -> List[str]:
        """
        Search for pages matching a term
        
        Args:
            search_term: Search query
            limit: Maximum number of results
            
        Returns:
            List of page titles
        """
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': search_term,
            'srlimit': min(limit, 500),
        }
        
        data = self._make_request(params)
        search_results = data.get('query', {}).get('search', [])
        
        return [result['title'] for result in search_results]
    
    def get_category_members(self, category: str, limit: int = 500) -> List[str]:
        """
        Get all pages in a category
        
        Args:
            category: Category name (e.g., 'Category:台灣政治')
            limit: Maximum number of results
            
        Returns:
            List of page titles
        """
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmlimit': min(limit, 500),
            'cmtype': 'page',  # Only get pages, not subcategories
        }
        
        members = []
        continue_params = {}
        
        while len(members) < limit:
            current_params = {**params, **continue_params}
            data = self._make_request(current_params)
            
            category_members = data.get('query', {}).get('categorymembers', [])
            members.extend([member['title'] for member in category_members])
            
            if 'continue' not in data or len(members) >= limit:
                break
                
            continue_params = data['continue']
        
        return members[:limit]
    
    def get_page_links(self, page_title: str, limit: int = 500) -> List[str]:
        """
        Get all links from a page
        
        Args:
            page_title: Title of the Wikipedia page
            limit: Maximum number of links to return
            
        Returns:
            List of linked page titles
        """
        params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'links',
            'pllimit': min(limit, 500),
            'plnamespace': 0,  # Only main namespace (articles)
        }
        
        links = []
        continue_params = {}
        
        while len(links) < limit:
            current_params = {**params, **continue_params}
            data = self._make_request(current_params)
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id != '-1':
                    page_links = page_data.get('links', [])
                    links.extend([link['title'] for link in page_links])
            
            if 'continue' not in data or len(links) >= limit:
                break
                
            continue_params = data['continue']
        
        return links[:limit]

    def get_user_groups(self, usernames: List[str], show_progress: bool = True) -> Dict[str, List[str]]:
        """
        Get user groups for a list of usernames (to identify bots)
        
        Bots belong to the 'bot' user group. This is the definitive way
        to identify bot accounts on Wikipedia.
        
        Args:
            usernames: List of Wikipedia usernames to check
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping username -> list of groups
            Example: {'ClueBot NG': ['bot', 'user'], 'John': ['user']}
        """
        result = {}
        
        # API allows up to 50 users per request
        batch_size = 50
        total_batches = (len(usernames) + batch_size - 1) // batch_size
        
        # Create iterator with optional progress bar
        batch_ranges = range(0, len(usernames), batch_size)
        if show_progress:
            batch_ranges = tqdm(batch_ranges, total=total_batches, 
                               desc="Querying user groups", unit="batch")
        
        for i in batch_ranges:
            batch = usernames[i:i + batch_size]
            
            # Filter out empty/anonymous usernames
            batch = [u for u in batch if u and u != 'Anonymous']
            if not batch:
                continue
            
            params = {
                'action': 'query',
                'list': 'users',
                'ususers': '|'.join(batch),
                'usprop': 'groups'
            }
            
            try:
                data = self._make_request(params)
                users = data.get('query', {}).get('users', [])
                
                for user in users:
                    name = user.get('name')
                    groups = user.get('groups', [])
                    if name:
                        result[name] = groups
                        
            except Exception as e:
                logger.warning(f"Failed to get user groups for batch: {e}")
                # Continue with next batch instead of failing entirely
                continue
        
        return result

    def identify_bots(self, usernames: List[str]) -> set:
        """
        Identify which usernames are bots
        
        Args:
            usernames: List of Wikipedia usernames to check
            
        Returns:
            Set of usernames that are bots
        """
        user_groups = self.get_user_groups(usernames)
        
        bots = set()
        for username, groups in user_groups.items():
            if 'bot' in groups:
                bots.add(username)
        
        return bots


def save_json(data, filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved data to {filepath}")


def load_json(filepath: str):
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

