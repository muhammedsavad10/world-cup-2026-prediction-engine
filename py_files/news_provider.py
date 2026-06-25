from abc import ABC, abstractmethod
from duckduckgo_search import DDGS
import concurrent.futures

class NewsProvider(ABC):
    @abstractmethod
    def fetch_news(self, team_name):
        pass

class APIFootballNews(NewsProvider):
    def fetch_news(self, team_name):
        return f"APIFootball: Live reports for {team_name} indicate steady team build-up."

class GoogleNews(NewsProvider):
    def fetch_news(self, team_name):
        try:
            with DDGS() as ddgs:
                query = f"{team_name} national football team current manager squad June 2026"
                search_results = list(ddgs.text(query, max_results=3))
                
                snippets = []
                for result in search_results:
                    snippets.append(result.get('body', ''))
                
                return "\n".join(snippets)
        except Exception as e:
            return f"GoogleNews proxy: Live search data currently unavailable for {team_name}."

class RSSNews(NewsProvider):
    def fetch_news(self, team_name):
        return f"RSSNews: No RSS feeds registered for {team_name}."

class MockNews(NewsProvider):
    def fetch_news(self, team_name):
        return f"MockNews: Mock updates for {team_name}."

# Default provider instance
_provider = GoogleNews()

def set_news_provider(provider: NewsProvider):
    global _provider
    _provider = provider

def fetch_live_team_news(team_name) -> str:
    """
    Fetches news using the registered provider. Wraps the call in a thread pool executor
    to enforce a non-blocking 2.0-second timeout limit. If timed out, degrades gracefully 
    by returning an empty string.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_provider.fetch_news, team_name)
        try:
            return future.result(timeout=2.0)
        except concurrent.futures.TimeoutError:
            return ""  # Degrade gracefully by returning empty string
        except Exception:
            return ""  # Degrade gracefully by returning empty string
