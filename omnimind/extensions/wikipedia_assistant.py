import wikipedia
from typing import Dict, Any, List, Optional

EXTENSION_NAME = "Wikipedia Assistant"

class WikipediaAssistant:
    def __init__(self):
        self.search_history = []
        
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search Wikipedia for information"""
        try:
            # Search for pages
            search_results = wikipedia.search(query)
            results = []
            
            # Get summaries for each result
            for title in search_results[:5]:  # Limit to top 5 results
                try:
                    page = wikipedia.page(title)
                    results.append({
                        'title': page.title,
                        'summary': page.summary,
                        'url': page.url
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    results.append({
                        'title': title,
                        'summary': f"Disambiguation: {', '.join(e.options[:5])}",
                        'url': ''
                    })
                except Exception as e:
                    print(f"Error fetching page {title}: {e}")
                    
            self.search_history.append(query)
            return results
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
            
    def get_random_article(self) -> Optional[Dict[str, str]]:
        """Get a random Wikipedia article"""
        try:
            page = wikipedia.random(1)
            article = wikipedia.page(page)
            return {
                'title': article.title,
                'summary': article.summary,
                'url': article.url
            }
        except Exception as e:
            print(f"Error fetching random article: {e}")
            return None
            
    def augment_data(self, data: Any) -> Any:
        """Add Wikipedia context to the data if possible"""
        if isinstance(data, str):
            # Try to find relevant Wikipedia information
            try:
                results = wikipedia.search(data)
                if results:
                    page = wikipedia.page(results[0])
                    return {
                        'original_data': data,
                        'wikipedia_context': {
                            'title': page.title,
                            'summary': page.summary,
                            'url': page.url
                        }
                    }
            except:
                pass
        return data
