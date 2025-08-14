import wikipedia
from datetime import datetime
from typing import List, Dict, Optional

wikipedia.set_lang('en')

def wiki_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Return top_k wikipedia pages as dicts with title, url, summary, content, crawl_date.
    """
    
    pages = []
    
    try:
        results = wikipedia.search(query, results=top_k)
    except Exception:
        results = []

    for title in results:
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            pages.append({
                "title": page.title,
                "url": page.url,
                "summary": page.summary,
                "content": page.content,
                "crawl_date": datetime.now().isoformat()
            })
        except Exception:
            continue

    return pages
    
