"""Browser tools.

search_web opens the default browser when ENABLE_REAL_WINDOWS_TOOLS=true;
otherwise it returns the URL it would open. Playwright-based automation is a
later phase (see requirements.txt / docs/ROADMAP.md).
"""

import webbrowser
from typing import Any
from urllib.parse import quote_plus

from app.config import real_windows_tools_enabled
from app.schemas.commands import SafetyLevel
from app.tools.registry import registry


@registry.register(
    name="search_web",
    description="Search the web in the default browser",
    safety_level=SafetyLevel.SAFE,
)
def search_web(query: str = "") -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"action": "search_web", "error": "Empty search query."}
    url = f"https://duckduckgo.com/?q={quote_plus(query)}"
    if real_windows_tools_enabled():
        webbrowser.open(url)
        return {"simulated": False, "action": "search_web", "opened": url}
    return {
        "simulated": True,
        "action": "search_web",
        "would_do": f"Open the default browser and search for {query!r}",
        "url": url,
    }
