from bs4 import BeautifulSoup

def clean_html(raw_html: str) -> str:
    """Removes HTML tags from text."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ").strip()