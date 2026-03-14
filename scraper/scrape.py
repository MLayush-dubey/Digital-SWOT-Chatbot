import os
import sys
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAGES = [
    "https://digitalswot.ae/",
    "https://digitalswot.ae/about-us/",
    "https://digitalswot.ae/contact-us/",
    "https://digitalswot.ae/services/",
    "https://digitalswot.ae/services/seo-services/",
    "https://digitalswot.ae/services/social-media-marketing/",
    "https://digitalswot.ae/services/paid-media-marketing/",
    "https://digitalswot.ae/services/growth-marketing/",
    "https://digitalswot.ae/services/content-and-production/",
    "https://digitalswot.ae/services/design-and-animation/",
    "https://digitalswot.ae/services/influencer-marketing/",
    "https://digitalswot.ae/services/affiliate-marketing/",
    "https://digitalswot.ae/services/web-design-and-development/",
    "https://digitalswot.ae/services/programmatic-advertising-with-dv360/",
    "https://digitalswot.ae/services/data-and-crm-management/",
    "https://digitalswot.ae/services/ai-solutions/",
    "https://digitalswot.ae/case-study/",
]

HARDCODED_KNOWLEDGE = """Company: Digital SWOT Marketing L.L.C.
Location: 1008, Grosvenor Business Tower, Tecom, Dubai, UAE
Phone: +971 522737711, 04 558 6320
Email: info@digitalswot.ae
WhatsApp: https://wa.me/971522737711
Working Hours: Monday to Friday, 9:00 AM to 6:00 PM (GST)
Website: https://digitalswot.ae
Social Media: Facebook, Instagram, TikTok, LinkedIn, YouTube, X (Twitter)

Digital SWOT is a full-service digital marketing agency based in Dubai with 10+ years of experience.
They have served 6800+ clients and completed 200+ successful digital projects.
They offer free initial consultations.

Services offered:
1. SEO Services
2. Social Media Marketing
3. Paid Media Marketing (Google Ads, Meta Ads, PPC)
4. Growth Marketing
5. Content & Production
6. Design & Animation
7. Influencer Marketing
8. Affiliate Marketing
9. Web Design & Development
10. Programmatic Advertising with DV360
11. Data & CRM Management
12. AI Solutions (chatbots, NLP, predictive analytics, ML, automation)

Notable clients include: Emaar Real Estate, Binance, Colgate, Valorgi, Immersive, and others.
Case studies available for: Valorgi, Immersive, Emaar Rea
l Estate, Easy Map, Binance, Fooskha, Beirut Street, Colgate, Bougee Cafe, Ophir Properties.
"""

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw"
)


def url_to_slug(url: str) -> str:
    """Convert a URL to a filesystem-safe slug."""
    url = url.rstrip("/")
    path = url.replace("https://digitalswot.ae", "").strip("/")
    if not path:
        return "home"
    slug = path.replace("/", "-")
    return slug


def extract_text(soup: BeautifulSoup) -> str:
    """Remove navigation, footer, sidebar, scripts, and styles; return clean text."""
    for tag in soup.find_all(
        ["nav", "footer", "aside", "script", "style", "noscript", "header"]
    ):
        tag.decompose()

    # Also remove common sidebar/menu class patterns
    for tag in soup.find_all(
        True,
        {"class": lambda c: c and any(
            kw in " ".join(c).lower()
            for kw in ["sidebar", "menu", "widget", "cookie", "popup", "modal", "banner", "breadcrumb"]
        )}
    ):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(id="main") or soup.find(id="content")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    return "\n".join(cleaned_lines).strip()


def get_page_title(soup: BeautifulSoup) -> str:
    """Extract the page title."""
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Unknown"


def scrape_page(url: str) -> dict | None:
    """Scrape a single page and return a dict with metadata and content."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = get_page_title(soup)
    content = extract_text(soup)

    return {
        "url": url,
        "title": title,
        "content": content,
        "scrape_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def save_page(data: dict, slug: str) -> None:
    """Save scraped page data to a .txt file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{slug}.txt")

    header = (
        f"source_url: {data['url']}\n"
        f"page_title: {data['title']}\n"
        f"scrape_date: {data['scrape_date']}\n"
        f"{'=' * 60}\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(data["content"])

    print(f"  [SAVED] {filepath}")


def save_core_info() -> None:
    """Save the hardcoded core knowledge block."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, "core_info.txt")

    header = (
        f"source_url: https://digitalswot.ae\n"
        f"page_title: Digital SWOT Core Information\n"
        f"scrape_date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"{'=' * 60}\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(HARDCODED_KNOWLEDGE)

    print(f"  [SAVED] {filepath}")


def main():
    print(f"Starting scrape. Output directory: {OUTPUT_DIR}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0

    for url in PAGES:
        slug = url_to_slug(url)
        print(f"Scraping: {url}")
        data = scrape_page(url)
        if data:
            save_page(data, slug)
            success_count += 1
        else:
            fail_count += 1

    print("\nSaving hardcoded core info...")
    save_core_info()

    print(f"\nScrape complete. Success: {success_count}, Failed: {fail_count}")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
