import csv
from playwright.sync_api import sync_playwright

# helper function to retrieve the 5 trending stocks via Yahoo Finanace
def get_trending_tickers(page, limit=5):
    page.goto("https://finance.yahoo.com/trending-tickers", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector("table tbody tr", timeout=15000)

    rows = page.query_selector_all("table tbody tr")
    tickers = []
    for row in rows[:limit]:
        cell = row.query_selector("td")
        if cell:
            tickers.append(cell.inner_text().strip())
    return tickers

# helper function to scrape the news titles and article links
def scrape_yahoo_news(page, tickers):

    articles_data = []

    for ticker in tickers:
        print(f"Scraping {ticker} news")
        news_url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        page.goto(news_url, wait_until = "domcontentloaded", timeout=120000)
        try:
            page.wait_for_selector("li.stream-item.story-item", timeout=10000)
            articles = page.query_selector_all("li.stream-item.story-item") # articles are associated with "stream-item story-item" class as per the html

            # Take at most 10
            for article in articles[:10]:
                title_el = article.query_selector("h3")
                link_el = article.query_selector("a")
                source_el = article.query_selector("publishing yf-m1e6lz")

                if title_el and link_el:
                    articles_data.append({
                        "ticker": ticker,
                        "title": title_el.inner_text(),
                        "url": link_el.get_attribute("href"),
                        "source": source_el.inner_text() if source_el else "trust me bro"
                    })

        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    # Save to CSV
    with open("nasdaq_stock_news.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "title", "url", "source"])
        writer.writeheader()
        writer.writerows(articles_data)
    return


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        tickers = get_trending_tickers(page, limit=5)
        print("Discovered trending tickers:", tickers)
        scrape_yahoo_news(page, tickers)
        browser.close()


if __name__ == "__main__":
    main()
