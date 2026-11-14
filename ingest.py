
import os
import time
import requests
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ARTICLES = [
    "Binary search algorithm",
    "Hash table",
    "Linked list",
    "Binary search tree",
    "Quicksort",
    "Merge sort",
    "Dynamic programming",
    "Breadth-first search",
    "Depth-first search",
    "Big O notation",
    "Stack (abstract data type)",
    "Queue (abstract data type)"
]

URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "DSA-Learning-App/1.0 (educational project)"
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json"
    }

    for attempt in range(5):
        try:
            response = session.get(
                URL,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                wait = 5 * (2 ** attempt)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            pages = response.json()["query"]["pages"]
            page = next(iter(pages.values()))

            return page.get("extract", "")

        except requests.RequestException as e:
            if attempt == 4:
                print(f"Failed to fetch {title}: {e}")
                return None

            time.sleep(5 * (attempt + 1))

    print(f"Skipping {title} after repeated rate limits.")
    return None


def main():
    success = 0
    failed = []

    for title in ARTICLES:
        filename = title.replace(" ", "_").replace("/", "_") + ".txt"
        filepath = DATA_DIR / filename

        # Preserve previously downloaded articles
        if filepath.exists() and filepath.stat().st_size > 0:
            print(f"Already exists: {filename}")
            success += 1
            continue

        text = fetch(title)

        if not text:
            failed.append(title)
            continue

        filepath.write_text(text, encoding="utf-8")

        print(f"Saved {filename} ({len(text.split())} words)")
        success += 1

        # Be polite to Wikipedia
        time.sleep(3)

    print("\n---------- INGESTION SUMMARY ----------")
    print(f"Successful: {success}/{len(ARTICLES)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("Failed articles:")
        for title in failed:
            print("-", title)


if __name__ == "__main__":
    main()
