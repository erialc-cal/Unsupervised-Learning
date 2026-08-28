#!/usr/bin/env python3
"""
Build a literary-styles dataset similar to Peng & Hengartner.

What it does:
1. Downloads public-domain texts from Project Gutenberg
2. Removes Gutenberg header/footer boilerplate
3. Tokenizes into lowercase word tokens
4. Splits each book into fixed-size blocks (default: 1700 words)
5. Counts selected function words in each block
6. Saves a CSV suitable for PCA / classification / stylometry experiments


Outputs:
    literary_styles_dataset.csv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


BLOCK_SIZE = 1700
COURSE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = COURSE_ROOT / "data" / "Author"
OUTFILE = COURSE_ROOT / "data" / "literary_styles_dataset.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# A common 69-word function-word set used in recreations of the dataset.
# The teaching version based on Peng & Hengartner includes words such as
# “a”, “by”, “no”, “that”, “with”, etc.
FUNCTION_WORDS = [
    "a", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "but", "by", "can", "do", "down", "even", "every", "for", "from", "had",
    "has", "have", "her", "his", "if", "in", "into", "is", "it", "its", "may",
    "more", "must", "my", "no", "not", "now", "of", "on", "one", "only", "or",
    "our", "should", "so", "some", "such", "than", "that", "the", "their",
    "then", "there", "things", "this", "to", "up", "upon", "was", "were",
    "what", "when", "which", "who", "will", "with", "would", "your"
]


@dataclass(frozen=True)
class Book:
    author: str
    book_id: int
    title: str
    gutenberg_id: int


# This is a practical starter corpus in the same style as the commonly used
# recreation: Austen, London, Milton, Shakespeare.
# You can add/remove books as needed.
BOOKS: list[Book] = [
    # Jane Austen (7)
    Book("Austen", 1, "Sense and Sensibility", 161),
    Book("Austen", 2, "Pride and Prejudice", 1342),
    Book("Austen", 3, "Mansfield Park", 141),
    Book("Austen", 4, "Emma", 158),
    Book("Austen", 5, "Northanger Abbey", 121),
    Book("Austen", 6, "Persuasion", 105),
    Book("Austen", 7, "Lady Susan", 946),

    # Jack London (6)
    Book("London", 8, "The Call of the Wild", 215),
    Book("London", 9, "White Fang", 910),
    Book("London", 10, "Martin Eden", 1056),
    Book("London", 11, "The Sea-Wolf", 1074),
    Book("London", 12, "Before Adam", 310),
    Book("London", 13, "Burning Daylight", 746),

    # John Milton (2)
    Book("Milton", 14, "Paradise Lost", 20),
    Book("Milton", 15, "Paradise Regained", 58),

    # William Shakespeare (12)
    Book("Shakespeare", 16, "Hamlet", 1524),
    Book("Shakespeare", 17, "Macbeth", 1533),
    Book("Shakespeare", 18, "Othello", 1531),
    Book("Shakespeare", 19, "King Lear", 1532),
    Book("Shakespeare", 20, "Romeo and Juliet", 1513),
    Book("Shakespeare", 21, "Julius Caesar", 1522),
    Book("Shakespeare", 22, "The Tempest", 23042),
    Book("Shakespeare", 23, "A Midsummer Night's Dream", 1514),
    Book("Shakespeare", 24, "Much Ado About Nothing", 1519),
    Book("Shakespeare", 25, "Twelfth Night", 1526),
    Book("Shakespeare", 26, "As You Like It", 1523),
    Book("Shakespeare", 27, "Merchant of Venice", 1515),
]


def gutenberg_candidate_urls(gid: int) -> list[str]:
    """
    Try several common Gutenberg URL patterns.
    """
    return [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
    ]


def download_book(book: Book) -> Path:
    outpath = DATA_DIR / f"{book.gutenberg_id}_{slugify(book.title)}.txt"
    if outpath.exists():
        return outpath

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; literary-styles-builder/1.0)"
    }

    last_err = None
    for url in gutenberg_candidate_urls(book.gutenberg_id):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                outpath.write_text(r.text, encoding="utf-8")
                return outpath
        except Exception as e:  # noqa: BLE001
            last_err = e

    raise RuntimeError(
        f"Could not download Gutenberg ID {book.gutenberg_id} ({book.title}). "
        f"Last error: {last_err}"
    )


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header/footer as robustly as possible.
    """
    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    start_idx = 0
    end_idx = len(text)

    for pat in start_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            start_idx = max(start_idx, m.end())
            break

    for pat in end_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            end_idx = min(end_idx, m.start())
            break

    return text[start_idx:end_idx].strip()


def tokenize(text: str) -> list[str]:
    """
    Lowercase alphabetic/apostrophe tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens


def chunked(tokens: list[str], block_size: int) -> Iterable[list[str]]:
    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        if len(block) == block_size:
            yield block


def count_function_words(block: list[str], function_words: list[str]) -> dict[str, int]:
    ctr = Counter(block)
    return {w: ctr.get(w, 0) for w in function_words}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for book in BOOKS:
        print(f"Downloading / processing: {book.author} — {book.title}", file=sys.stderr)
        path = download_book(book)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        clean = strip_gutenberg_boilerplate(raw)
        tokens = tokenize(clean)

        for block_num, block in enumerate(chunked(tokens, BLOCK_SIZE), start=1):
            row: dict[str, object] = {
                "Author": book.author,
                "BookID": book.book_id,
                "Title": book.title,
                "GutenbergID": book.gutenberg_id,
                "BlockID": block_num,
                "BlockWords": BLOCK_SIZE,
            }
            row.update(count_function_words(block, FUNCTION_WORDS))
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, object]], outfile: str | Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_csv(rows, OUTFILE)
    print(f"Wrote {len(rows)} rows to {OUTFILE}")


if __name__ == "__main__":
    main()
