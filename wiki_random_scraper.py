#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Wikipedia Random Article Text Scraper
-------------------------------------
- Fetches random articles via Special:Random (follows 302 to the real article)
- Extracts clean text (paragraphs only), strips footnote refs and junk
- Saves as .txt with atomic writes
- Threaded with retry/backoff + politeness delay
- SQLite-backed deduplication on final URL + title hash
- Resource guards (RAM/disk), graceful shutdown

Usage:
    python wiki_random_scraper.py --out D:\wiki_corpus --workers 8 --delay 1.2

Dependencies:
    pip install requests beautifulsoup4 psutil
"""

import argparse
import contextlib
import hashlib
import html
import logging
import logging.handlers
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import psutil
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# Global config defaults
# -------------------------
WIKI_RANDOM_URL = "https://en.wikipedia.org/wiki/Special:Random"
USER_AGENT = "Amit-WikiScraper/1.0 (+contact: local; respectful)"
DEFAULT_DELAY = 1.2          # polite delay between requests per thread
DEFAULT_WORKERS = 8
DEFAULT_MIN_FREE_RAM_MB = 300
DEFAULT_MIN_FREE_DISK_MB = 1024
DEFAULT_LOG_ROTATE_MB = 20
DEFAULT_LOG_BACKUPS = 5
REQUEST_TIMEOUT = 20         # seconds
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5
CONNECT_POOLSIZE = 100
# sections that are lists of references/links rather than article prose (skipped when extracting text)
BOILERPLATE_SECTIONS = {"references", "notes", "citations", "footnotes", "sources", "bibliography",
                        "further reading", "external links", "see also"}

STOP_EVENT = threading.Event()

# Reduce per-thread stack size a bit (helps on Windows w/ many threads)
# Be conservative; don't set too low or you risk crashes.
try:
    threading.stack_size(512 * 1024)  # 512 KB
except Exception:
    pass

# -------------------------
# Helpers
# -------------------------

def setup_logger(log_dir: Path, verbose: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wiki_scraper")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    fh = logging.handlers.RotatingFileHandler(
        log_dir / "scraper.log",
        maxBytes=DEFAULT_LOG_ROTATE_MB * 1024 * 1024,
        backupCount=DEFAULT_LOG_BACKUPS,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def make_requests_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=CONNECT_POOLSIZE, pool_maxsize=CONNECT_POOLSIZE)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    return sess


def ensure_sqlite(db_path: Path, logger: logging.Logger) -> sqlite3.Connection:
    first_time = not db_path.exists()
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    if first_time:
        logger.info("Initializing SQLite database for dedup…")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title_hash TEXT NOT NULL,
            saved_path TEXT NOT NULL,
            saved_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title_hash ON articles(title_hash)")
    return conn


def canonicalize_title(title: str) -> str:
    # Strip whitespace, decode HTML entities, remove illegal filename chars
    t = html.unescape(title).strip()
    t = re.sub(r"[\\/:*?\"<>|\x00-\x1F]", "_", t)
    t = re.sub(r"\s+", " ", t)
    return t[:220]  # keep filename manageable


def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def url_hash(u: str) -> str:
    return hashlib.sha256(u.encode("utf-8")).hexdigest()

def clean_text_from_html(html_text: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove junk (infoboxes are <table class="infobox"> on current Wikipedia, so match any tag;
    # .noprint covers things like the "[ nl ]" interlanguage-link hints)
    for tag in soup(["script", "style"]):
        tag.decompose()
    for junk in soup.select("sup.reference, .infobox, .navbox, .toc, .mw-references-wrap, .hatnote, "
                            ".mw-editsection, .noprint"):
        junk.decompose()

    title_tag = soup.find("h1", id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    content_div = soup.select_one("div.mw-parser-output")
    if not content_div:
        return title, None

    text_lines = []
    skip_h2 = skip = False  # inside a References / External links / See also ... section
    for elem in content_div.find_all(["p", "ul", "ol", "dl", "h2", "h3"], recursive=True):
        # nested lists/paragraphs are already part of their parent item's text; don't emit them twice
        if elem.find_parent(["li", "dd", "dt"]):
            continue
        if elem.name in ["h2", "h3"]:
            heading = elem.get_text(" ", strip=True)
            boilerplate = heading.lower() in BOILERPLATE_SECTIONS
            if elem.name == "h2":
                skip_h2 = boilerplate
            skip = skip_h2 or boilerplate
            if heading and not skip:
                text_lines.append(f"\n## {heading}\n")
            continue
        if skip:
            continue
        if elem.name in ["ul", "ol", "dl"]:
            # one list item per line, instead of every item run together on a single line
            items = [li.get_text(" ", strip=True) for li in elem.find_all(["li", "dt", "dd"], recursive=False)]
            text = "\n".join(f"- {item}" for item in items if item)
        else:
            text = elem.get_text(" ", strip=True)
        # strip ref markers like [12]
        text = re.sub(r"\[\d+\]", "", text)
        if text:
            text_lines.append(text)

    body = "\n\n".join(text_lines).strip()
    return title, body



def atomic_write_text(target_path: Path, text: str):
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, target_path)


def resource_ok(min_free_ram_mb: int, min_free_disk_mb: int, out_dir: Path, logger: logging.Logger) -> bool:
    vm = psutil.virtual_memory()
    if vm.available < min_free_ram_mb * 1024 * 1024:
        logger.warning("Low RAM: %.1f MB available", vm.available / (1024 * 1024))
        return False
    try:
        usage = psutil.disk_usage(str(out_dir))
    except Exception:
        usage = psutil.disk_usage(str(out_dir.resolve()))
    free_mb = usage.free / (1024 * 1024)
    if free_mb < min_free_disk_mb:
        logger.error("Low disk space at %s: %.0f MB free", out_dir, free_mb)
        return False
    return True


def ymd_subdir(base: Path) -> Path:
    today = datetime.now(timezone.utc)
    sub = base / f"{today.year:04d}" / f"{today.month:02d}" / f"{today.day:02d}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def fetch_random_article(session: requests.Session, logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (final_url, title, body) or (None, None, None) on failure.
    """
    try:
        # GET with redirects allowed -> ends at the real article
        resp = session.get(WIKI_RANDOM_URL, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("HTTP status %s", resp.status_code)
            return None, None, None

        final_url = str(resp.url)
        title, body = clean_text_from_html(resp.text)
        if not title or not body:
            return None, None, None
        return final_url, title, body
    except requests.RequestException as e:
        logger.debug("RequestException: %s", e)
        return None, None, None
    except Exception as e:
        logger.debug("Unexpected fetch error: %s", e)
        return None, None, None


def already_saved(conn: sqlite3.Connection, url: str, title: str) -> bool:
    try:
        cur = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
        if cur.fetchone():
            return True
        th = content_hash(title)
        cur = conn.execute("SELECT 1 FROM articles WHERE title_hash = ?", (th,))
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False


def insert_record(conn: sqlite3.Connection, url: str, title: str, saved_path: str):
    th = content_hash(title)
    conn.execute(
        "INSERT OR IGNORE INTO articles (url, title_hash, saved_path, saved_at) VALUES (?, ?, ?, ?)",
        (url, th, saved_path, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    )


def worker(thread_id: int,
           session_factory,
           out_dir: Path,
           db_conn: sqlite3.Connection,
           delay: float,
           min_free_ram_mb: int,
           min_free_disk_mb: int,
           logger: logging.Logger):

    session = session_factory()
    consecutive_skips = 0

    while not STOP_EVENT.is_set():
        if not resource_ok(min_free_ram_mb, min_free_disk_mb, out_dir, logger):
            time.sleep(max(5.0, delay))
            continue

        final_url, title, body = fetch_random_article(session, logger)
        if not final_url or not title or not body:
            consecutive_skips += 1
            time.sleep(delay)
            continue

        # De-dup check
        if already_saved(db_conn, final_url, title):
            consecutive_skips += 1
            time.sleep(delay)
            continue

        safe_title = canonicalize_title(title)
        subdir = ymd_subdir(out_dir)
        # Use URL hash to ensure unique file + stable name
        fname = f"{safe_title}__{url_hash(final_url)[:16]}.txt"
        fpath = subdir / fname

        # In case of rare race, double-check file presence
        if fpath.exists():
            consecutive_skips += 1
            time.sleep(delay)
            continue

        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        text = f"# {title}\nURL: {final_url}\nFetched: {fetched_at}\n\n{body}\n"

        try:
            atomic_write_text(fpath, text)
            insert_record(db_conn, final_url, title, str(fpath))
            logger.info("Saved (%s): %s", f"{thread_id}", fpath.name)
            consecutive_skips = 0
        except Exception as e:
            logger.error("Write/DB error: %s", e)

        # politeness delay per thread
        time.sleep(delay)


def install_signal_handlers(logger: logging.Logger):
    def handle_sigint(sig, frame):
        logger.warning("Shutdown requested (signal %s). Stopping threads…", sig)
        STOP_EVENT.set()
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)


def parse_args():
    p = argparse.ArgumentParser(description="Wikipedia Special:Random text scraper")
    p.add_argument("--out", type=Path, required=True, help="Output directory (prefer your external disk)")
    p.add_argument("--db", type=Path, default=None, help="SQLite path (default: <out>/scraper.sqlite3)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of threads")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay (seconds) between requests per thread")
    p.add_argument("--min-free-ram-mb", type=int, default=DEFAULT_MIN_FREE_RAM_MB, help="Pause if below this RAM")
    p.add_argument("--min-free-disk-mb", type=int, default=DEFAULT_MIN_FREE_DISK_MB, help="Pause if below this free disk")
    p.add_argument("--verbose", action="store_true", help="Verbose console logging")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "_logs"
    logger = setup_logger(log_dir, verbose=args.verbose)
    install_signal_handlers(logger)

    db_path = args.db or (out_dir / "scraper.sqlite3")
    conn = ensure_sqlite(db_path, logger)

    logger.info("Output dir: %s", out_dir)
    logger.info("DB: %s", db_path)
    logger.info("Workers: %d | Delay/thread: %.2fs", args.workers, args.delay)

    session_factory = make_requests_session

    threads = []
    for i in range(args.workers):
        t = threading.Thread(
            target=worker,
            name=f"worker-{i+1}",
            args=(
                i + 1, session_factory, out_dir, conn, args.delay,
                args.min_free_ram_mb, args.min_free_disk_mb, logger
            ),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        # Keep main thread alive while workers run
        while not STOP_EVENT.is_set():
            time.sleep(1.5)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt; stopping…")
        STOP_EVENT.set()

    for t in threads:
        t.join(timeout=5.0)

    with contextlib.suppress(Exception):
        conn.close()
    logger.info("Exited cleanly.")


if __name__ == "__main__":
    main()