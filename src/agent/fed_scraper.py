"""Federal Reserve document scraper.

Fetches and parses FOMC minutes, statements, Beige Book summaries, and
Fed speeches from the Federal Reserve's public website.

All documents are returned as :class:`FedDocument` dataclasses with
standardised metadata.

Note:
    This module makes HTTP requests to ``federalreserve.gov``.  Respect
    the site's robots.txt and rate-limit your requests.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger


_BASE_URL = "https://www.federalreserve.gov"
_FOMC_CALENDAR_URL = f"{_BASE_URL}/monetarypolicy/fomccalendars.htm"
_BEIGE_BOOK_URL = f"{_BASE_URL}/monetarypolicy/beige-book-default.htm"
_SPEECHES_URL = f"{_BASE_URL}/apps/feds/speeches"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MacroNowcasterBot/1.0; "
        "+https://github.com/macro-regime-nowcaster)"
    )
}


@dataclass
class FedDocument:
    """A single Federal Reserve document.

    Attributes:
        date: Publication date.
        document_type: One of ``"minutes"``, ``"statement"``, ``"beige_book"``,
            ``"speech"``.
        title: Document title or headline.
        text: Plain-text body (may be truncated for very long documents).
        url: Source URL.
    """

    date: str
    document_type: str
    title: str
    text: str
    url: str


class FedScraper:
    """Scraper for public Federal Reserve communications.

    Args:
        request_delay: Seconds to wait between HTTP requests (be polite).
        timeout: HTTP request timeout in seconds.
        max_text_length: Maximum characters to store per document body.
    """

    def __init__(
        self,
        request_delay: float = 1.0,
        timeout: int = 30,
        max_text_length: int = 50_000,
    ) -> None:
        self._delay = request_delay
        self._timeout = timeout
        self._max_len = max_text_length
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_fomc_minutes(self, n_recent: int = 3) -> list[FedDocument]:
        """Fetch the most recent FOMC meeting minutes.

        Args:
            n_recent: Number of most-recent minutes documents to fetch.

        Returns:
            List of :class:`FedDocument` objects.
        """
        logger.info(f"Fetching {n_recent} FOMC minutes documents")
        links = self._get_fomc_links(document_type="minutes", n=n_recent)
        documents = []
        for url, meeting_date in links:
            doc = self._fetch_document(url, "minutes", meeting_date)
            if doc:
                documents.append(doc)
        return documents

    def fetch_fomc_statements(self, n_recent: int = 3) -> list[FedDocument]:
        """Fetch the most recent FOMC press statements.

        Args:
            n_recent: Number of most-recent statements to fetch.

        Returns:
            List of :class:`FedDocument` objects.
        """
        logger.info(f"Fetching {n_recent} FOMC statements")
        links = self._get_fomc_links(document_type="statement", n=n_recent)
        documents = []
        for url, meeting_date in links:
            doc = self._fetch_document(url, "statement", meeting_date)
            if doc:
                documents.append(doc)
        return documents

    def fetch_beige_book(self, n_recent: int = 2) -> list[FedDocument]:
        """Fetch recent Beige Book summaries.

        Args:
            n_recent: Number of recent Beige Books to fetch.

        Returns:
            List of :class:`FedDocument` objects.
        """
        logger.info(f"Fetching {n_recent} Beige Book documents")
        try:
            resp = self._get(_BEIGE_BOOK_URL)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.select("a[href*='beigebook']")[:n_recent]:
                href = a["href"]
                if not href.startswith("http"):
                    href = _BASE_URL + href
                links.append(href)

            documents = []
            for url in links:
                doc = self._fetch_document(url, "beige_book", title="Beige Book")
                if doc:
                    documents.append(doc)
            return documents
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fetch Beige Book: {exc}")
            return []

    def fetch_all_recent(self, n_each: int = 2) -> list[FedDocument]:
        """Convenience method: fetch minutes + statements + Beige Book.

        Args:
            n_each: Number of documents to fetch per category.

        Returns:
            Combined list of :class:`FedDocument` objects, sorted by date.
        """
        docs: list[FedDocument] = []
        docs += self.fetch_fomc_minutes(n_each)
        docs += self.fetch_fomc_statements(n_each)
        docs += self.fetch_beige_book(n_each)
        docs.sort(key=lambda d: d.date, reverse=True)
        return docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, url: str) -> requests.Response:
        time.sleep(self._delay)
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    def _get_fomc_links(
        self, document_type: str, n: int
    ) -> list[tuple[str, str]]:
        """Parse the FOMC calendar page for document links."""
        try:
            resp = self._get(_FOMC_CALENDAR_URL)
            soup = BeautifulSoup(resp.text, "html.parser")
            results: list[tuple[str, str]] = []

            # FOMC calendar rows contain anchors with href matching doc type
            keyword = "minutes" if document_type == "minutes" else "monetary"
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if keyword in href.lower() and href.endswith(".htm"):
                    full_url = href if href.startswith("http") else _BASE_URL + href
                    meeting_date = _extract_date_from_url(href)
                    results.append((full_url, meeting_date))
                    if len(results) >= n:
                        break
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fetch FOMC calendar: {exc}")
            return []

    def _fetch_document(
        self,
        url: str,
        doc_type: str,
        meeting_date: str = "",
        title: str = "",
    ) -> Optional[FedDocument]:
        """Fetch a single document page and extract its text."""
        try:
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title
            h1 = soup.find("h1") or soup.find("h2")
            doc_title = title or (h1.get_text(strip=True) if h1 else url)

            # Extract date from page if not already provided
            if not meeting_date:
                meeting_date = _extract_date_from_page(soup)

            # Extract body text
            body = soup.find("div", class_=re.compile(r"col-xs|article|content", re.I))
            if body is None:
                body = soup.find("body")
            text = body.get_text(separator="\n", strip=True) if body else ""
            text = _clean_text(text)[: self._max_len]

            return FedDocument(
                date=meeting_date,
                document_type=doc_type,
                title=doc_title,
                text=text,
                url=url,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fetch document at {url}: {exc}")
            return None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _extract_date_from_url(url: str) -> str:
    """Attempt to extract a date string from a FOMC URL."""
    match = re.search(r"(\d{8})", url)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return ""


def _extract_date_from_page(soup: BeautifulSoup) -> str:
    """Attempt to extract a publication date from a page."""
    date_el = soup.find(class_=re.compile(r"article__time|date|pubdate", re.I))
    if date_el:
        return date_el.get_text(strip=True)
    time_el = soup.find("time")
    if time_el:
        return time_el.get("datetime", time_el.get_text(strip=True))
    return ""


def _clean_text(text: str) -> str:
    """Strip excessive whitespace from extracted text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
