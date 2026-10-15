"""
Document parser — reads files and extracts text + metadata.

Supports: PDF, Markdown, TXT.
Each format has a dedicated parser. The main parse() method dispatches
based on file extension.

Design principle: every parsed page knows its page number. This is
non-negotiable — citations require page numbers. If a format doesn't
have pages (like Markdown), we synthesize page 1.

Critical: PDF parsing is synchronous (pdfplumber is sync). We wrap it
in asyncio.to_thread() so it doesn't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from atlas.models import Chunk

logger = logging.getLogger(__name__)


class ParsedPage:
    """A single page of a parsed document."""
    def __init__(self, page_num: int, text: str, source: str):
        self.page_num = page_num
        self.text = text
        self.source = source


class ParsedDocument:
    """A fully parsed document with metadata and pages."""
    def __init__(
        self,
        source: str,
        title: Optional[str],
        author: Optional[str],
        date_text: Optional[str],
        pages: list[ParsedPage],
    ):
        self.source = source
        self.title = title
        self.author = author
        self.date_text = date_text
        self.pages = pages

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class DocumentParser:
    """Multi-format document parser.

    Dispatches based on file extension:
    - .pdf → pdfplumber (sync, wrapped in thread)
    - .md  → markdown library
    - .txt → direct read
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a document. Returns ParsedDocument with pages and metadata.

        Raises:
            ValueError: if extension is not supported
            FileNotFoundError: if path doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {suffix}. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if suffix == ".pdf":
            return await self._parse_pdf(path)
        elif suffix in (".md", ".markdown"):
            return await self._parse_markdown(path)
        elif suffix == ".txt":
            return await self._parse_text(path)
        else:
            raise ValueError(f"Cannot parse: {path}")

    async def _parse_pdf(self, path: Path) -> ParsedDocument:
        """Parse a PDF using pdfplumber.

        pdfplumber is synchronous, so we run it in a thread to avoid
        blocking the event loop. This is critical — without it, the
        FastAPI server would freeze during ingestion.
        """
        def _extract() -> ParsedDocument:
            import pdfplumber

            pages: list[ParsedPage] = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append(ParsedPage(
                        page_num=i,
                        text=text,
                        source=str(path),
                    ))

            # Try to extract metadata from the PDF info dict
            title = path.stem  # default: filename without extension
            author = None
            date_text = None
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    meta = pdf.metadata or {}
                    title = meta.get("Title", None) or path.stem
                    author = meta.get("Author", None)
                    date_text = meta.get("CreationDate", None)
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata from {path}: {e}")

            return ParsedDocument(
                source=str(path),
                title=title,
                author=author,
                date_text=date_text,
                pages=pages,
            )

        # Run sync pdfplumber in a thread — don't block the event loop
        return await asyncio.to_thread(_extract)

    async def _parse_markdown(self, path: Path) -> ParsedDocument:
        """Parse a Markdown file.

        Markdown doesn't have pages, so we treat the whole file as page 1.
        We extract the title from the first H1 heading if present.
        """
        def _extract() -> ParsedDocument:
            content = path.read_text(encoding="utf-8")

            # Try to extract title from first H1
            title = path.stem
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Convert markdown to plain text (strip formatting)
            try:
                import markdown
                from bs4 import BeautifulSoup
                html = markdown.markdown(content)
                text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
            except ImportError:
                # Fallback: use raw content if libraries missing
                text = content

            return ParsedDocument(
                source=str(path),
                title=title,
                author=None,
                date_text=None,
                pages=[ParsedPage(page_num=1, text=text, source=str(path))],
            )

        return await asyncio.to_thread(_extract)

    async def _parse_text(self, path: Path) -> ParsedDocument:
        """Parse a plain text file.

        Try UTF-8 first, fall back to latin-1 if that fails.
        Log a warning if fallback is used — it indicates encoding issues.
        """
        def _extract() -> ParsedDocument:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {path}, trying latin-1")
                content = path.read_text(encoding="latin-1")

            return ParsedDocument(
                source=str(path),
                title=path.stem,
                author=None,
                date_text=None,
                pages=[ParsedPage(page_num=1, text=content, source=str(path))],
            )

        return await asyncio.to_thread(_extract)

    async def parse_directory(self, dir_path: Path) -> list[ParsedDocument]:
        """Parse all supported files in a directory.

        Skips files with unsupported extensions. Logs warnings for
        files that fail to parse (doesn't crash the whole batch).
        """
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        documents: list[ParsedDocument] = []
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            try:
                doc = await self.parse(file_path)
                documents.append(doc)
                logger.info(f"Parsed {file_path.name}: {doc.page_count} pages")
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        return documents
