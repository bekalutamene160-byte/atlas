"""Unit tests for DocumentParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.ingestion.parser import DocumentParser


class TestDocumentParser:
    """Tests for the multi-format document parser."""

    async def test_parse_text_file(self, tmp_path: Path):
        """TXT files are parsed as page 1."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.\nIt has two lines.")

        parser = DocumentParser()
        doc = await parser.parse(test_file)

        assert doc.source == str(test_file)
        assert doc.title == "test"
        assert doc.page_count == 1
        assert "test document" in doc.pages[0].text
        assert doc.pages[0].page_num == 1

    async def test_parse_markdown_file(self, tmp_path: Path):
        """Markdown files extract title from first H1."""
        test_file = tmp_path / "guide.md"
        test_file.write_text("# Pump Maintenance Guide\n\nThis is the content.")

        parser = DocumentParser()
        doc = await parser.parse(test_file)

        assert doc.title == "Pump Maintenance Guide"
        assert doc.page_count == 1
        assert "Pump Maintenance Guide" in doc.pages[0].text or "content" in doc.pages[0].text

    async def test_parse_nonexistent_file(self, tmp_path: Path):
        """Nonexistent files raise FileNotFoundError."""
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            await parser.parse(tmp_path / "nonexistent.txt")

    async def test_unsupported_extension(self, tmp_path: Path):
        """Unsupported extensions raise ValueError."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("content")

        parser = DocumentParser()
        with pytest.raises(ValueError, match="Unsupported format"):
            await parser.parse(test_file)

    async def test_parse_directory(self, tmp_path: Path):
        """parse_directory processes all supported files."""
        (tmp_path / "file1.txt").write_text("content 1")
        (tmp_path / "file2.md").write_text("# Title\ncontent 2")
        (tmp_path / "file3.csv").write_text("ignored")  # unsupported

        parser = DocumentParser()
        docs = await parser.parse_directory(tmp_path)

        assert len(docs) == 2  # txt and md, not csv

    async def test_parse_directory_nonexistent(self, tmp_path: Path):
        """parse_directory on missing dir raises FileNotFoundError."""
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            await parser.parse_directory(tmp_path / "nonexistent")

    async def test_text_encoding_fallback(self, tmp_path: Path):
        """Latin-1 files are parsed via fallback."""
        # Write bytes that are invalid UTF-8 but valid latin-1
        test_file = tmp_path / "latin.txt"
        test_file.write_bytes(b"Caf\xe9 menu with caf\xe9 items")

        parser = DocumentParser()
        doc = await parser.parse(test_file)

        assert doc.page_count == 1
        assert len(doc.pages[0].text) > 0
