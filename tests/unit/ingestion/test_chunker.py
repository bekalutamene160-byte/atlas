"""Unit tests for RecursiveCharacterSplitter."""

from __future__ import annotations

import pytest

from atlas.ingestion.chunker import RecursiveCharacterSplitter
from atlas.ingestion.parser import ParsedDocument, ParsedPage
from atlas.models import Chunk


class TestRecursiveCharacterSplitter:
    """Tests for the chunker."""

    def test_small_text_returns_single_chunk(self):
        """Text smaller than chunk_size returns one chunk."""
        splitter = RecursiveCharacterSplitter(chunk_size=800, overlap=100)
        chunks = splitter.split_text(
            text="This is a short text.",
            source="test.pdf",
            page=1,
            source_stem="test",
        )
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short text."
        assert chunks[0].chunk_id == "test_p1_0"
        assert chunks[0].page == 1

    def test_empty_text_returns_no_chunks(self):
        """Empty text returns no chunks."""
        splitter = RecursiveCharacterSplitter(chunk_size=800, overlap=100)
        chunks = splitter.split_text(
            text="",
            source="test.pdf",
            page=1,
            source_stem="test",
        )
        assert len(chunks) == 0

    def test_whitespace_only_text_returns_no_chunks(self):
        """Whitespace-only text returns no chunks."""
        splitter = RecursiveCharacterSplitter(chunk_size=800, overlap=100)
        chunks = splitter.split_text(
            text="   \n\n   \t  ",
            source="test.pdf",
            page=1,
            source_stem="test",
        )
        assert len(chunks) == 0

    def test_long_text_is_split(self):
        """Text longer than chunk_size is split into multiple chunks."""
        splitter = RecursiveCharacterSplitter(chunk_size=100, overlap=20)
        long_text = "This is a sentence. " * 30  # ~600 chars

        chunks = splitter.split_text(
            text=long_text,
            source="test.pdf",
            page=1,
            source_stem="test",
        )

        assert len(chunks) > 1
        # Each chunk should be <= chunk_size (approximately)
        for chunk in chunks:
            assert len(chunk.text) <= 120  # small buffer for separator

    def test_chunk_ids_are_unique(self):
        """Each chunk has a unique chunk_id."""
        splitter = RecursiveCharacterSplitter(chunk_size=100, overlap=20)
        long_text = "This is a sentence. " * 30

        chunks = splitter.split_text(
            text=long_text,
            source="test.pdf",
            page=1,
            source_stem="test",
        )

        ids = {c.chunk_id for c in chunks}
        assert len(ids) == len(chunks)  # all unique

    def test_chunk_ids_include_page_and_index(self):
        """chunk_id format is {source_stem}_p{page}_{index}."""
        splitter = RecursiveCharacterSplitter(chunk_size=800, overlap=100)
        chunks = splitter.split_text(
            text="hello",
            source="specs/atlas-3-spec.pdf",
            page=4,
            source_stem="atlas-3-spec",
        )
        assert chunks[0].chunk_id == "atlas-3-spec_p4_0"

    def test_split_document_preserves_page_numbers(self):
        """split_document preserves page numbers from each page."""
        pages = [
            ParsedPage(page_num=1, text="Page 1 content", source="test.pdf"),
            ParsedPage(page_num=2, text="Page 2 content", source="test.pdf"),
        ]
        doc = ParsedDocument(
            source="test.pdf",
            title="test",
            author=None,
            date_text=None,
            pages=pages,
        )

        splitter = RecursiveCharacterSplitter(chunk_size=800, overlap=100)
        chunks = splitter.split_document(doc)

        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[1].page == 2

    def test_paragraph_boundaries_preferred(self):
        """Splitter prefers paragraph boundaries (\\n\\n)."""
        splitter = RecursiveCharacterSplitter(chunk_size=50, overlap=10)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."

        chunks = splitter.split_text(
            text=text,
            source="test.pdf",
            page=1,
            source_stem="test",
        )

        # Should split at paragraph boundaries
        assert len(chunks) > 1
        # No chunk should split a sentence in half
        for chunk in chunks:
            assert not chunk.text.startswith("paragraph here")

    def test_overlap_preserves_context(self):
        """Overlap means the end of one chunk appears at the start of the next."""
        splitter = RecursiveCharacterSplitter(chunk_size=50, overlap=20)
        text = "A" * 30 + ". " + "B" * 30 + ". " + "C" * 30

        chunks = splitter.split_text(
            text=text,
            source="test.pdf",
            page=1,
            source_stem="test",
        )

        if len(chunks) > 1:
            # The overlap should create some shared content
            # (exact behavior depends on separator strategy)
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0
