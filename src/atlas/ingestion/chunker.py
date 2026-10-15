"""
Recursive character chunker — splits documents into retrievable chunks.

Strategy: try to split on paragraph boundaries (\\n\\n) first, then
newlines (\\n), then sentence boundaries (". "), then spaces. This
preserves natural boundaries wherever possible.

Each chunk has:
- chunk_id: "{source_stem}_p{page}_{index}" — unique, traceable
- source: the original file path
- page: the page number (for citations)
- text: the chunk content
- chunk_index: position in the source document

Overlap: the last N characters of one chunk are prepended to the next.
This prevents a sentence split across chunks from losing context.
"""

from __future__ import annotations

from atlas.models import Chunk


class RecursiveCharacterSplitter:
    """Splits text into chunks using recursive character splitting.

    Tries separators in order: ["\\n\\n", "\\n", ". ", " ", ""].
    The first separator that splits the text into pieces smaller than
    chunk_size is used. This preserves natural boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str, source: str, page: int,
                   source_stem: str) -> list[Chunk]:
        """Split a single page's text into chunks.

        Args:
            text: The text to split.
            source: The source file path (for the chunk's source field).
            page: The page number (for citations).
            source_stem: The filename without extension (for chunk_id).

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        # If text is small enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [Chunk(
                chunk_id=f"{source_stem}_p{page}_0",
                source=source,
                page=page,
                text=text.strip(),
                chunk_index=0,
            )]

        chunks: list[Chunk] = []
        self._split_recursive(
            text, source, page, source_stem, 0, chunks
        )
        return chunks

    def _split_recursive(
        self,
        text: str,
        source: str,
        page: int,
        source_stem: str,
        chunk_index: int,
        chunks: list[Chunk],
    ) -> None:
        """Recursively split text using separators.

        Try each separator in order. If a separator splits the text
        into pieces, accumulate them into chunks of chunk_size. If no
        separator works (text is one long string), force-split at
        chunk_size.
        """
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{source_stem}_p{page}_{chunk_index}",
                    source=source,
                    page=page,
                    text=text.strip(),
                    chunk_index=chunk_index,
                ))
            return

        # Find the first separator that actually splits the text
        separator = ""
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        if separator == "":
            # No separator found — force-split at chunk_size
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk_text = text[i:i + self.chunk_size]
                if chunk_text.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{source_stem}_p{page}_{chunk_index}",
                        source=source,
                        page=page,
                        text=chunk_text.strip(),
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
            return

        # Split by the separator
        pieces = text.split(separator)
        current_chunk = ""

        for piece in pieces:
            candidate = current_chunk + separator + piece if current_chunk else piece

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Current chunk is full — save it
                if current_chunk.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{source_stem}_p{page}_{chunk_index}",
                        source=source,
                        page=page,
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1

                # Start next chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:] + separator + piece
                else:
                    current_chunk = piece

                # If the piece itself is too big, recurse
                if len(current_chunk) > self.chunk_size:
                    self._split_recursive(
                        current_chunk, source, page, source_stem,
                        chunk_index, chunks
                    )
                    # Update chunk_index based on how many chunks were added
                    chunk_index = len(chunks)
                    current_chunk = ""

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                chunk_id=f"{source_stem}_p{page}_{chunk_index}",
                source=source,
                page=page,
                text=current_chunk.strip(),
                chunk_index=chunk_index,
            ))

    def split_document(self, parsed_doc) -> list[Chunk]:
        """Split a ParsedDocument into chunks.

        Each page is split independently, preserving page numbers
        for citations.
        """
        from atlas.ingestion.parser import ParsedDocument

        all_chunks: list[Chunk] = []
        source_stem = str(parsed_doc.source).split("\\")[-1].split("/")[-1]
        # Remove extension
        if "." in source_stem:
            source_stem = source_stem.rsplit(".", 1)[0]

        for page in parsed_doc.pages:
            page_chunks = self.split_text(
                text=page.text,
                source=page.source,
                page=page.page_num,
                source_stem=source_stem,
            )
            all_chunks.extend(page_chunks)

        return all_chunks
