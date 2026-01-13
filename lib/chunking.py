#!/usr/bin/env python3
"""
Polymath V2 Chunking - Structure-Aware Markdown Splitting

Key insight: MinerU outputs structured Markdown. Splitting by character count
destroys that structure. Instead, split by Markdown headers (H1, H2, H3).

This preserves:
- Section context (Methods, Results, Discussion)
- Table integrity
- Mathematical proofs
- Code blocks
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """A semantically meaningful chunk of text."""
    header: str           # Section header (e.g., "Methods", "Results")
    content: str          # Full markdown content of section
    level: int            # Header level (1, 2, or 3)
    parent_header: Optional[str] = None  # Parent section for hierarchy
    char_start: int = 0   # Character offset in original document
    char_end: int = 0     # Character offset end


def chunk_markdown_by_headers(md_text: str, max_size: int = 4000, min_size: int = 50) -> List[Chunk]:
    """
    Split Markdown by headers (H1, H2, H3) to preserve semantic context.

    Args:
        md_text: Full markdown text from MinerU
        max_size: Maximum chars per chunk (only split if section exceeds this)
        min_size: Minimum chars to keep a section (skip empty sections)

    Returns:
        List of Chunk objects with header, content, and hierarchy info
    """
    chunks = []

    # Split by headers (# ## ###)
    # Pattern captures the header line and everything until the next header
    header_pattern = r'^(#{1,3})\s+(.+?)$'

    # Find all header positions
    headers = list(re.finditer(header_pattern, md_text, re.MULTILINE))

    if not headers:
        # No headers found - treat entire text as one chunk
        if len(md_text.strip()) >= min_size:
            chunks.append(Chunk(
                header="Document",
                content=md_text.strip(),
                level=0,
                char_start=0,
                char_end=len(md_text)
            ))
        return chunks

    # Track parent headers for hierarchy
    current_h1 = None
    current_h2 = None

    for i, match in enumerate(headers):
        level = len(match.group(1))  # Count # symbols
        header_text = match.group(2).strip()

        # Clean markdown artifacts from headers (MinerU often preserves **bold** in headers)
        header_text = header_text.replace('**', '').replace('*', '').replace('__', '').replace('_', ' ')
        header_text = re.sub(r'\s+', ' ', header_text).strip()  # Normalize whitespace

        # Get content between this header and the next (or end of document)
        start_pos = match.end()
        if i + 1 < len(headers):
            end_pos = headers[i + 1].start()
        else:
            end_pos = len(md_text)

        content = md_text[start_pos:end_pos].strip()

        # Skip empty sections
        if len(content) < min_size:
            continue

        # Update hierarchy tracking
        if level == 1:
            current_h1 = header_text
            current_h2 = None
            parent = None
        elif level == 2:
            current_h2 = header_text
            parent = current_h1
        else:  # level == 3
            parent = current_h2 or current_h1

        # If section is too large, split it further (fallback to paragraph splitting)
        if len(content) > max_size:
            sub_chunks = _split_large_section(content, header_text, level, parent, max_size, match.start())
            chunks.extend(sub_chunks)
        else:
            chunks.append(Chunk(
                header=header_text,
                content=content,
                level=level,
                parent_header=parent,
                char_start=match.start(),
                char_end=end_pos
            ))

    return chunks


def _split_large_section(
    content: str,
    header: str,
    level: int,
    parent: Optional[str],
    max_size: int,
    base_offset: int
) -> List[Chunk]:
    """
    Split a large section by paragraphs (double newline).
    Only used as fallback for extremely long sections.
    """
    chunks = []
    paragraphs = re.split(r'\n\n+', content)

    current_chunk = []
    current_size = 0
    chunk_start = base_offset
    part_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_size + len(para) > max_size and current_chunk:
            # Flush current chunk
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                header=f"{header} (Part {part_num})",
                content=chunk_content,
                level=level,
                parent_header=parent,
                char_start=chunk_start,
                char_end=chunk_start + len(chunk_content)
            ))
            current_chunk = [para]
            current_size = len(para)
            chunk_start += len(chunk_content)
            part_num += 1
        else:
            current_chunk.append(para)
            current_size += len(para)

    # Flush remaining
    if current_chunk:
        chunk_content = '\n\n'.join(current_chunk)
        chunks.append(Chunk(
            header=f"{header} (Part {part_num})" if part_num > 1 else header,
            content=chunk_content,
            level=level,
            parent_header=parent,
            char_start=chunk_start,
            char_end=chunk_start + len(chunk_content)
        ))

    return chunks


def chunk_text_sliding_window(text: str, size: int = 1500, overlap: int = 200) -> List[str]:
    """
    DEPRECATED: Old-style sliding window chunking.
    Only use this as a last resort for non-markdown text.

    This destroys document structure - avoid if possible.
    """
    import warnings
    warnings.warn(
        "chunk_text_sliding_window is deprecated. Use chunk_markdown_by_headers instead.",
        DeprecationWarning
    )

    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks


# Convenience function for migration
def get_chunk_texts(chunks: List[Chunk]) -> List[str]:
    """Extract just the text content from chunks (for embedding)."""
    return [c.content for c in chunks]


def get_chunk_with_context(chunk: Chunk) -> str:
    """Get chunk text with header context prepended (better for embedding)."""
    context = ""
    if chunk.parent_header:
        context += f"# {chunk.parent_header}\n\n"
    context += f"## {chunk.header}\n\n"
    context += chunk.content
    return context
