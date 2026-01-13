#!/usr/bin/env python3
"""
Polymath V2 Metadata Extraction - pdf2doi + Zotero Integration

Priority order for metadata extraction:
1. pdf2doi: Scan PDF binary for DOI/arXiv ID (most reliable)
2. Zotero lookup: If DOI found, Zotero auto-fills all metadata
3. Filename regex: Last resort fallback

NEVER trust filename parsing if we can find a DOI in the PDF.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class PaperMetadata:
    """Structured paper metadata."""
    title: str
    authors: list[str]
    year: Optional[int]
    doi: Optional[str]
    arxiv_id: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    venue: Optional[str]
    abstract: Optional[str]
    source_method: str  # 'pdf2doi', 'crossref', 'zotero', 'filename'
    confidence: float   # 0.0-1.0


def extract_identifier_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract DOI or arXiv ID from PDF binary content.

    Uses pdf2doi which searches:
    - PDF metadata
    - First pages text
    - CrossRef lookup by title

    Returns:
        Dict with 'identifier', 'identifier_type', 'method'
    """
    try:
        import pdf2doi
    except ImportError:
        raise ImportError("pdf2doi not installed. Run: pip install pdf2doi")

    result = pdf2doi.pdf2doi(str(pdf_path))

    if result and result.get('identifier'):
        return {
            'identifier': result['identifier'],
            'identifier_type': result.get('identifier_type', 'doi'),  # 'doi' or 'arxiv'
            'method': result.get('method', 'pdf2doi'),
            'validation_info': result.get('validation_info')
        }

    return {'identifier': None, 'identifier_type': None, 'method': 'failed'}


def lookup_crossref(doi: str) -> Optional[PaperMetadata]:
    """
    Look up paper metadata from CrossRef using DOI.

    CrossRef is the authoritative source for DOI metadata.
    """
    import requests

    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": "Polymath/2.0 (mailto:max.van.belkum@vanderbilt.edu)"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()['message']

        # Extract authors
        authors = []
        for author in data.get('author', []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)

        # Extract year
        year = None
        if 'published-print' in data:
            year = data['published-print']['date-parts'][0][0]
        elif 'published-online' in data:
            year = data['published-online']['date-parts'][0][0]
        elif 'created' in data:
            year = data['created']['date-parts'][0][0]

        # Extract venue
        venue = None
        if data.get('container-title'):
            venue = data['container-title'][0] if isinstance(data['container-title'], list) else data['container-title']

        return PaperMetadata(
            title=data.get('title', ['Unknown'])[0] if isinstance(data.get('title'), list) else data.get('title', 'Unknown'),
            authors=authors,
            year=year,
            doi=doi,
            arxiv_id=None,
            pmid=None,
            pmcid=None,
            venue=venue,
            abstract=data.get('abstract'),
            source_method='crossref',
            confidence=0.95
        )

    except Exception as e:
        print(f"CrossRef lookup failed for {doi}: {e}")
        return None


def lookup_arxiv(arxiv_id: str) -> Optional[PaperMetadata]:
    """
    Look up paper metadata from arXiv API.
    """
    import requests
    import xml.etree.ElementTree as ET

    # Clean arxiv ID
    arxiv_id = arxiv_id.replace('arXiv:', '').strip()

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        title = entry.find('atom:title', ns)
        title_text = title.text.strip().replace('\n', ' ') if title is not None else 'Unknown'

        # Authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)

        # Year from published date
        published = entry.find('atom:published', ns)
        year = None
        if published is not None:
            year = int(published.text[:4])

        # Abstract
        summary = entry.find('atom:summary', ns)
        abstract = summary.text.strip() if summary is not None else None

        return PaperMetadata(
            title=title_text,
            authors=authors,
            year=year,
            doi=None,
            arxiv_id=arxiv_id,
            pmid=None,
            pmcid=None,
            venue='arXiv',
            abstract=abstract,
            source_method='arxiv',
            confidence=0.95
        )

    except Exception as e:
        print(f"arXiv lookup failed for {arxiv_id}: {e}")
        return None


def extract_metadata_from_filename(pdf_path: Path) -> PaperMetadata:
    """
    FALLBACK: Extract metadata from filename.
    This is the LEAST reliable method - only use if pdf2doi fails.
    """
    name = pdf_path.stem

    # Year extraction
    year = None
    if match := re.search(r'(19|20)\d{2}', name):
        year = int(match.group(0))

    # DOI pattern (10.xxxx/...)
    doi = None
    if name.startswith("10."):
        doi = name.replace("_", "/")

    # arXiv pattern (YYMM.NNNNN)
    arxiv_id = None
    if re.match(r'\d{4}\.\d+', name):
        match = re.match(r'(\d{4}\.\d+)', name)
        arxiv_id = match.group(1) if match else None

    # Clean up title from filename
    title = name
    title = re.sub(r'^1-s2\.0-S\d+-', '', title)
    title = re.sub(r'^10\.\d+_', '', title)
    title = re.sub(r'^\d{4}\.\d+v?\d*', '', title)
    title = re.sub(r'^PMC\d+_', '', title)
    title = title.replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\s+', ' ', title).strip()

    if not title or len(title) < 5:
        title = name

    return PaperMetadata(
        title=title[:200],
        authors=[],
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        pmid=None,
        pmcid=None,
        venue=None,
        abstract=None,
        source_method='filename',
        confidence=0.3
    )


def get_paper_metadata(pdf_path: Path) -> PaperMetadata:
    """
    Get best available metadata for a PDF.

    Priority:
    1. pdf2doi scan → CrossRef/arXiv lookup (high confidence)
    2. Filename parsing (low confidence)
    """
    pdf_path = Path(pdf_path)

    # Step 1: Try to extract identifier from PDF
    id_result = extract_identifier_from_pdf(pdf_path)

    if id_result['identifier']:
        # Step 2: Look up metadata using the identifier
        if id_result['identifier_type'] == 'doi':
            metadata = lookup_crossref(id_result['identifier'])
            if metadata:
                return metadata
        elif id_result['identifier_type'] == 'arxiv':
            metadata = lookup_arxiv(id_result['identifier'])
            if metadata:
                return metadata

    # Step 3: Fallback to filename parsing
    return extract_metadata_from_filename(pdf_path)


def metadata_to_zotero_item(metadata: PaperMetadata, pdf_path: Path) -> Dict[str, Any]:
    """
    Convert PaperMetadata to Zotero item format.

    If we have a DOI or arXiv ID, Zotero can auto-fill using its translator.
    """
    # Determine item type
    if metadata.arxiv_id:
        item_type = 'preprint'
    else:
        item_type = 'journalArticle'

    item = {
        'itemType': item_type,
        'title': metadata.title,
    }

    # Authors
    if metadata.authors:
        item['creators'] = [
            {'creatorType': 'author', 'name': name}
            for name in metadata.authors
        ]

    # Year/Date
    if metadata.year:
        item['date'] = str(metadata.year)

    # Identifiers
    if metadata.doi:
        item['DOI'] = metadata.doi

    if metadata.arxiv_id:
        item['archiveID'] = f"arXiv:{metadata.arxiv_id}"
        item['url'] = f"https://arxiv.org/abs/{metadata.arxiv_id}"

    # Venue
    if metadata.venue:
        item['publicationTitle'] = metadata.venue

    # Abstract
    if metadata.abstract:
        item['abstractNote'] = metadata.abstract[:5000]  # Zotero limit

    # Extra field for tracking
    extra = [
        f"Source: {metadata.source_method}",
        f"Confidence: {metadata.confidence:.2f}",
        f"Filename: {pdf_path.name}"
    ]
    item['extra'] = '\n'.join(extra)

    return item
