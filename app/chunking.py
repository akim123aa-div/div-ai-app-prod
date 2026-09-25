"""PDF to chunks: the GenAI Lesson 13 pipeline, unchanged in behaviour.

Four steps, in this order, and the order is the lesson:

1. extract text page by page with pypdf
2. drop the running furniture, the lines that appear on most pages
3. split each page at its headings, carrying the last heading across the break
4. cut each section into token windows with an overlap

Then every chunk is given a header line naming the company, the page and the
section, and it is that headed text that gets embedded. A chunk that does not
say what it is about is a chunk the retriever cannot find.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pypdf
import tiktoken

from app.config import settings
from app.logs import get_logger

log = get_logger(__name__)
ENC = tiktoken.get_encoding("cl100k_base")

HEAD = re.compile(r"^[A-Z][A-Za-z&,'\-() ]{4,58}$")


@dataclass
class Chunk:
    """One retrievable unit, and everything a citation needs to name it."""

    id: str
    doc: str
    title: str
    page: int
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        head = f"{self.title}, page {self.page}" + (f", {self.section}" if self.section else "")
        return f"{head}\n{self.text}"

    @property
    def source(self) -> str:
        return f"{self.title}, page {self.page}"

    @property
    def n_tokens(self) -> int:
        return len(ENC.encode(self.text))

    def to_dict(self) -> dict:
        return asdict(self)


def read_pages(path: Path) -> list[str]:
    """Raw page text, from rung 1 of the GenAI Lesson 13 parser ladder.

    pypdf reads characters in stream order, so a table arrives as a column of
    fragments and its rows are gone. That is a real loss and it is silent. It is
    still the right default here, and the reason is measured rather than assumed:
    re-parsing this corpus with pdfplumber recovers none of the answers the
    golden set currently misses, and costs three to five times the parse time.

    The rule Lesson 13 actually teaches is run rung 1 over everything and
    escalate the pages that fail a check. This function is only the first half.
    The second half -- flag pages that look like tables, re-parse just those with
    pdfplumber -- is left to you, because which pages need it is a property of
    your corpus and not of this code.

    NUL characters are dropped. Two 10-K cover pages in this corpus render their
    checkboxes as `\x00`. Qdrant stored them without complaint in Lesson 1;
    Postgres refuses them in a text column, which is how Lesson 2 found them.
    """
    return [(p.extract_text() or "").replace("\x00", "")
            for p in pypdf.PdfReader(str(path)).pages]


def page_count(path: Path) -> int:
    """Pages in the file, including the ones that yield no text."""
    return len(pypdf.PdfReader(str(path)).pages)


def boilerplate(pages: list[str], share: float = 0.35) -> set[str]:
    """Lines that appear on at least `share` of the pages of one document."""
    seen = collections.Counter(
        ln.strip() for pg in pages for ln in set(pg.splitlines())
        if 2 < len(ln.strip()) < 90)
    return {ln for ln, n in seen.items() if n >= share * len(pages)}


def strip_furniture(pages: list[str]) -> list[str]:
    boiler = boilerplate(pages)
    return ["\n".join(ln for ln in pg.splitlines() if ln.strip() not in boiler)
            for pg in pages]


def sections(page: str, carry: str = "") -> tuple[list[tuple[str, str]], str]:
    """Split one page at heading lines, carrying the last heading in from before.

    A heading is short, starts with a capital, and is followed by a real line of
    prose. Three cheap rules; a layout-aware parser would do better and cost more.
    """
    lines, current, buf, out = page.splitlines(), carry, [], []
    for i, ln in enumerate(lines):
        s = ln.strip()
        is_head = (HEAD.match(s) and len(s.split()) <= 8
                   and i + 1 < len(lines) and len(lines[i + 1].strip()) > 60)
        if is_head:
            if buf:
                out.append((current, "\n".join(buf)))
            current, buf = s, []
        else:
            buf.append(ln)
    if buf:
        out.append((current, "\n".join(buf)))
    return out, current


def windows(text: str, size: int, overlap: int) -> list[str]:
    """Token windows, measured in the tokenizer's units rather than characters."""
    ids = ENC.encode(text)
    return [ENC.decode(ids[i:i + size]) for i in range(0, len(ids), size - overlap)
            if ids[i:i + size]]


def chunk_document(path: Path, doc: str, title: str,
                   size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """One PDF in, a list of chunks out. The only public function here."""
    size = size or settings.chunk_tokens
    overlap = overlap or settings.chunk_overlap
    pages = strip_furniture(read_pages(path))
    out: list[Chunk] = []
    carry = ""
    for page_no, body in enumerate(pages, 1):
        if not body.strip():
            continue
        blocks, carry = sections(body, carry)
        for j, (sec, part) in enumerate(blocks):
            for w in windows(part, size, overlap):
                out.append(Chunk(id=f"{doc}#p{page_no}#{len(out)}", doc=doc, title=title,
                                 page=page_no, section=sec, text=w))
    log.info("%s: %d pages -> %d chunks", doc, len(pages), len(out))
    return out
