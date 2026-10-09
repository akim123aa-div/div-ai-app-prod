"""The chunker: token windows, and what counts as furniture on a page.

Run with `pytest tests/unit`. No database, no model, no network: these are the
functions whose output is decided entirely by their input.
"""

from app.chunking import ENC, boilerplate, windows


def test_windows_are_bounded_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    ws = windows(text, size=100, overlap=20)
    ids = [ENC.encode(w) for w in ws]
    assert all(len(w) <= 100 for w in ids)
    assert ids[0][-20:] == ids[1][:20]          # the overlap is the next window's start
    assert ENC.decode(ENC.encode(text)[-5:]) in ws[-1]   # nothing falls off the end


def test_a_line_on_most_pages_is_furniture():
    pages = [f"Annual Report 2022\nPage text number {i}, all different." for i in range(10)]
    assert boilerplate(pages) == {"Annual Report 2022"}


def test_a_short_document_keeps_its_lines():
    # The Lesson 4 bug: with two pages the threshold fell below one page, and every
    # short line counted as furniture. A line that appears once is never furniture.
    pages = ["Refund policy\nRefunds take five days.", "Contact\nWrite to us."]
    assert boilerplate(pages) == set()
