"""Small python-docx helpers for revising the manuscript in place.

Text passed to these helpers uses a minimal markdown subset: **bold** and
*italic*. When `mark` is True, words that are new relative to the paragraph's
previous text are highlighted yellow (the "changes highlighted" copy).
"""
from __future__ import annotations

import copy
import difflib
import re

from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

_MD = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*)")


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("–", "--")
    return re.sub(r"\s+", " ", s).strip()


def parse_md(text: str):
    segs = []
    for part in _MD.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            segs.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            segs.append((part[1:-1], False, True))
        else:
            segs.append((part, False, False))
    return segs


def remove_all_highlights(doc):
    for el in list(doc.element.body.iter(qn("w:highlight"))):
        el.getparent().remove(el)


def _new_mask(old: str, new: str):
    """Per-character flag: True where `new` text is not carried over from `old`."""
    tok = lambda s: re.findall(r"\S+|\s+", s)
    a, b = tok(norm(old)), tok(new)
    mask = []
    sm = difflib.SequenceMatcher(None, [norm(x) for x in a], [norm(x) for x in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for t in b[j1:j2]:
            is_new = tag != "equal" and t.strip() != ""
            mask.extend([is_new] * len(t))
    # whitespace between two new words is new too
    for i in range(1, len(mask) - 1):
        if not mask[i] and new[i].isspace() and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    return mask


def _template_rpr(p):
    """Base run formatting = the paragraph's LAST text run (plain body text,
    the non-bold part of a figure legend, or an italic table caption)."""
    runs = [r for r in p._p.iter(qn("w:r")) if r.find(qn("w:t")) is not None]
    if not runs:
        return None
    rpr = runs[-1].find(qn("w:rPr"))
    if rpr is None:
        return None
    rpr = copy.deepcopy(rpr)
    for el in rpr.findall(qn("w:highlight")):
        rpr.remove(el)
    return rpr


def set_text(p: Paragraph, md: str, mark: bool, old: str | None = None, all_new=False):
    """Replace a paragraph's text, keeping its paragraph/run base formatting."""
    old_plain = p.text if old is None else old
    rpr = _template_rpr(p)
    for child in list(p._p):
        if child.tag in (qn("w:r"), qn("w:hyperlink"), qn("w:ins"), qn("w:del")):
            p._p.remove(child)
    segs = parse_md(md)
    plain = "".join(s for s, _, _ in segs)
    mask = [True] * len(plain) if all_new else _new_mask(old_plain, plain)
    pos = 0
    for text, bold, italic in segs:
        start = 0
        while start < len(text):
            flag = mask[pos + start]
            end = start
            while end < len(text) and mask[pos + end] == flag:
                end += 1
            run = p.add_run(text[start:end])
            if rpr is not None:
                run._r.insert(0, copy.deepcopy(rpr))
            if bold:
                run.bold = True
            if italic:
                run.italic = True
            if mark and flag:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            start = end
        pos += len(text)


def insert_after(anchor, template: Paragraph, md: str, mark: bool) -> Paragraph:
    """Insert a new paragraph (formatted like `template`) after `anchor`
    (a Paragraph or Table)."""
    el = copy.deepcopy(template._p)
    anchor_el = anchor._p if isinstance(anchor, Paragraph) else anchor._tbl
    anchor_el.addnext(el)
    p = Paragraph(el, template._parent)
    set_text(p, md, mark, old="", all_new=True)
    return p


def clone_table(template: Table, anchor, rows, mark: bool) -> Table:
    """Deep-copy `template`, resize it to len(rows) rows, fill cell text
    (row 0 is the header), and insert it after `anchor`."""
    tbl = copy.deepcopy(template._tbl)
    anchor_el = anchor._p if isinstance(anchor, Paragraph) else anchor._tbl
    anchor_el.addnext(tbl)
    t = Table(tbl, template._parent)
    trs = tbl.findall(qn("w:tr"))
    while len(trs) < len(rows):
        trs[-1].addnext(copy.deepcopy(trs[-1]))
        trs = tbl.findall(qn("w:tr"))
    for tr in trs[len(rows):]:
        tbl.remove(tr)
    for r, values in enumerate(rows):
        cells = t.rows[r].cells
        assert len(cells) == len(values), (len(cells), values)
        for c, v in zip(cells, values):
            set_cell(c, v, mark, all_new=True)
    return t


def set_cell(cell, md: str, mark: bool, all_new=False):
    ps = cell.paragraphs
    for extra in ps[1:]:
        extra._p.getparent().remove(extra._p)
    set_text(ps[0], md, mark, all_new=all_new)
