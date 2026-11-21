"""Here-document lines, read the way bash reads them, for the shell quote walk.

bash reads a heredoc body line by line after the line that holds its ``<<``, up to
the first line equal to its tag, and the body's characters open no quote for the
text after it. :func:`shell_normalizer_walk._iter_shell_chars` reads each ``<<`` and
each closing line through these helpers. They read raw lines only, and never look
for the close of a ``$(...)`` or a backtick.
"""

from __future__ import annotations

from .shell_normalizer_escapes import _decode_ansi_c_body

#: Characters that end the tag word after ``<<`` when they stand outside quotes.
_TAG_ENDS = frozenset(" \t\n;&|()<>`")


def heredoc_opener(text: str, at: int) -> "tuple[str, bool, bool] | None":
    """``(tag, strip tabs, quoted)`` of the heredoc whose ``<<`` starts at *at*, else None.

    None for a ``<<<`` here-string, and for a ``<<`` shift inside a ``((`` that is
    still open on that line.
    """
    if not text.startswith("<<", at) or text.startswith("<<<", at) or text[at - 1 : at] == "<":
        return None
    line = text[text.rfind("\n", 0, at) + 1 : at]
    if line.count("((") > line.count("))"):
        return None
    return heredoc_tag(text, at + 2)


def heredoc_tag(text: str, i: int) -> "tuple[str, bool, bool]":
    """``(tag, strip tabs, quoted)`` of the heredoc whose ``<<`` ends at *i*, with its
    quotes removed as bash removes them."""
    strip = text.startswith("-", i)
    i += strip
    while text[i : i + 1] in (" ", "\t"):
        i += 1
    tag: "list[str]" = []
    quoted = False
    quote = ""
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            elif quote == '"' and ch == "\\" and text[i + 1 : i + 2] in ('"', "\\", "$", "`"):
                i += 1
                tag.append(text[i])
            else:
                tag.append(ch)
        elif ch in ("'", '"'):
            quote, quoted = ch, True
        elif ch == "$" and text[i + 1 : i + 2] == "'":
            # bash decodes an ANSI-C tag before it compares lines with it:
            # ``$'e\x6ff'`` closes on ``eof``
            end = i + 2
            while end < len(text) and text[end] != "'":
                end += 2 if text[end] == "\\" else 1
            tag.append(_decode_ansi_c_body(text[i + 2 : min(end, len(text))]))
            quoted, i = True, end
        elif ch == "$" and text[i + 1 : i + 2] == '"':
            pass  # ``$"EOF"`` names the tag ``EOF``
        elif ch == "\\":
            quoted = True
            i += 1
            tag.append(text[i : i + 1])
        elif ch in _TAG_ENDS:
            break
        else:
            tag.append(ch)
        i += 1
    return "".join(tag), strip, quoted


def heredoc_closing_line(
    text: str, at: int, tag: str, strip: bool, quoted: bool
) -> "tuple[int, int] | None":
    """``(start, end)`` of the closing line of a body that starts at *at*, else None.

    *end* is the offset of the newline after that line, or the end of *text*. In a
    body whose tag is not quoted, bash joins a line that ends in an odd run of
    backslashes to the next one before it compares the line with the tag.
    """
    while at < len(text):
        newline = text.find("\n", at)
        end = len(text) if newline < 0 else newline
        line = text[at:end]
        while not quoted and newline >= 0 and (len(line) - len(line.rstrip("\\"))) % 2:
            newline = text.find("\n", end + 1)
            joined = len(text) if newline < 0 else newline
            line, end = line[:-1] + text[end + 1 : joined], joined
        if (line.lstrip("\t") if strip else line) == tag:
            return at, end
        at = end + 1
    return None


def heredoc_bodies(
    text: str, at: int, tags: "list[tuple[str, bool, bool]]"
) -> "list[tuple[int, int]]":
    """``(start, end)`` of the bodies of *tags*, read one after another from *at*.

    *end* is where a body's closing line ends. A body that never closes ends at the
    end of *text*, as in bash, and no body after it is read.
    """
    spans: "list[tuple[int, int]]" = []
    for tag in tags:
        line = heredoc_closing_line(text, at, *tag)
        spans.append((at, len(text) if line is None else line[1]))
        if line is None:
            break
        at = line[1] + 1
    return spans


def heredoc_body_spans(text: str) -> "list[tuple[int, int]]":
    """``(start, end)`` of each closed heredoc body that a walk of *text* from offset
    0 meets outside quotes, in order; *end* is where its closing line ends, and the
    walk reads the text from there unquoted."""
    if "<<" not in text:
        return []
    from .shell_normalizer_walk import _iter_shell_chars  # circular: the walk imports this

    spans: "list[tuple[int, int]]" = []
    for _step in _iter_shell_chars(text, bodies=spans):
        pass
    return spans
