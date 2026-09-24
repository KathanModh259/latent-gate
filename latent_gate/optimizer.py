"""
TokenOptimizer — deterministic, extractive-first input-token reduction.

Stages run safest-first; lossy stages only run when the level (or a token
budget) asks for them:

  1. protect   Lock fenced code, inline code, URLs and quoted strings behind
               placeholders so no later stage can alter them.
  2. lossless  Normalize whitespace/Unicode, strip ANSI codes, minify JSON
               (whitespace only — number and string spellings are untouched),
               fold exact-duplicate lines and paragraphs.
  3. collapse  Fold runs of log lines that differ only in numbers/ids.
  4. filler    Remove politeness/hedging phrases inside sentences.
  5. select    Only when over a token budget: keep the sentences most relevant
               to the question (BM25 + requirement cues), in original order.

Guarantees:
  - The result never has more tokens than the input.
  - Protected spans are restored byte-for-byte.
  - Same input -> same output, so provider-side prompt caching keeps working
    (LLM rewriting breaks it, since each call phrases things differently).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

LEVELS = ("lossless", "balanced", "aggressive")
DEFAULT_AGGRESSIVE_RATIO = 0.5

# ============================================================================
# Token counting
# ============================================================================


@lru_cache(maxsize=1)
def _get_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # not installed, or offline with no cached BPE file
        return None


_PIECE_RE = re.compile(r"[^\W\d_]+|\d{1,3}|[^\w\s]+|\s+")


def _heuristic_tokens(text: str) -> int:
    """
    BPE-shaped estimate used when tiktoken is unavailable.

    Calibrated against o200k_base on this repo's code, docs and JSON
    (mean abs error ~6%, worst ~15%): words ~1 token per 8 chars, punctuation
    runs ~1 per 3 chars, digit groups of 3, multi-char whitespace runs 1.
    """
    n = 0
    for m in _PIECE_RE.finditer(text):
        piece = m.group()
        first = piece[0]
        if first.isalpha():
            n += 1 + (len(piece) - 1) // 8
        elif first.isdigit():
            n += 1
        elif first.isspace():
            n += 1 if len(piece) > 1 else 0
        else:
            n += math.ceil(len(piece) / 3)
    return n


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken (o200k_base) when available, else a calibrated estimate."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return _heuristic_tokens(text)


def token_counter_name() -> str:
    return "tiktoken/o200k_base" if _get_encoder() is not None else "heuristic"


# ============================================================================
# Critical facts (used to verify that a rewrite didn't drop information)
# ============================================================================

_URL_RE = re.compile(r"\bhttps?://[^\s<>\"'`]+")
_FACT_RES = [
    _URL_RE,
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # emails
    re.compile(r"`[^`\n]+`"),  # inline code
    re.compile(r"(?<![\w.])\d+(?:[.,:/-]\d+)*%?"),  # numbers, versions, dates, times
    re.compile(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b"),  # snake_case
    re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b"),  # camelCase
    re.compile(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b"),  # PascalCase
    re.compile(r"\b[A-Z]{2,}[0-9]*\b"),  # ACRONYMS
    re.compile(
        r"\b[\w-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|md|txt|csv|html|css|go|rs|java|sql|toml|sh)\b"
    ),
]


def extract_facts(text: str) -> set:
    """Spans a faithful rewrite must keep verbatim: numbers, identifiers, URLs, code..."""
    facts = set()
    for rx in _FACT_RES:
        for m in rx.finditer(text):
            facts.add(m.group().rstrip(".,;:!?)"))
    facts.discard("")
    return facts


def missing_facts(original: str, candidate: str) -> List[str]:
    """Facts present in `original` that `candidate` lost (sorted for stable output)."""
    return sorted(f for f in extract_facts(original) if f not in candidate)


# ============================================================================
# Filler removal
# ============================================================================


# Parenthetical hedges only (comma-delimited): "Use caching if possible" is a real
# soft preference and must stay; ", if at all possible," is pure padding.
_HEDGE_RE = re.compile(
    r",\s*(?:if (?:at all )?possible|if you (?:can|could|don'?t mind|do not mind)|"
    r"when you (?:get|have) a (?:chance|moment|minute)|at your (?:earliest )?convenience)\s*,",
    re.I,
)
_TRAILING_HEDGE_RE = re.compile(
    r"\s*,\s*(?:when you (?:get|have) a (?:chance|moment|minute)|at your (?:earliest )?convenience)"
    r"(?=\s*[.!?]*$)",
    re.I,
)
# Request wrappers at the start of a sentence (after an optional bullet marker)
_LEAD_RE = re.compile(
    r"(^\s*(?:[-*•]\s+|\d+[.)]\s+)?|(?<=[.!?] ))"
    r"(?:(?:could|can|would|will) you(?:\s+(?:please|kindly))*|please|kindly|"
    r"i(?: would|'d) (?:really )?(?:like|love) (?:for )?you to|i (?:want|need) you to|"
    r"i was wondering if you could|do you think you could|"
    r"(?:is|would) it be possible (?:for you )?to|is it possible (?:for you )?to|"
    r"i(?: would|'d) (?:really )?appreciate it if you could)"
    r"[\s,]+([a-z])",
    re.I,
)
_APPRECIATE_RE = re.compile(
    r",?\s*(?:and\s+)?i(?: would|'d)? (?:really |greatly |truly )?appreciate "
    r"(?:your|any|the) help(?: with (?:this|it|that))?",
    re.I,
)
_TRAILING_THANKS_RE = re.compile(
    r"\s*[,;–—-]\s*(?:thanks|thank you)(?: (?:so|very) much)?(?: in advance)?\s*[!.]*$", re.I
)
_WORD_FILLER_RE = re.compile(r"\b(?:kindly|please)\s+(?=[a-z])", re.I)
_TRAILING_PLEASE_RE = re.compile(r",?\s*\bplease\b(?=\s*[.!?]*$)", re.I)
_TO_ME_RE = re.compile(r"\b(explain|describe)\s+(?:to\s+)?me\b", re.I)
_INTENSE_DETAIL_RE = re.compile(
    r"\bin (?:great|full|much|a lot of|extreme|lots of) (detail|depth)\b", re.I
)
# "explain what the differences are between A and B" -> "explain the differences between A and B"
_WHAT_ARE_RE = re.compile(
    r"\b(explain|describe|list|compare|summari[sz]e|outline)((?: in (?:detail|depth))?) "
    r"what (the [^,.?!\ue000]{1,60}?) (?:are|is) (between|of|in|for|among)\b",
    re.I,
)
# Meaning-identical plain-English replacements (deliberately short list: e.g.
# "with respect to" is NOT here — "derivative with respect to x" is precise math)
_WORDY = [
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bdue to the fact that\b", re.I), "because"),
    (re.compile(r"\bfor the purpose of\b", re.I), "for"),
    (re.compile(r"\bat this point in time\b", re.I), "now"),
    (re.compile(r"\bin the event that\b", re.I), "if"),
]
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])(?=\s|$)")  # not ".env", ":8080"
_DUP_COMMA_RE = re.compile(r",(?:\s*,)+")
_COMMA_BEFORE_END_RE = re.compile(r",\s*([.!?])")
_MULTI_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}")


def strip_filler_phrases(line: str) -> str:
    """
    Remove politeness/hedging filler *inside* a line, keeping its meaning.

    Runs on every prose line at the "balanced" and "aggressive" levels, so it
    must never drop content words the LLM needs. Protected spans (code, URLs,
    quotes) arrive as opaque placeholders and are never affected.

    Deliberately kept: soft preferences ("use caching if possible"), scope words
    ("in detail", "step by step", "only", "exactly"), and anything not in a
    known-safe pattern — a missed saving costs a few tokens, a wrong deletion
    changes the task.

    Example:
        "Could you please, if at all possible, kindly explain to me in great detail
         what the main differences are between TCP and UDP protocols?"
        -> "Explain in detail the main differences between TCP and UDP protocols?"
    """
    s = _HEDGE_RE.sub(" ", line)
    s = _TRAILING_HEDGE_RE.sub("", s)
    s = _APPRECIATE_RE.sub("", s)
    s = _TRAILING_THANKS_RE.sub("", s)
    s = _LEAD_RE.sub(lambda m: m.group(1) + m.group(2).upper(), s)
    s = _WORD_FILLER_RE.sub("", s)
    s = _TRAILING_PLEASE_RE.sub("", s)
    s = _TO_ME_RE.sub(r"\1", s)
    s = _INTENSE_DETAIL_RE.sub(r"in \1", s)
    s = _WHAT_ARE_RE.sub(r"\1\2 \3 \4", s)
    for rx, repl in _WORDY:
        s = rx.sub(repl, s)
    # Tidy what the removals left behind, without touching leading indentation
    s = _MULTI_SPACE_RE.sub(" ", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _DUP_COMMA_RE.sub(",", s)
    s = _COMMA_BEFORE_END_RE.sub(r"\1", s)
    s = s.rstrip()
    return "" if not any(ch.isalnum() or ch == _PH_OPEN for ch in s) else s


# ============================================================================
# Protection (placeholders)
# ============================================================================

_PH_OPEN, _PH_CLOSE = "\ue000", "\ue001"
_PH_RE = re.compile(f"{_PH_OPEN}(\\d+){_PH_CLOSE}")
_FENCE_RE = re.compile(r"(?ms)^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_QUOTE_RE = re.compile(r'"[^"\n]{1,300}"')
_LOG_FENCE_LANGS = {"", "log", "logs", "text", "txt", "console", "output", "stderr", "stdout"}
_JSON_FENCE_LANGS = {"json", "jsonc", "json5"}


class _Vault:
    """Holds protected spans; `lock` swaps a span for an opaque placeholder."""

    def __init__(self):
        self.spans: List[str] = []

    def lock(self, span: str) -> str:
        self.spans.append(span)
        return f"{_PH_OPEN}{len(self.spans) - 1}{_PH_CLOSE}"

    def restore(self, text: str) -> str:
        for _ in range(4):  # spans may nest (e.g. a URL inside a quote)
            if _PH_OPEN not in text:
                break
            text = _PH_RE.sub(lambda m: self.spans[int(m.group(1))], text)
        return text


def _minify_json_text(src: str) -> str:
    """Remove insignificant whitespace from JSON text without re-serializing values."""
    out, in_str, esc = [], False, False
    for ch in src:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


def _minify_json_spans(text: str, vault: Optional[_Vault]) -> str:
    """Find multi-line JSON values starting at a line start; minify (and optionally lock) them."""
    decoder = json.JSONDecoder()
    out, pos = [], 0
    for m in re.finditer(r"(?m)^[ \t]*([\[{])", text):
        start = m.start(1)
        if start < pos:
            continue
        try:
            _, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        span = text[start:end]
        if "\n" not in span:
            continue
        mini = _minify_json_text(span)
        out.append(text[pos:start])
        out.append(vault.lock(mini) if vault else mini)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _clean_fence(block: str, level: str) -> str:
    """Lossless cleanup inside a fenced block; log-style fences also get run collapsing."""
    lines = block.split("\n")
    opener, closer, body = lines[0], lines[-1], lines[1:-1]
    lang = opener.strip().lstrip("`~").strip().lower()
    body = [ln.rstrip() for ln in body]
    body = _collapse_blank_runs(body, keep=1)
    if lang in _JSON_FENCE_LANGS or (lang == "" and body and body[0].lstrip()[:1] in "{["):
        joined = "\n".join(body)
        mini = _minify_json_spans(joined, None)
        body = mini.split("\n")
    if lang in _LOG_FENCE_LANGS:
        body = _collapse_line_runs(body, templated=level != "lossless", all_lines_are_logs=True)
    return "\n".join([opener.rstrip(), *body, closer.rstrip()])


def _protect(text: str, vault: _Vault, level: str) -> str:
    text = _FENCE_RE.sub(lambda m: vault.lock(_clean_fence(m.group(0), level)), text)
    # An unterminated fence protects everything after it (pasted code without closing ```)
    idx = text.find("```")
    if idx != -1:
        line_start = text.rfind("\n", 0, idx) + 1
        text = text[:line_start] + vault.lock(text[line_start:].rstrip())
    text = _minify_json_spans(text, vault)
    text = _INLINE_CODE_RE.sub(lambda m: vault.lock(m.group(0)), text)

    def _lock_url(m):
        url = m.group(0)
        trail = len(url) - len(url.rstrip(".,;:!?)"))
        return vault.lock(url[: len(url) - trail]) + url[len(url) - trail :]

    text = _URL_RE.sub(_lock_url, text)
    return _QUOTE_RE.sub(lambda m: vault.lock(m.group(0)), text)


# ============================================================================
# Lossless normalization & repetition
# ============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_INNER_SPACES_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
_LOG_LINE_RE = re.compile(
    r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b"
    r"|^\s*\[?\d{4}-\d{2}-\d{2}"
    r"|\b\d{2}:\d{2}:\d{2}"
    r"|^\s*at \S+\(.*\)$"
)
_TEMPLATE_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|0x[0-9a-f]+|[0-9a-f]{12,}|\d+",
    re.I,
)


def _pre_normalize(text: str) -> str:
    """Safe everywhere, including inside code."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ANSI_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    return text.replace("\u00a0", " ")


def _normalize_prose(text: str) -> str:
    """Whitespace cleanup that preserves leading indentation (unfenced code stays valid)."""
    lines = [_INNER_SPACES_RE.sub(" ", ln.rstrip()) for ln in text.split("\n")]
    return "\n".join(_collapse_blank_runs(lines, keep=1)).strip()


def _collapse_blank_runs(lines: List[str], keep: int) -> List[str]:
    out, blanks = [], 0
    for ln in lines:
        blanks = blanks + 1 if not ln.strip() else 0
        if blanks <= keep:
            out.append(ln)
    return out


def _collapse_line_runs(
    lines: List[str], templated: bool, all_lines_are_logs: bool = False
) -> List[str]:
    """
    Fold consecutive repeats. Exact repeats become `line [×N]` (lossless).
    With `templated`, runs of 3+ LOG lines differing only in numbers/ids keep
    first and last. Never applied to non-log prose: a list like
    "req 1..req 8" must not be folded.
    """
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        j = i + 1
        while j < len(lines) and lines[j] == line and line.strip():
            j += 1
        if j - i >= 2:
            out.append(f"{line} [×{j - i}]")
            i = j
            continue
        is_log = all_lines_are_logs or bool(_LOG_LINE_RE.search(line))
        if templated and is_log and len(line.strip()) >= 12:
            key = _TEMPLATE_RE.sub("#", line.strip())
            j = i + 1
            while j < len(lines) and _TEMPLATE_RE.sub("#", lines[j].strip()) == key:
                j += 1
            if j - i >= 3:
                out.extend([line, _summarize_run(lines[i + 1 : j - 1]), lines[j - 1]])
                i = j
                continue
        out.append(line)
        i += 1
    return out


_SLOT_LABEL_RE = re.compile(r"[\w\-=:/.]*$")


def _summarize_run(run: List[str], max_fields: int = 4) -> str:
    """
    Describe folded log lines by the range of each field that varies, with its label:
    "[… 58 similar lines: id=5001…5058, worker-1…3 …]". Keeps what changed, drops repetition.
    """
    first = run[0].strip()
    matches = list(_TEMPLATE_RE.finditer(first))
    slot_values = [[m.group() for m in _TEMPLATE_RE.finditer(ln.strip())] for ln in run]
    fields = []
    for k, m in enumerate(matches):
        vals = [s[k] for s in slot_values if len(s) > k]
        if len(set(vals)) == 1:
            continue
        if all(v.isdigit() for v in vals):
            lo, hi = min(vals, key=int), max(vals, key=int)
        else:
            lo, hi = vals[0], vals[-1]
        label = _SLOT_LABEL_RE.search(first[: m.start()]).group()
        fields.append(f"{label}{lo}…{hi}")
        if len(fields) == max_fields:
            break
    detail = f": {', '.join(fields)}" if fields else ""
    return f"[… {len(run)} similar lines{detail} …]"


def _drop_duplicate_paragraphs(text: str) -> str:
    seen, out = set(), []
    for para in text.split("\n\n"):
        key = para.strip()
        if len(key) >= 40 and key in seen:
            continue
        seen.add(key)
        out.append(para)
    return "\n\n".join(out)


_CODE_LINE_RE = re.compile(
    r"^(\t| {4,})|[;{}]\s*$|^\s*(def|class|import|from|return|if|for|while|const|let|var|function)\b"
)


# Sentences that are *entirely* social niceties. Anchored on both ends, so a sentence
# with any real content ("Thanks — also add tests") is never matched.
_PLEASANTRY_RE = re.compile(
    r"^(?:"
    r"(?:hi|hello|hey|greetings|good (?:morning|afternoon|evening))"
    r"(?: there| all| everyone| team| guys| folks)?"
    r"|(?:thanks|thank you|thx|many thanks|cheers)(?: (?:so|very) much)?(?: in advance)?"
    r"(?: for (?:your|the|any) help)?"
    r"|i hope (?:you'?re|you are|this (?:message |email )?finds you) (?:doing )?(?:well|good|great)"
    r"|(?:any|your) help (?:would be|is|will be) (?:greatly |much |really )?appreciated"
    r"|(?:best|kind|warm) regards|regards|sincerely"
    r")[\s!.,:;)(-]*$",
    re.I,
)


def _drop_pleasantries(line: str) -> str:
    kept = [s for s in _split_sentences(line) if not _PLEASANTRY_RE.match(s.strip())]
    return " ".join(kept)


def _apply_filler(text: str) -> str:
    out = []
    for ln in text.split("\n"):
        if ln.strip() and not _CODE_LINE_RE.search(ln):
            ln = strip_filler_phrases(_drop_pleasantries(ln))
            if not ln.strip():
                continue
        out.append(ln)
    return "\n".join(out)


# ============================================================================
# Query-aware extractive selection
# ============================================================================

_STOPWORDS = frozenset("""a an the and or but if then else of to in on at by for with from into over
    is are was were be been being am do does did done have has had having it its
    this that these those there here what which who whom whose when where why how
    i me my we our you your he she they them their his her as so than too very can
    could would should will shall may might must just also about not no yes all any
    some such only own same other more most each few both up down out off again
    further once please kindly""".split())
_CUE_RE = re.compile(
    r"\b(must|should|need|needs|required?|never|always|don'?t|do not|ensure|only|exactly|"
    r"return|output|format|include|avoid|without|at least|at most|limit|deadline)\b",
    re.I,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_ROLE_RE = re.compile(r"^\s*(?:please\s+)?(?:act as|you are|imagine you)\b", re.I)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:act as|build|write|create|implement|explain|fix|review|summari[sz]e|"
    r"make|generate|add|find|list|compare|design|refactor|convert|translate|analy[sz]e|"
    r"describe|debug|optimi[sz]e|rewrite|draft|answer|tell|show|give|help|update|remove)\b",
    re.I,
)
_HEADING_RE = re.compile(r"^\s*(#{1,6}\s|\[Doc \d+\])")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[\ue000])")
_ABBREV_RE = re.compile(r"\b(e\.g|i\.e|etc|vs|mr|mrs|dr|no|fig)\.$", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _terms(text: str) -> List[str]:
    return [_stem(w) for w in (t.lower() for t in _WORD_RE.findall(text)) if w not in _STOPWORDS]


def _split_sentences(line: str) -> List[str]:
    parts, buf = [], ""
    for piece in _SENT_SPLIT_RE.split(line):
        buf = f"{buf} {piece}" if buf else piece
        if not _ABBREV_RE.search(buf):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


@dataclass
class _Unit:
    section: int
    block: int
    line: int
    text: str
    tokens: int
    locked: bool
    score: float = 0.0
    pos: int = 0
    relevant: bool = True


def _build_units(sections: Sequence[str], vault: _Vault, counter) -> List[_Unit]:
    units = []
    for si, sec in enumerate(sections):
        for bi, block in enumerate(sec.split("\n\n")):
            for li, line in enumerate(block.split("\n")):
                if not line.strip():
                    continue
                structural = (
                    _BULLET_RE.match(line)
                    or _HEADING_RE.match(line)
                    or _CODE_LINE_RE.search(line)
                    or _PH_RE.fullmatch(line.strip())
                )
                pieces = [line] if structural else _split_sentences(line)
                for piece in pieces:
                    whole_span = bool(_PH_RE.fullmatch(piece.strip()))  # a code block etc.
                    units.append(
                        _Unit(
                            si,
                            bi,
                            li,
                            piece,
                            counter(vault.restore(piece)),
                            locked=whole_span or bool(_HEADING_RE.match(piece)),
                        )
                    )
    return units


def _score_units(units: List[_Unit], question: str, vault: _Vault) -> None:
    """BM25 relevance to the question (or to the document's recurring terms), plus cues."""
    docs = [_terms(vault.restore(u.text)) for u in units]
    n = len(docs)
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    query = _terms(question)
    if not query:  # no question: the terms that recur across the text are its topic
        query = [t for t, c in sorted(df.items(), key=lambda kv: (-kv[1], kv[0])) if c >= 2][:15]
    avg_len = sum(len(d) for d in docs) / max(n, 1) or 1.0
    k1, b = 1.2, 0.75
    raw = []
    for d in docs:
        s = 0.0
        for t in set(query):
            tf = d.count(t)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
        raw.append(s)
    top = max(raw) or 1.0
    # With a question, relevance must dominate: an off-topic "Visitors must register"
    # shouldn't outrank an on-topic sentence just for containing a cue word.
    weight = 0.35 if question.strip() else 1.0
    # No position bonus: "first sentence = the ask" rewarded preambles like
    # "I'm working on a project…"; imperative/bullet/question cues find the ask directly.
    for i, u in enumerate(units):
        text = vault.restore(u.text)
        bonus = 1.5 if _CUE_RE.search(text) else 0.0
        bonus += 2.0 if _IMPERATIVE_RE.match(text) else 0.0  # "Build …", "Fix …": the ask
        bonus += 2.5 if _BULLET_RE.match(u.text) else 0.0  # listed requirements
        bonus += 1.0 if "?" in text else 0.0
        bonus += 0.5 if any(ch.isdigit() for ch in text) else 0.0
        bonus += 1.0 if _PH_OPEN in u.text else 0.0
        u.score = 3.0 * raw[i] / top + weight * bonus
        u.relevant = raw[i] > 0


def _select(
    sections: Sequence[str],
    labels: Sequence[Optional[str]],
    question: str,
    budget: int,
    vault: _Vault,
    counter,
) -> Tuple[List[str], int]:
    """Keep the highest-value units that fit `budget`; returns (sections, units_dropped)."""
    units = _build_units(sections, vault, counter)
    if not units:
        return list(sections), 0
    for i, u in enumerate(units):
        u.pos = i
    _score_units(units, question, vault)

    # A section's label (e.g. "[Doc 2]: ") is paid for only once that section
    # contributes content — dropped documents cost nothing.
    label_cost = [counter(f"{lb}: ") if lb else 0 for lb in labels]
    opened = set()
    keep, used = set(), 0

    def take(u: _Unit) -> bool:
        nonlocal used
        cost = u.tokens + (label_cost[u.section] if u.section not in opened else 0)
        if not u.locked and used + cost > budget:
            return False
        keep.add(id(u))
        opened.add(u.section)
        used += cost
        return True

    # In a single prompt, the first instruction or question IS the task: never drop it.
    # (Role preambles like "Act as an expert…" don't count as the ask.)
    if len(sections) == 1:
        for u in units:
            text = vault.restore(u.text)
            if not u.locked and not _ROLE_RE.match(text):
                if _IMPERATIVE_RE.match(text) or text.rstrip().endswith("?"):
                    u.locked = True
                    break
    for u in units:
        if u.locked:
            take(u)
    # For RAG chunks with a question, the budget is a ceiling, not a target:
    # sentences sharing no terms with the question are never sent.
    gate = bool(question.strip()) and len(sections) > 1
    for u in sorted((u for u in units if not u.locked), key=lambda u: (-u.score, u.pos)):
        if gate and not u.relevant:
            continue
        take(u)

    rebuilt = []
    for si in range(len(sections)):
        blocks = {}
        for u in units:
            if u.section == si and id(u) in keep:
                blocks.setdefault(u.block, {}).setdefault(u.line, []).append(u.text)
        rebuilt.append(
            "\n\n".join(
                "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))
                for _, lines in sorted(blocks.items())
            )
        )
    return rebuilt, len(units) - len(keep)


# ============================================================================
# Public API
# ============================================================================


@dataclass
class OptimizationResult:
    text: str
    original_tokens: int
    optimized_tokens: int
    level: str
    stages: List[Tuple[str, int]] = field(default_factory=list)  # (stage, tokens after)
    units_dropped: int = 0
    counter: str = ""

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.optimized_tokens

    @property
    def compression_ratio(self) -> float:
        return self.original_tokens / max(self.optimized_tokens, 1)

    @property
    def savings_pct(self) -> float:
        return 100.0 * self.tokens_saved / max(self.original_tokens, 1)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "original_tokens": self.original_tokens,
            "optimized_tokens": self.optimized_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round(self.compression_ratio, 2),
            "savings_pct": round(self.savings_pct, 1),
            "level": self.level,
            "stages": [{"stage": s, "tokens": t} for s, t in self.stages],
            "units_dropped": self.units_dropped,
            "token_counter": self.counter,
        }


class TokenOptimizer:
    """
    Deterministic prompt optimizer.

    Levels:
      lossless    whitespace/JSON/duplicate folding only — meaning is unchanged
      balanced    + log-run folding and filler-phrase removal (default)
      aggressive  + question-aware sentence selection to `target_ratio` (default 0.5)

    `max_tokens` (any level) enables selection whenever the text is over budget.
    """

    def __init__(self, level: str = "balanced", counter: Optional[Callable[[str], int]] = None):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.level = level
        self.counter = counter or count_tokens

    def optimize(
        self, text: str, question: str = "", max_tokens: int = 0, target_ratio: float = 0.0
    ) -> OptimizationResult:
        return self._run([text], [None], question, max_tokens, target_ratio)

    def optimize_documents(
        self,
        documents: Sequence[str],
        question: str = "",
        max_tokens: int = 0,
        target_ratio: float = 0.0,
    ) -> OptimizationResult:
        """Optimize RAG chunks jointly: the budget is spent on the most relevant sentences
        across all documents, and documents with nothing relevant are dropped entirely."""
        labels = [f"[Doc {i + 1}]" for i in range(len(documents))]
        return self._run(list(documents), labels, question, max_tokens, target_ratio)

    # ------------------------------------------------------------------

    def _run(self, texts, labels, question, max_tokens, target_ratio) -> OptimizationResult:
        original = "\n\n".join(f"{lb}: {t}" if lb else t for lb, t in zip(labels, texts))
        original_tokens = self.counter(original)
        vault, stages = _Vault(), []

        def measure(stage: str, secs: List[str]) -> None:
            stages.append((stage, self.counter(vault.restore(self._join(secs, labels)))))

        secs = [_pre_normalize(t) for t in texts]
        secs = [_normalize_prose(_protect(t, vault, self.level)) for t in secs]
        secs = [
            _drop_duplicate_paragraphs(
                "\n".join(_collapse_line_runs(s.split("\n"), templated=self.level != "lossless"))
            )
            for s in secs
        ]
        measure("lossless" if self.level == "lossless" else "lossless+collapse", secs)

        if self.level != "lossless":
            secs = [_normalize_prose(_apply_filler(s)) for s in secs]
            measure("filler", secs)

        if not max_tokens and self.level == "aggressive":
            target_ratio = target_ratio or DEFAULT_AGGRESSIVE_RATIO
        budget = max_tokens or (int(original_tokens * target_ratio) if target_ratio else 0)
        dropped = 0
        if budget and stages[-1][1] > budget:
            secs, dropped = _select(secs, labels, question, budget, vault, self.counter)
            measure("select", secs)

        result_text = vault.restore(self._join(secs, labels)).strip()
        optimized_tokens = self.counter(result_text)
        if optimized_tokens >= original_tokens:  # never make a prompt bigger
            result_text, optimized_tokens = original, original_tokens
        return OptimizationResult(
            text=result_text,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            level=self.level,
            stages=stages,
            units_dropped=dropped,
            counter=token_counter_name(),
        )

    @staticmethod
    def _join(secs: List[str], labels) -> str:
        return "\n\n".join(f"{lb}: {s}" if lb else s for lb, s in zip(labels, secs) if s)


def optimize(
    text: str, question: str = "", level: str = "balanced", **kwargs
) -> OptimizationResult:
    """Convenience wrapper: `optimize(text, question, level="aggressive").text`."""
    return TokenOptimizer(level=level).optimize(text, question=question, **kwargs)
