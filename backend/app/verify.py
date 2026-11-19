"""Checks the AI-written parts of an investigation report against the evidence, in code.

A finding passes only if:
  - it cites at least one evidence item, and every cited item exists;
  - every claim ID, customer ID, date and model version in it appears in the cited evidence;
  - every number in it matches a number in the cited evidence (allowing normal rounding, and
    fractions written as percentages: 0.909 may be written 0.91 or 90.9% or 91%);
  - a comparison between two numbers ("0.214 is below 0.2336") points the right way;
  - it doesn't use accusatory words (a score is a reason to review, never proof).

What this does NOT check: that a number is attached to the right label ("6 claims" passes if any
cited item contains a 6), comparisons written without numbers, or whether the reasoning is sound. The reviewer reads the evidence for that.
"""
import re

# IDs, dates and versions are compared as text; everything else that looks like a number, as a number.
TOKEN = re.compile(r"\b(?:CLM|CUS)-\d+\b|\b\d{4}-\d{2}-\d{2}\b|\bv\d+(?:\.\d+)*\b"
                   r"|(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])\d+(?:\.\d+)?%?")
EXACT = re.compile(r"^(?:CLM-|CUS-|\d{4}-\d{2}-\d{2}$|v\d)")
ACCUSATORY = re.compile(r"\b(fraud\w*|guilty|criminal|lied|lying|scam\w*|proves?|proof of)\b", re.I)
MAX_REL_ERROR = 0.02     # "0.23" for 0.2336 is fine; "16" for 15.58 or "1" for 0.909 is not


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.replace("−", "-"))


def _parse(tok: str) -> tuple[float, int, bool]:
    """'57,059' → (57059.0, 0 decimals, not a percent); '90.9%' → (90.9, 1, True)."""
    pct = tok.endswith("%")
    s = tok.rstrip("%").replace(",", "")
    return float(s), (len(s.split(".")[1]) if "." in s else 0), pct


def _close(written: float, decimals: int, actual: float) -> bool:
    if abs(round(actual, decimals) - written) > 1e-9:
        return False
    return actual == 0 or abs(written - actual) / abs(actual) <= MAX_REL_ERROR


def number_matches(tok: str, evidence_numbers: list[float]) -> bool:
    x, d, pct = _parse(tok)
    for a in evidence_numbers:
        a = abs(a)
        if _close(x, d, a) or (pct and _close(x, d, a * 100)):
            return True
    return False


COMPARE = re.compile(r"\b(higher|greater|above|exceeds?|more than|lower|less than|fewer than|below|under)\b", re.I)
BIGGER = ("higher", "greater", "above", "exceed", "exceeds", "more than")


def comparison_problems(text: str) -> list[str]:
    """'X is higher/lower than Y' must be true for the two numbers it compares.

    In testing, the LLM wrote "a higher rate (11.3%) compared to all categories (12.4%)": every number
    was right, so the number check passed it. The two numbers are the nearest one before the word and
    the first one after it, or the first two after it when none comes before (same sentence only).
    """
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\[E\d+\]", " ", text)):
        for m in COMPARE.finditer(sentence):
            before = [t for t in tokens(sentence[:m.start()]) if not EXACT.match(t)]
            after = [t for t in tokens(sentence[m.end():]) if not EXACT.match(t)]
            pair = (before[-1], after[0]) if before and after else tuple(after[:2]) if not before and len(after) >= 2 else None
            if not pair:
                continue
            x, y = (_parse(t)[0] for t in pair)
            bigger = m.group(1).lower() in BIGGER
            if x != y and (x > y) != bigger:
                problems.append(f'says {pair[0]} is "{m.group(1)}" {pair[1]}, which is the wrong way round')
    return problems


def evidence_text(item: dict) -> str:
    return f"{item['id']} {item['title']}: " + "; ".join(f"{f['label']} = {f['value']}" for f in item["facts"])


def check(text: str, cited: list[str], evidence: list[dict]) -> list[str]:
    """Problems with one AI-written statement; an empty list means it passed."""
    by_id = {e["id"]: e for e in evidence}
    problems = []
    if not cited:
        problems.append("cites no evidence")
    unknown = [c for c in cited if c not in by_id]
    if unknown:
        problems.append(f"cites evidence that doesn't exist: {', '.join(unknown)}")
    source = " ".join(evidence_text(by_id[c]) for c in cited if c in by_id)
    exact = {t for t in tokens(source) if EXACT.match(t)}
    numbers = [_parse(t)[0] for t in tokens(source) if not EXACT.match(t)]
    text_wo_cites = re.sub(r"\[E\d+\]", " ", text)
    for tok in tokens(text_wo_cites):
        if EXACT.match(tok):
            if tok not in exact:
                problems.append(f'"{tok}" is not in the cited evidence')
        elif not number_matches(tok, numbers):
            problems.append(f'the number {tok} is not in the cited evidence')
    problems += comparison_problems(text)
    word = ACCUSATORY.search(text)
    if word:
        problems.append(f'uses the word "{word.group(0)}": a score is a reason to review, not proof of wrongdoing')
    return problems


def self_check() -> None:
    ev = [{"id": "E1", "title": "Risk score", "facts": [{"label": "Score", "value": 0.909}, {"label": "Threshold", "value": 0.2336},
                                                        {"label": "Model", "value": "claims-review v1"}]},
          {"id": "E2", "title": "Claim", "facts": [{"label": "Claim", "value": "CLM-021505"}, {"label": "Amount", "value": "57,058.87"},
                                                   {"label": "Claim date", "value": "2025-12-20"}, {"label": "Claims in the last 12 months", "value": 6}]}]
    assert check("The score 0.91 (91%) is above the threshold 0.23 [E1].", ["E1"], ev) == []
    assert check("CLM-021505 was filed on 2025-12-20 for 57,059 [E2].", ["E2"], ev) == []
    assert check("It has 6 claims in the last 12 months [E2], scored by claims-review v1 [E1].", ["E1", "E2"], ev) == []
    assert check("The score is 1 [E1].", ["E1"], ev)                       # too crude a rounding
    assert check("The score 0.909 is high [E2].", ["E2"], ev)              # right number, wrong citation
    assert check("Filed on 2025-12-21 [E2].", ["E2"], ev)                  # wrong date
    assert check("Amount 57,059 [E9].", ["E9"], ev)                        # citation doesn't exist
    assert check("This looks like fraud [E1].", ["E1"], ev)
    assert check("No citation.", [], ev)
    # comparisons (both cases seen in real runs)
    assert comparison_problems("The category has a higher rate (11.3%) compared to all categories (12.4%).")
    assert not comparison_problems("A slightly lower rate (11.3%) than all categories (12.4%).")
    assert not comparison_problems("An anomaly score of 0.625, slightly above the threshold of 0.608.")
    assert not comparison_problems("The score 0.214 is below the review threshold of 0.2336, so no review.")
    assert comparison_problems("The score 0.214 is above the review threshold of 0.2336.")
    assert not comparison_problems("The amount 57,058.87 is much higher than the median 3,661.55.")
    assert not comparison_problems("It is 15.58 times higher than the median.")       # one number: nothing to compare


if __name__ == "__main__":
    self_check()
    print("verify.py self-check passed")
