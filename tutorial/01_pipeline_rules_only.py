# =====================================================================
#  01_pipeline_rules_only.py
# =====================================================================
#
#  A tiny, dependency-free version of the OncoTimeline pipeline. It does
#  the same steps as the full project, but with the simplest possible
#  code, so the whole flow fits on one page and is easy to follow.
#
#  Run it with:  python tutorial/01_pipeline_rules_only.py
#  It needs nothing installed: no torch, no internet. Plain Python.
#
#  The full project replaces two of these simple pieces with real models:
#     the keyword "risk formula"   -> a trained Transformer
#     the word-matching dictionary -> a Sentence-BERT embedding model
#  The shape of the pipeline is identical. This file is the place to
#  understand the logic before reading the model-based version.
# =====================================================================

import math


# ---------------------------------------------------------------------
# The clinical note (free text, the way it might be written by a clinician)
# ---------------------------------------------------------------------
note = """
Patient diagnosed with Stage III NSCLC in January 2023.
Started Cisplatin chemotherapy in March 2023.
Tumor size reduced by 40 percent in September 2023.
Disease progression observed in January 2024.
"""


# ---------------------------------------------------------------------
# A small code lookup
# ---------------------------------------------------------------------
# In hospitals every diagnosis and drug has a standard code so different
# systems agree on what is meant. ICD-10 is one such coding system for
# diagnoses. Here is a small lookup: if a word on the left appears, we
# attach the standard name and code on the right.
CODES = {
    "nsclc":         ("Non-small cell lung cancer", "C34.9"),
    "lung cancer":   ("Non-small cell lung cancer", "C34.9"),
    "breast cancer": ("Breast cancer", "C50.9"),
    "colorectal":    ("Colorectal cancer", "C18.9"),
    "cisplatin":     ("Cisplatin chemotherapy", None),
    "carboplatin":   ("Carboplatin chemotherapy", None),
    "radiotherapy":  ("Radiotherapy", None),
    "surgery":       ("Surgical resection", None),
    "pembrolizumab": ("Pembrolizumab", None),
}


# ---------------------------------------------------------------------
# Find the date inside a sentence
# ---------------------------------------------------------------------
# Notes say "January 2023". The "2023-01" form (year then month) is easier
# to sort, so month names are translated to numbers.
MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

def find_date(sentence):
    """Return a 'Month Year' found in the sentence as '2023-01', or None."""
    words = sentence.lower().replace(",", " ").split()
    for i, word in enumerate(words):
        if word in MONTHS and i + 1 < len(words):
            year = words[i + 1]
            if year.isdigit() and len(year) == 4:
                return year + "-" + MONTHS[word]
    return None


# ---------------------------------------------------------------------
# Decide what kind of event a sentence describes
# ---------------------------------------------------------------------
# This labels a sentence by looking for a few signal words.
def find_event_type(sentence):
    s = sentence.lower()
    if "diagnos" in s:
        return "diagnosis"
    if "progress" in s or "metasta" in s:
        return "progression"
    if "reduced" in s or "response" in s or "remission" in s or "stable" in s:
        return "response"
    if ("chemo" in s or "cisplatin" in s or "carboplatin" in s
            or "radio" in s or "surgery" in s or "pembrolizumab" in s
            or "started" in s):
        return "treatment"
    return None


# ---------------------------------------------------------------------
# Extraction: turn the note into a list of events
# ---------------------------------------------------------------------
# Go through the note sentence by sentence. Keep a sentence as an event
# only if it has both a date and a recognised event type.
def extract(note_text):
    events = []
    for sentence in note_text.split("."):
        sentence = sentence.strip()
        if not sentence:
            continue
        date = find_date(sentence)
        kind = find_event_type(sentence)
        if date and kind:
            events.append({
                "date": date,
                "type": kind,
                "text": sentence,
                "concept": None,
                "code": None,
            })
    return events


# ---------------------------------------------------------------------
# Harmonization: attach the standard name and code
# ---------------------------------------------------------------------
def add_codes(events):
    for ev in events:
        text = ev["text"].lower()
        for keyword, (concept, code) in CODES.items():
            if keyword in text:
                ev["concept"] = concept
                ev["code"] = code
                break
    return events


# ---------------------------------------------------------------------
# Timeline: put the events in date order
# ---------------------------------------------------------------------
# "2023-01" sorts before "2023-09" as plain text, so a normal sort gives
# a chronological timeline.
def build_timeline(events):
    return sorted(events, key=lambda ev: ev["date"])


# ---------------------------------------------------------------------
# Risk score: one number summarising how concerning the trajectory is
# ---------------------------------------------------------------------
# The full project uses a trained Transformer here. To keep this version
# transparent, a simple points formula is used instead. It follows the
# same idea the real model learns from data:
#   progression pushes risk up a lot, treatments push it up a little, and
#   a good response pushes it down.
def risk_points(events):
    points = -0.5
    for ev in events:
        if ev["type"] == "progression":
            points += 1.6
        elif ev["type"] == "treatment":
            points += 0.4
        elif ev["type"] == "response":
            text = ev["text"].lower()
            if "complete" in text or "no evidence" in text:
                points -= 1.1
            elif "reduced" in text or "partial" in text:
                points -= 0.55
    return points

def squash(points):
    """Map any number to a probability between 0 and 1 (the sigmoid function).

    Large positive points approach 1, large negative approach 0. This is the
    same final step the neural network in the full project uses.
    """
    return 1 / (1 + math.exp(-points))

def risk_score(events):
    return squash(risk_points(events))


# ---------------------------------------------------------------------
# Explain: which event mattered most
# ---------------------------------------------------------------------
# Remove one event, score again, and measure how much the risk changed.
# A large change means that event mattered a lot. The full project calls
# this leave-one-out attribution.
def explain(events):
    base = risk_score(events)
    rows = []
    for i in range(len(events)):
        without = events[:i] + events[i + 1:]
        rows.append((events[i], base - risk_score(without)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return base, rows


# ---------------------------------------------------------------------
# Run the whole thing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  OncoTimeline tutorial 1: rules only")
    print("=" * 55)

    events = extract(note)
    events = add_codes(events)
    events = build_timeline(events)

    print("\nTimeline:")
    for ev in events:
        concept = ev["concept"] or "(no standard name)"
        code = f"  [{ev['code']}]" if ev["code"] else ""
        print(f"  {ev['date']}  {ev['type']:<11} {concept}{code}")

    risk = risk_score(events)
    print(f"\nPredicted trajectory risk: {risk:.0%}")

    base, rows = explain(events)
    print("\nWhy? (how much each event changed the risk)")
    for ev, contribution in rows:
        arrow = "raises" if contribution > 0 else "lowers"
        print(f"  {ev['date']}  {ev['type']:<11} {arrow} risk by {abs(contribution):.0%}")
