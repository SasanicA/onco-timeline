# =====================================================================
#  02_pipeline_with_models.py
# =====================================================================
#
#  The same one-page pipeline as 01_pipeline_rules_only.py, but two of the
#  steps now use real models instead of hand-written rules:
#
#     harmonization  ->  Sentence-BERT (maps text to meaning vectors)
#     risk score     ->  a small PyTorch Transformer, trained on the spot
#
#  Run it with:  python tutorial/02_pipeline_with_models.py
#  It prints everything to the terminal.
#
#  Requirements (already in the project): torch, sentence-transformers, numpy.
#  The first run downloads a small (~90 MB) embedding model and caches it. If
#  sentence-transformers is not available, harmonization falls back to a simple
#  word match so the file still runs.
# =====================================================================

import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# STEP 0:  The clinical note (the text we will read)
# ---------------------------------------------------------------------
NOTE = """
Patient diagnosed with Stage III NSCLC in January 2023.
Started Cisplatin chemotherapy in March 2023.
Tumor size reduced by 40 percent in September 2023.
Disease progression observed in January 2024.
"""


# ---------------------------------------------------------------------
# A tiny clinical concept table (official name + ICD-10 code)
# ---------------------------------------------------------------------
# We do NOT keep a synonym list: the AI model matches by MEANING, so it can
# recognise "NSCLC", "lung cancer" or "carcinoma of the lung" on its own.
# Example wordings are shown in the comments only, to help you read the table.
CONCEPTS = [
    ("Non-small cell lung cancer", "C34.9"),  # e.g. "NSCLC", "lung cancer"
    ("Breast cancer",              "C50.9"),  # e.g. "breast carcinoma"
    ("Colorectal cancer",          "C18.9"),  # e.g. "colon cancer"
    ("Cisplatin chemotherapy",     None),     # e.g. "cisplatin"
    ("Radiotherapy",               None),     # e.g. "radiation therapy"
    ("Surgical resection",         None),     # e.g. "surgery", "resection"
    ("Partial response",           None),     # e.g. "tumor reduced", "shrank"
    ("Complete response",          None),     # e.g. "no evidence of disease"
    ("Stable disease",             None),     # e.g. "no change"
    ("Progressive disease",        None),     # e.g. "progression", "new metastases"
]
CONCEPT_NAMES = [c[0] for c in CONCEPTS]


# =====================================================================
# STEP 1 + 2:  EXTRACTION  (still simple keyword rules, kept readable)
# =====================================================================
MONTHS = {"january":"01","february":"02","march":"03","april":"04","may":"05",
          "june":"06","july":"07","august":"08","september":"09","october":"10",
          "november":"11","december":"12"}

def find_date(sentence):
    """Turn 'January 2023' into '2023-01' so dates sort correctly."""
    words = sentence.lower().replace(",", " ").split()
    for i, w in enumerate(words):
        if w in MONTHS and i + 1 < len(words) and words[i+1].isdigit():
            return f"{words[i+1]}-{MONTHS[w]}"
    return None

def find_event_type(sentence):
    """Label the sentence by simple clue words."""
    s = sentence.lower()
    if "diagnos" in s:                               return "diagnosis"
    if "progress" in s or "metasta" in s:            return "progression"
    if any(k in s for k in ["reduced","response","remission","stable"]):
        return "response"
    if any(k in s for k in ["chemo","cisplatin","radio","surgery","resection",
                            "pembrolizumab","immunotherapy","started"]):
        return "treatment"
    return None

def extract(note_text):
    """Pull (date, type, text) events out of the note."""
    events = []
    for sentence in note_text.split("."):
        sentence = sentence.strip()
        if not sentence:
            continue
        date = find_date(sentence)
        kind = find_event_type(sentence)
        if date and kind:
            events.append({"date": date, "type": kind, "text": sentence,
                           "concept": None, "code": None})
    return events


# =====================================================================
# STEP 3:  HARMONIZATION  --  now using a REAL AI model (Sentence-BERT)
# =====================================================================
# Sentence-BERT turns any sentence into a list of 384 numbers (a "vector")
# that captures its meaning. Two sentences with similar meaning get similar
# vectors. We measure similarity with the "cosine" between vectors.
class Harmonizer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        # Load the model (downloads once, then cached). 'all-MiniLM-L6-v2'
        # is a small, fast model that outputs 384-number meaning-vectors.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Pre-compute the vectors for our concept names, once.
        # normalize_embeddings=True scales each vector to length 1, which makes
        # the cosine similarity a simple dot product.
        self.concept_vecs = self.model.encode(
            CONCEPT_NAMES, normalize_embeddings=True, convert_to_numpy=True)

    def match(self, text):
        """Return (concept_name, code, similarity) for the closest concept."""
        vec = self.model.encode([text], normalize_embeddings=True,
                                convert_to_numpy=True)[0]
        sims = self.concept_vecs @ vec        # dot product = cosine similarity
        best = int(np.argmax(sims))           # index of the highest score
        return CONCEPT_NAMES[best], CONCEPTS[best][1], float(sims[best])

def harmonize(events, harmonizer):
    for ev in events:
        name, code, score = harmonizer.match(ev["text"])
        if score > 0.35:                      # ignore weak matches
            ev["concept"], ev["code"] = name, code
    return events


# =====================================================================
# STEP 4:  TIMELINE  --  sort events by date
# =====================================================================
def build_timeline(events):
    return sorted(events, key=lambda ev: ev["date"])


# =====================================================================
# STEP 5a:  FEATURIZE  --  turn each event into numbers the model can read
# =====================================================================
EVENT_TYPES = ["diagnosis", "treatment", "response", "progression"]

def response_score(ev):
    """Hand-coded clinical signal: +1 great, +0.5 good, -1 bad, 0 neutral.

    We set this mapping by hand on purpose, so the feature stays transparent. A
    bigger system could instead learn it from the text or from RECIST tumor
    measurements; here we keep it explicit and simple.
    """
    text = (ev.get("concept") or ev["text"]).lower()
    if ev["type"] == "progression" or "progress" in text:  return -1.0
    if "complete response" in text:                         return  1.0
    if "partial" in text or "reduced" in text:              return  0.5
    return 0.0

def month_index(date):  # "2023-03" -> a single number of months
    y, m = date.split("-"); return int(y) * 12 + int(m)

def featurize(events):
    """Make a table of shape (number_of_events, 6 numbers each)."""
    rows, prev = [], None
    for ev in events:
        onehot = [1.0 if ev["type"] == t else 0.0 for t in EVENT_TYPES]  # 4 numbers
        idx = month_index(ev["date"])
        gap = 0.0 if prev is None else min(idx - prev, 24) / 24.0        # 1 number
        prev = idx
        rows.append(onehot + [gap, response_score(ev)])                 # +1 = 6 total
    return np.array(rows, dtype=np.float32)

FEAT_DIM = len(EVENT_TYPES) + 2   # 6


# =====================================================================
# STEP 5b:  THE PYTORCH TRANSFORMER  --  the real risk model
# =====================================================================
class RiskNet(nn.Module):
    """A small Transformer that reads an event sequence and outputs a risk."""

    def __init__(self, feat_dim=FEAT_DIM, dim=32, heads=2, layers=1, max_len=16):
        super().__init__()
        # nn.Linear(in, out): a layer that turns each 6-number event into a
        # richer 32-number vector the Transformer can work with.
        self.proj = nn.Linear(feat_dim, dim)
        # A learnable "position" signal so the model knows event order.
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        # One Transformer encoder layer = the self-attention machinery.
        #   d_model=dim         : size of each vector flowing through
        #   nhead=heads         : number of parallel "attention heads"
        #   dim_feedforward     : size of the little internal network
        #   batch_first=True    : our data is (batch, time, features)
        #   dropout=0.1         : randomly ignore 10% during training (anti-overfit)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2,
            batch_first=True, dropout=0.1)
        # Stack the layer(s). enable_nested_tensor=False just silences a warning.
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers,
                                             enable_nested_tensor=False)
        # Final layer: turn the summary vector into ONE number (the risk logit).
        self.head = nn.Linear(dim, 1)

    def forward(self, x):                 # x shape: (1, T, 6)
        h = self.proj(x)                  # -> (1, T, 32)
        h = h + self.pos[:, :x.size(1), :]  # add the position signal
        h = self.encoder(h)               # self-attention over the events
        pooled = h.mean(dim=1)            # average the events into one summary
        return self.head(pooled).squeeze(-1)   # -> (1,) a single logit


# =====================================================================
# STEP 5c:  MAKE FAKE TRAINING DATA AND TEACH THE MODEL
# =====================================================================
def latent_risk(feats):
    """The hidden rule the model must learn: progression + treatments raise
    risk, good responses lower it."""
    types = feats[:, :len(EVENT_TYPES)]
    n_prog = types[:, EVENT_TYPES.index("progression")].sum()
    n_treat = types[:, EVENT_TYPES.index("treatment")].sum()
    resp = feats[:, len(EVENT_TYPES) + 1].sum()
    logit = 1.6*n_prog + 0.4*n_treat - 1.1*resp - 0.5
    return 1 / (1 + math.exp(-logit))

def make_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        length = int(rng.integers(3, 8))
        rows, prev = [], None
        for _ in range(length):
            t = EVENT_TYPES[rng.integers(0, len(EVENT_TYPES))]
            onehot = [1.0 if t == e else 0.0 for e in EVENT_TYPES]
            gap = float(rng.random())
            resp = -1.0 if t == "progression" else (
                float(rng.choice([0.5, 1.0])) if t == "response" else 0.0)
            rows.append(onehot + [gap, resp])
        feats = np.array(rows, dtype=np.float32)
        label = 1.0 if rng.random() < latent_risk(feats) else 0.0
        data.append((feats, label))
    return data

def train(model, data, epochs=15):
    # Adam = the optimizer that nudges the model's knobs; lr=learning rate (step size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # BCEWithLogitsLoss = the right "how wrong was I?" measure for yes/no problems
    loss_fn = nn.BCEWithLogitsLoss()
    torch.manual_seed(42); np.random.seed(42)
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(data)
        total = 0.0
        for feats, label in data:
            x = torch.from_numpy(feats).unsqueeze(0)        # (1, T, 6)
            y = torch.tensor([label])                       # (1,)
            optimizer.zero_grad()                           # clear old gradients
            logit = model(x)                                # forward pass -> guess
            loss = loss_fn(logit, y)                        # how wrong was the guess
            loss.backward()                                 # compute the nudges
            optimizer.step()                                # apply the nudges
            total += loss.item()
        print(f"  epoch {epoch+1:02d}/{epochs}  loss={total/len(data):.3f}")
    model.eval()
    return model


@torch.no_grad()
def predict(model, feats):
    """Run a real patient through the trained model -> risk between 0 and 1."""
    x = torch.from_numpy(feats).unsqueeze(0)
    logit = model(x)
    return float(torch.sigmoid(logit).item())   # sigmoid squashes to 0..1


# =====================================================================
# STEP 6:  EXPLAIN  --  remove each event, see how the risk changes
# =====================================================================
def explain(model, events, feats):
    base = predict(model, feats)
    rows = []
    for i in range(len(events)):
        reduced = np.delete(feats, i, axis=0)               # drop event i
        if reduced.shape[0] == 0:
            reduced = np.zeros((1, FEAT_DIM), dtype=np.float32)
        rows.append((events[i], base - predict(model, reduced)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return base, rows


# =====================================================================
# RUN EVERYTHING
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Simple OncoTimeline WITH real models")
    print("=" * 60)

    print("\nThe clinical note:")
    print(NOTE.strip())

    print("\n[1] Training the Transformer on synthetic data ...")
    model = train(RiskNet(), make_dataset())

    print("\n[2] Loading the Sentence-BERT model (downloads once) ...")
    harmonizer = Harmonizer()

    print("\n[3] Reading the note ...")
    events = extract(NOTE)
    events = harmonize(events, harmonizer)
    events = build_timeline(events)

    print("\nTimeline:")
    for ev in events:
        concept = ev["concept"] or "(no match)"
        code = f"  [{ev['code']}]" if ev["code"] else ""
        print(f"  {ev['date']}  {ev['type']:<11} {concept}{code}")

    feats = featurize(events)
    risk = predict(model, feats)
    print(f"\nPredicted trajectory risk: {risk:.1%}")

    base, rows = explain(model, events, feats)
    print("\nWhy? (how much each event changed the risk)")
    for ev, contribution in rows:
        arrow = "raises" if contribution > 0 else "lowers"
        name = ev["concept"] or ev["type"]
        print(f"  {ev['date']}  {name:<28} {arrow} risk by {abs(contribution):.0%}")
