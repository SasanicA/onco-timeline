"""A small clinical concept registry.

This is the part that makes the project recognisably clinical rather than just
"some NLP on text". Each canonical concept carries the kind of standardized codes
a real harmonization layer would emit: an ICD-10 code for diagnoses, a SNOMED CT
concept id, a LOINC code for lab observations, and (where it applies) a RECIST
response category.

To be clear for anyone reading the repo: these code mappings are a small,
hand-written teaching subset, not a validated terminology service. A real system
would query UMLS, a SNOMED CT terminology server, the official LOINC table, or an
OMOP vocabulary. The point here is to show the mechanism (free-text span ->
canonical concept -> standard codes), which is exactly the harmonization step the
project is about.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    """A canonical clinical concept with its codes and surface synonyms."""

    canonical: str
    category: str  # diagnosis | treatment | response | observation | stage
    synonyms: tuple[str, ...]
    icd10: str | None = None
    snomed: str | None = None
    loinc: str | None = None   # only for observation / lab concepts
    recist: str | None = None  # only for response concepts


# --- The registry ------------------------------------------------------
# Kept intentionally compact and readable. Extend it by adding rows.
CONCEPTS: list[Concept] = [
    # --- Diagnoses ---
    Concept(
        canonical="Non-small cell lung cancer",
        category="diagnosis",
        synonyms=("nsclc", "non-small cell lung cancer", "non small cell lung carcinoma",
                  "lung cancer", "lung carcinoma", "carcinoma of the lung"),
        icd10="C34.9",
        snomed="254637007",
    ),
    Concept(
        canonical="Breast cancer",
        category="diagnosis",
        synonyms=("breast cancer", "breast carcinoma", "carcinoma of the breast",
                  "mammary carcinoma"),
        icd10="C50.9",
        snomed="254837009",
    ),
    Concept(
        canonical="Colorectal cancer",
        category="diagnosis",
        synonyms=("colorectal cancer", "colon cancer", "rectal cancer",
                  "colorectal carcinoma", "crc"),
        icd10="C18.9",
        snomed="363406005",
    ),
    Concept(
        canonical="Melanoma",
        category="diagnosis",
        synonyms=("melanoma", "malignant melanoma", "cutaneous melanoma"),
        icd10="C43.9",
        snomed="372244006",
    ),
    Concept(
        canonical="Prostate cancer",
        category="diagnosis",
        synonyms=("prostate cancer", "prostate carcinoma", "carcinoma of the prostate"),
        icd10="C61",
        snomed="399068003",
    ),

    # --- Treatments ---
    Concept(
        canonical="Cisplatin chemotherapy",
        category="treatment",
        synonyms=("cisplatin",),
        snomed="387318005",
    ),
    Concept(
        canonical="Carboplatin chemotherapy",
        category="treatment",
        synonyms=("carboplatin",),
        snomed="395926009",
    ),
    Concept(
        canonical="Docetaxel chemotherapy",
        category="treatment",
        synonyms=("docetaxel",),
    ),
    Concept(
        canonical="Chemotherapy",
        category="treatment",
        synonyms=("chemotherapy", "adjuvant chemotherapy", "second-line chemotherapy",
                  "first-line chemotherapy", "chemo"),
        snomed="367336001",
    ),
    Concept(
        canonical="Pembrolizumab",
        category="treatment",
        synonyms=("pembrolizumab", "keytruda"),
        snomed="703423002",
    ),
    Concept(
        canonical="Immune checkpoint inhibitor therapy",
        category="treatment",
        synonyms=("immunotherapy", "adjuvant immunotherapy", "checkpoint inhibitor",
                  "anti-pd-1"),
    ),
    Concept(
        canonical="Radiotherapy",
        category="treatment",
        synonyms=("radiotherapy", "radiation therapy", "radiation"),
        snomed="108290001",
    ),
    Concept(
        canonical="Surgical resection",
        category="treatment",
        synonyms=("surgical resection", "surgery", "resection", "lobectomy",
                  "tumor resection", "wide local excision", "excision", "lumpectomy"),
        snomed="387713003",
    ),

    # --- Lab observations (LOINC) ---
    # These let the harmonizer demonstrate the third vocabulary the project names.
    Concept(
        canonical="Prostate specific antigen",
        category="observation",
        synonyms=("psa", "prostate specific antigen", "prostate-specific antigen"),
        loinc="2857-1",
    ),
    Concept(
        canonical="Carcinoembryonic antigen",
        category="observation",
        synonyms=("cea", "carcinoembryonic antigen"),
        loinc="2039-6",
    ),
    Concept(
        canonical="Lactate dehydrogenase",
        category="observation",
        synonyms=("ldh", "lactate dehydrogenase"),
        loinc="14804-9",
    ),
    Concept(
        canonical="Cancer antigen 19-9",
        category="observation",
        synonyms=("ca 19-9", "ca19-9", "carbohydrate antigen 19-9"),
        loinc="24108-3",
    ),

    # --- Responses (RECIST 1.1 categories) ---
    Concept(
        canonical="Partial response",
        category="response",
        synonyms=("partial response", "tumor shrinkage", "tumor reduced",
                  "tumor size reduced", "tumor reduced in size", "decrease in tumor size",
                  "responded well"),
        recist="PR",
        snomed="399409002",
    ),
    Concept(
        canonical="Complete response",
        category="response",
        synonyms=("complete response", "no evidence of disease", "ned",
                  "complete remission"),
        recist="CR",
    ),
    Concept(
        canonical="Stable disease",
        category="response",
        synonyms=("stable disease", "no significant change", "unchanged disease"),
        recist="SD",
    ),
    Concept(
        canonical="Progressive disease",
        category="response",
        synonyms=("progressive disease", "disease progression", "progression",
                  "tumor growth", "new metastasis", "new metastases", "worsening"),
        recist="PD",
        snomed="277022003",
    ),
]


def concepts_in(categories: set[str] | None = None) -> list[Concept]:
    """Return the concepts whose category is in the given set (all if None)."""
    if categories is None:
        return CONCEPTS
    return [c for c in CONCEPTS if c.category in categories]


def all_surface_forms() -> list[tuple[str, Concept]]:
    """Flatten the registry into (synonym, concept) pairs for lexical matching."""
    pairs: list[tuple[str, Concept]] = []
    for c in CONCEPTS:
        for s in c.synonyms:
            pairs.append((s.lower(), c))
    return pairs
