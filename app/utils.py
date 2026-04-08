"""
app/utils.py
============
Static lookup tables and pure helper functions.
No external dependencies; no side effects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# CPT-style billing code table
# Format: code -> (description, report_category, is_valid)
# ---------------------------------------------------------------------------

CPT_CODE_TABLE: Dict[str, tuple] = {
    # Imaging
    "71046": ("Chest X-Ray PA and Lateral", "imaging", True),
    "71045": ("Chest X-Ray Single View", "imaging", True),
    "70553": ("MRI Brain with Contrast", "imaging", True),
    "74177": ("CT Abdomen and Pelvis with Contrast", "imaging", True),
    # Lab
    "80050": ("General Health Panel", "lab", True),
    "80053": ("Comprehensive Metabolic Panel", "lab", True),
    "85025": ("Complete Blood Count with Differential", "lab", True),
    "84443": ("Thyroid Stimulating Hormone", "lab", True),
    # Office / Billing
    "99213": ("Office Visit – Established Patient, Low Complexity", "billing", True),
    "99214": ("Office Visit – Established Patient, Moderate Complexity", "billing", True),
    "99215": ("Office Visit – Established Patient, High Complexity", "billing", True),
    # Prescription
    "90837": ("Psychotherapy 60 minutes", "prescription", True),
    "J0696":  ("Ceftriaxone Sodium Injection 500mg", "prescription", True),
    # Discharge
    "99238": ("Hospital Discharge Day Management ≤30 min", "discharge", True),
    "99239": ("Hospital Discharge Day Management >30 min", "discharge", True),
    # Invalid codes (used in hard scenarios)
    "FAKE99": ("Unknown / Unrecognised Code", "unknown", False),
    "00000":  ("Null Code – Not in Schedule", "unknown", False),
}


# ---------------------------------------------------------------------------
# Insurance plan coverage table
# Format: plan_id -> list of covered CPT codes
# ---------------------------------------------------------------------------

INSURANCE_COVERAGE_TABLE: Dict[str, List[str]] = {
    "PLAN-PREMIUM": [
        "71046", "71045", "70553", "74177",
        "80050", "80053", "85025", "84443",
        "99213", "99214", "99215",
        "90837", "J0696",
        "99238", "99239",
    ],
    "PLAN-BLUE": [
        "71046", "71045",
        "80050", "80053", "85025",
        "99213", "99214",
        "99238",
    ],
    "PLAN-BASIC": [
        "80050", "85025",
        "99213",
    ],
    "PLAN-NONE": [],
}


# ---------------------------------------------------------------------------
# Report classification – keyword → label mapping
# Used by the heuristic fallback agent (not the grader)
# ---------------------------------------------------------------------------

KEYWORD_LABEL_MAP: Dict[str, List[str]] = {
    "lab": [
        "blood count", "wbc", "rbc", "hemoglobin", "hematocrit",
        "metabolic panel", "glucose", "creatinine", "electrolytes",
        "urinalysis", "culture", "sensitivity", "tsh", "lipid panel",
        "complete blood", "cbc", "bmp", "cmp",
    ],
    "imaging": [
        "x-ray", "xray", "mri", "ct scan", "computed tomography",
        "ultrasound", "echo", "echocardiogram", "radiograph",
        "fluoroscopy", "mammogram", "pet scan", "nuclear",
        "chest film", "bone scan",
    ],
    "prescription": [
        "prescribed", "prescription", "medication", "drug", "dose",
        "tablet", "capsule", "injection", "mg", "mcg", "units/day",
        "pharmacy", "refill", "sig:", "dispense",
    ],
    "billing": [
        "office visit", "consultation", "follow-up", "appointment",
        "established patient", "new patient", "copay", "deductible",
        "insurance", "claim", "procedure", "cpt",
    ],
    "discharge": [
        "discharge", "discharged", "discharge summary",
        "follow up in", "follow-up care", "outpatient instructions",
        "released", "home with",
    ],
}

# All valid report labels
VALID_REPORT_LABELS: List[str] = list(KEYWORD_LABEL_MAP.keys())

# All valid blood types
VALID_BLOOD_TYPES: List[str] = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

# Universal donor for red blood cells
UNIVERSAL_DONOR = "O-"

# Blood compatibility: recipient → acceptable donor types (RBC)
BLOOD_COMPATIBILITY: Dict[str, List[str]] = {
    "A+":  ["A+", "A-", "O+", "O-"],
    "A-":  ["A-", "O-"],
    "B+":  ["B+", "B-", "O+", "O-"],
    "B-":  ["B-", "O-"],
    "AB+": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    "AB-": ["A-", "B-", "AB-", "O-"],
    "O+":  ["O+", "O-"],
    "O-":  ["O-"],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_valid_cpt_code(code: str) -> bool:
    """Return True if the code exists in the table AND is marked valid."""
    entry = CPT_CODE_TABLE.get(code)
    return entry is not None and entry[2] is True


def get_code_category(code: str) -> Optional[str]:
    """Return the report category for a CPT code, or None if unknown."""
    entry = CPT_CODE_TABLE.get(code)
    return entry[1] if entry else None


def is_covered_by_plan(plan_id: str, code: str) -> bool:
    """Return True if the given plan covers the CPT code."""
    covered = INSURANCE_COVERAGE_TABLE.get(plan_id, [])
    return code in covered


def is_compatible_blood(requested_type: str, offered_type: str) -> bool:
    """Return True if offered_type can safely be given to a requested_type patient."""
    compatible = BLOOD_COMPATIBILITY.get(requested_type, [])
    return offered_type in compatible


def heuristic_classify(report_text: str) -> str:
    """
    Simple keyword-based classification for the heuristic agent fallback.
    Returns the label with the most keyword hits, defaulting to 'billing'.
    """
    text_lower = report_text.lower()
    scores = {label: 0 for label in VALID_REPORT_LABELS}
    for label, keywords in KEYWORD_LABEL_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "billing"


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))
