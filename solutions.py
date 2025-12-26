# solution.py
"""
Material Request Extraction System
Converts unstructured construction text into structured JSON using LLM + deterministic guardrails.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Optional, Literal, List

from groq import Groq
from pydantic import BaseModel, ValidationError, field_validator

# ============================================================================
# 1. IMPORTS & CONSTANTS
# ============================================================================

MODEL_NAME = "llama-3.1-8b-instant"
MAX_RETRIES = 1
CURRENT_DATE = date(2025, 12, 24)  # or use date.today()

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

groq_client = Groq(api_key=api_key)


# ============================================================================
# 2. DATA CONTRACT (PYDANTIC SCHEMA)
# ============================================================================

class MaterialRequest(BaseModel):
    material_name: Optional[str]
    quantity: Optional[float]
    unit: Optional[str]
    project_name: Optional[str]
    location: Optional[str]
    urgency_signal: Optional[str]  # raw signal from LLM
    urgency: Optional[Literal["low", "medium", "high"]]  # computed
    deadline: Optional[date]

    @field_validator("deadline", mode="before")
    @classmethod
    def validate_deadline(cls, value):
        """Sanitize deadline: parse ISO string or return None if invalid."""
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except (ValueError, TypeError):
                return None
        return None

    class Config:
        extra = "forbid"  # reject hallucinated fields


# ============================================================================
# 3. PROMPT CONSTRUCTION
# ============================================================================

def build_system_prompt() -> str:
    """
    Returns the system prompt that defines task, schema, and rules.
    No logic here, only instructions.
    """
    return """
You are an information extraction system for construction material requests.

Your task is to extract structured data from unstructured construction-related text and return it as JSON.

You must follow ALL rules strictly.

OUTPUT FORMAT:
- Return ONLY a single JSON object.
- Do NOT include explanations, comments, or extra text.
- Do NOT include any keys other than those defined below.

JSON SCHEMA:
{
  "material_name": string,
  "quantity": number | null,
  "unit": string | null,
  "project_name": string | null,
  "location": string | null,
  "urgency_signal": string | null,
  "deadline": string | null
}

FIELD RULES:
- If a field is NOT explicitly mentioned or clearly inferable from the text, set it to null.
- Do NOT guess or invent values.
- Do NOT assume project names, locations, quantities, or dates.
- If quantity is ambiguous or not numeric, set it to null.
- Deadline must be an ISO date string (YYYY-MM-DD) ONLY if clearly stated or directly computable.
- If the deadline is vague (e.g. "soon", "ASAP"), set deadline to null.

URGENCY:
- Do NOT output final urgency levels.
- Only extract raw urgency words if present (e.g. "urgent", "ASAP", "immediately").
- Put raw urgency words into "urgency_signal".
- Final urgency will be computed outside the model.

EXAMPLE:

Input:
"Create 25mm steel bars, 120 units for Project Phoenix, required before 15th March"

Output:
{
  "material_name": "25mm steel bars",
  "quantity": 120,
  "unit": "units",
  "project_name": "Project Phoenix",
  "location": null,
  "urgency_signal": null,
  "deadline": "2025-03-15"
}
"""


# ============================================================================
# 4. LLM CALL WRAPPER (UNTRUSTED ZONE)
# ============================================================================

def call_llm(user_text: str, system_prompt: str) -> Optional[str]:
    """
    Calls Groq LLM with system + user messages.
    Returns raw response text or None on failure.
    """
    try:
        completion = groq_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


# ============================================================================
# 5. PARSING + VALIDATION (CRITICAL GATE)
# ============================================================================

def parse_and_validate(raw_output: str) -> Optional[MaterialRequest]:
    """
    Parse JSON and validate against MaterialRequest schema.
    Returns MaterialRequest object or None if invalid.
    """
    try:
        data = json.loads(raw_output)
        # urgency will be None here; we compute it later
        return MaterialRequest(
            material_name=data.get("material_name") or None,
            quantity=data.get("quantity"),
            unit=data.get("unit"),
            project_name=data.get("project_name"),
            location=data.get("location"),
            urgency_signal=data.get("urgency_signal"),
            urgency=None,  # computed later
            deadline=data.get("deadline"),
        )
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Parse/validation failed: {e}")
        return None


def auto_correct(user_text: str, system_prompt: str) -> Optional[MaterialRequest]:
    """
    Single bounded retry: re-prompt LLM to fix invalid JSON.
    Returns corrected MaterialRequest or None.
    """
    fix_prompt = (
        system_prompt
        + "\n\nYour previous response was invalid JSON or failed schema validation.\n"
        + "Return ONLY corrected JSON matching the schema. Do not change field meanings."
    )

    corrected_output = call_llm(user_text, fix_prompt)
    if corrected_output is None:
        return None

    return parse_and_validate(corrected_output)


def safe_fallback() -> MaterialRequest:
    """
    Returns schema-compliant object with all fields null/empty.
    Ensures the system never crashes.
    """
    return MaterialRequest(
        material_name=None,
        quantity=None,
        unit=None,
        project_name=None,
        location=None,
        urgency_signal=None,
        urgency="low",
        deadline=None,
    )


# ============================================================================
# 6. DETERMINISTIC BUSINESS LOGIC (NO LLM)
# ============================================================================

def compute_urgency(
    urgency_signal: Optional[str],
    deadline: Optional[date],
    current_date: date = CURRENT_DATE,
) -> str:
    """
    Compute final urgency deterministically using:
    - urgency_signal keywords
    - deadline delta from current_date

    Returns: "low", "medium", or "high"
    """
    # Check for strong urgency keywords
    if urgency_signal:
        signal_lower = urgency_signal.lower()
        if any(word in signal_lower for word in ["urgent", "asap", "immediately", "critical"]):
            return "high"

    # Use deadline delta if available
    if deadline and deadline >= current_date:
        delta_days = (deadline - current_date).days
        if delta_days <= 7:
            return "high"
        elif delta_days <= 15:
            return "medium"
        else:
            return "low"

    # Default
    return "low"


# ============================================================================
# 7. ORCHESTRATION (SINGLE PUBLIC ENTRY POINT)
# ============================================================================

def process_material_request(text: str) -> dict:
    """
    Main pipeline:
    1. Call LLM
    2. Parse & validate
    3. Retry once if needed
    4. Compute urgency
    5. Return final dict

    Always returns a schema-compliant dict.
    """
    system_prompt = build_system_prompt()

    # First LLM call
    raw_output = call_llm(text, system_prompt)
    if raw_output is None:
        obj = safe_fallback()
    else:
        obj = parse_and_validate(raw_output)

        # Auto-correction if first attempt failed
        if obj is None:
            obj = auto_correct(text, system_prompt)

        # Final fallback if retry also failed
        if obj is None:
            obj = safe_fallback()

    # Compute urgency (deterministic)
    obj.urgency = compute_urgency(obj.urgency_signal, obj.deadline, CURRENT_DATE)

    # Return as dict (JSON-serializable)
    return obj.model_dump(mode="json")


# ============================================================================
# 8. FILE I/O HELPERS
# ============================================================================

def read_inputs(filepath: str) -> List[str]:
    """Read input lines from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_outputs(filepath: str, results: List[dict]) -> None:
    """Write results to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


# ============================================================================
# 9. MAIN (THIN, BORING, CLEAN)
# ============================================================================

def main():
    """
    Entry point:
    - Read testinputs.txt
    - Process each input
    - Write outputs.json
    """
    inputs = read_inputs("testinputs.txt")
    results = []

    for i, text in enumerate(inputs, 1):
        print(f"Processing input {i}/{len(inputs)}...")
        output = process_material_request(text)
        results.append({
            "input": text,
            "output": output,
        })

    write_outputs("outputs.json", results)
    print(f"✓ Processed {len(inputs)} inputs → outputs.json")


if __name__ == "__main__":
    main()
