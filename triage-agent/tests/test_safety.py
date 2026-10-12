"""
Clinical Safety Layer Diagnostic Test.

Checks:
  1. Red-flag symptom keyword detection (chest pain, stroke, breathing difficulty).
  2. Deterministic handoff & emergency escalation (bypassing the LLM).
  3. Anti-fabrication placeholder filtering ([Insert Department], etc.).
  4. Patient self-correction handling.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.safety.red_flag_rules import check_red_flags
from src.safety.handoff import evaluate_handoff, contains_fabricated_placeholder, FABRICATION_FALLBACK_MESSAGE
from src.slm.correction_handling import detect_correction, annotate_transcript


def run_safety_test():
    print("=" * 60)
    print(" 🛡️  CLINICAL SAFETY LAYER DIAGNOSTIC TEST")
    print("=" * 60)

    # 1. Red-Flag Detection Test
    print("\n[1/4] Testing Red-Flag Symptom Detection...")
    red_flag_cases = [
        ("I have severe crushing chest pain radiating to my shoulder.", True, "cardiac"),
        ("I am having sudden difficulty breathing and shortness of breath.", True, "respiratory"),
        ("I noticed sudden facial droop and slurred speech.", True, "neurological"),
        ("I have had a mild runny nose and cough.", False, None),
        ("I have a cough but no chest pain and no shortness of breath.", False, None),
        ("a bit shortness of breath and cough and mucus nothing else", False, None),
        ("i also have a slight chest pain", False, None),
        ("a bit of chest pain when coughing", False, None),
        ("I have crushing chest pain and left arm numbness.", True, "cardiac"),
        ("I have chest pain", True, "cardiac"),
    ]

    all_passed = True
    for text, should_flag, expected_cat in red_flag_cases:
        res = check_red_flags(text)
        flagged = res.triggered
        status = "✅ PASS" if flagged == should_flag else "❌ FAIL"
        print(f"  {status} | Flagged: {flagged!s:<5} (Expected: {should_flag!s:<5}) | Category: {res.category} | Input: {text!r}")
        if flagged != should_flag:
            all_passed = False

    # 2. Handoff Decision Test
    print("\n[2/4] Testing Emergency Escalation & Handoff...")
    handoff_cases = [
        ("I have severe chest pain and cannot breathe.", 1, True),
        ("Just a slight itchy skin rash on my forearm.", 2, False),
        ("Still waiting and this is taking too long.", 10, True), # Turn limit escalation (MAX_TURNS_BEFORE_HANDOFF=6)
    ]

    for text, turn_count, expected_escalate in handoff_cases:
        decision = evaluate_handoff(text, turn_count)
        status = "✅ PASS" if decision.escalate == expected_escalate else "❌ FAIL"
        print(f"  {status} | Escalate: {decision.escalate!s:<5} (Reason: {decision.reason}) | Turn {turn_count}")
        if decision.escalate != expected_escalate:
            all_passed = False

    # 3. Anti-Fabrication Detection Test
    print("\n[3/4] Testing Anti-Fabrication Placeholder Filter...")
    fab_cases = [
        ("Please go to [Hospital Name] on the [Room Number] floor.", True),
        ("Your wait time will be [Wait Time Minutes] minutes.", True),
        ("Please head to the General Medicine clinic for a routine check.", False),
    ]

    for text, expected_fab in fab_cases:
        is_fab = contains_fabricated_placeholder(text)
        status = "✅ PASS" if is_fab == expected_fab else "❌ FAIL"
        print(f"  {status} | Fabrication: {is_fab!s:<5} | Text: {text!r}")
        if is_fab != expected_fab:
            all_passed = False

    # 4. Patient Correction Handling Test
    print("\n[4/4] Testing Patient Self-Correction Parsing...")
    correction_cases = [
        ("It started yesterday, actually I meant 3 days ago.", True),
        ("No, I said left leg, not right.", True),
        ("I have had a cough for two weeks.", False),
    ]

    for text, expected_corr in correction_cases:
        res = detect_correction(text)
        detected = res.is_correction_attempt
        annotated = annotate_transcript(text)
        status = "✅ PASS" if detected == expected_corr else "❌ FAIL"
        print(f"  {status} | Correction Detected: {detected!s:<5} | Input: {text!r}")
        if detected != expected_corr:
            all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("✅ SUCCESS: All safety rules and guardrails passed validation!")
    else:
        print("❌ FAILED: One or more safety checks failed.")
    return all_passed


if __name__ == "__main__":
    success = run_safety_test()
    sys.exit(0 if success else 1)
