"""
Hospital Directory & Synthetic Vignette Dataset Generator for OmniSpec-Triage.

Generates:
1. eval/data/hospital_directory.json:
   Comprehensive directory of hospital departments, room numbers, floor locations,
   and clinical symptom triggers (Cardiology Room 120, Fever Clinic Room 20, etc.).

2. eval/data/synthetic_hospital_eval.json:
   Synthetic intake vignettes testing selective room routing, nurse clarification,
   and non-clinical rejection without context bloat.
"""

import json
import os
from typing import Dict, Any, List

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_hospital_directory() -> Dict[str, Any]:
    directory = {
        "metadata": {
            "facility_name": "St. Jude Metropolitan Hospital",
            "campus": "Main Clinical Center",
            "version": "1.0",
            "total_rooms": 14,
        },
        "rooms": [
            {
                "room_id": "cardiac_120",
                "room_number": "120",
                "department": "Cardiology & Chest Pain Bay",
                "floor": "1st Floor, East Wing",
                "acuity_tier": "Urgent",
                "clinical_keywords": [
                    "chest pain", "chest tightness", "heart pounding", "palpitations",
                    "angina", "irregular heartbeat", "cardiac", "heart fluttering",
                    "chest pressure", "radiating chest discomfort"
                ],
                "routing_guidance": "Direct patients with chest discomfort or cardiac symptoms to Room 120 once initial intake is complete."
            },
            {
                "room_id": "fever_20",
                "room_number": "20",
                "department": "Fever & Acute Infection Clinic",
                "floor": "Ground Floor, West Wing",
                "acuity_tier": "Routine",
                "clinical_keywords": [
                    "fever", "high temperature", "chills", "shivering", "burning up",
                    "sweats", "viral infection", "flu-like symptoms", "feverish"
                ],
                "routing_guidance": "Direct febrile patients to Room 20 for temperature check and infectious disease intake."
            },
            {
                "room_id": "emergency_10",
                "room_number": "10",
                "department": "Emergency Resuscitation & Trauma",
                "floor": "Ground Floor, Red Bay",
                "acuity_tier": "Red Flag Emergency",
                "clinical_keywords": [
                    "severe bleeding", "unconscious", "passed out", "stroke", "facial droop",
                    "cannot breathe", "anaphylaxis", "severe trauma", "massive hemorrhage",
                    "unresponsive", "sudden collapse"
                ],
                "routing_guidance": "Immediate emergency room for critical life-threatening conditions."
            },
            {
                "room_id": "ortho_105",
                "room_number": "105",
                "department": "Orthopedics & Minor Injuries",
                "floor": "1st Floor, South Wing",
                "acuity_tier": "Routine",
                "clinical_keywords": [
                    "broken bone", "fracture", "sprained ankle", "twisted wrist", "knee injury",
                    "swollen ankle", "joint pain", "cannot walk", "bone pain", "dislocation"
                ],
                "routing_guidance": "Direct musculoskeletal trauma, sprains, and fractures to Room 105 for splinting/x-ray."
            },
            {
                "room_id": "pulmonary_45",
                "room_number": "45",
                "department": "Pulmonology & Respiratory Care",
                "floor": "Ground Floor, North Wing",
                "acuity_tier": "Urgent",
                "clinical_keywords": [
                    "wheezing", "asthma attack", "inhaler", "persistent cough", "bronchitis",
                    "shortness of breath", "stridor", "chest congestion", "heavy breathing"
                ],
                "routing_guidance": "Direct stable asthmatic or respiratory patients needing nebulization to Room 45."
            },
            {
                "room_id": "neuro_210",
                "room_number": "210",
                "department": "Neurology Clinic",
                "floor": "2nd Floor, Central Tower",
                "acuity_tier": "Standard / Specialist",
                "clinical_keywords": [
                    "migraine", "severe headache", "dizziness", "vertigo", "seizure",
                    "numbness in fingers", "pins and needles", "light sensitivity"
                ],
                "routing_guidance": "Direct non-acute neurological presentations, migraines, and vertigo to Room 210."
            },
            {
                "room_id": "gastro_80",
                "room_number": "80",
                "department": "Gastroenterology & Abdominal Clinic",
                "floor": "Ground Floor, East Corridor",
                "acuity_tier": "Standard",
                "clinical_keywords": [
                    "stomach pain", "abdominal cramps", "nausea", "vomiting", "food poisoning",
                    "diarrhea", "belly ache", "acid reflux", "stomach ulcer"
                ],
                "routing_guidance": "Direct abdominal discomfort and acute gastroenteritis to Room 80."
            },
            {
                "room_id": "ent_15",
                "room_number": "15",
                "department": "Ear, Nose & Throat (ENT)",
                "floor": "Ground Floor, West Corridor",
                "acuity_tier": "Routine",
                "clinical_keywords": [
                    "ear pain", "ear infection", "sore throat", "tonsils", "sinus pressure",
                    "blocked ear", "runny nose", "difficulty swallowing", "hoarse voice"
                ],
                "routing_guidance": "Direct upper respiratory, ear, and throat complaints to Room 15."
            },
            {
                "room_id": "eye_130",
                "room_number": "130",
                "department": "Ophthalmology & Eye Bay",
                "floor": "1st Floor, West Wing",
                "acuity_tier": "Urgent / Routine",
                "clinical_keywords": [
                    "eye pain", "foreign object in eye", "chemical in eye", "blurry vision",
                    "swollen eye", "conjunctivitis", "pink eye", "eye redness"
                ],
                "routing_guidance": "Direct acute eye issues and vision disturbances to Room 130."
            },
            {
                "room_id": "pediatric_25",
                "room_number": "25",
                "department": "Pediatric Triage & Care",
                "floor": "Ground Floor, Yellow Wing",
                "acuity_tier": "Routine / Urgent",
                "clinical_keywords": [
                    "child", "baby", "toddler", "infant", "pediatric", "my 5 year old",
                    "my son has fever", "my daughter", "newborn"
                ],
                "routing_guidance": "Direct infant and child presentations to Room 25 (Pediatrics)."
            },
            {
                "room_id": "derma_112",
                "room_number": "112",
                "department": "Dermatology & Skin Clinic",
                "floor": "1st Floor, North Corridor",
                "acuity_tier": "Routine",
                "clinical_keywords": [
                    "skin rash", "hives", "eczema", "itchy skin", "blisters", "skin allergic reaction",
                    "boil", "skin swelling"
                ],
                "routing_guidance": "Direct dermatological rashes and localized allergic reactions to Room 112."
            },
            {
                "room_id": "general_05",
                "room_number": "5",
                "department": "General Medicine & Health Intake",
                "floor": "Ground Floor, Main Atrium",
                "acuity_tier": "Routine",
                "clinical_keywords": [
                    "general checkup", "feeling weak", "mild malaise", "fatigue",
                    "prescription refill", "blood pressure check", "routine consultation"
                ],
                "routing_guidance": "Direct non-specific, mild or general constitutional issues to Room 5."
            }
        ]
    }
    return directory


def generate_synthetic_eval_cases() -> List[Dict[str, Any]]:
    cases = [
        # --- CARDIOLOGY (Room 120) ---
        {
            "id": "syn_cardiac_01",
            "transcript": "I have had this persistent tightness and pressure in my chest since this morning.",
            "expected_room": "120",
            "expected_dept": "Cardiology & Chest Pain Bay",
            "symptom_category": "cardiac",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },
        {
            "id": "syn_cardiac_02",
            "transcript": "My heart feels like it is pounding out of my chest and fluttering when I sit down.",
            "expected_room": "120",
            "expected_dept": "Cardiology & Chest Pain Bay",
            "symptom_category": "cardiac",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },
        {
            "id": "syn_cardiac_03",
            "transcript": "I feel angina pain across my chest wall that radiates up my neck.",
            "expected_room": "120",
            "expected_dept": "Cardiology & Chest Pain Bay",
            "symptom_category": "cardiac",
            "expected_retrieval": True,
            "is_emergency": True,
            "nurse_clarification_needed": False
        },

        # --- FEVER & INFECTION (Room 20) ---
        {
            "id": "syn_fever_01",
            "transcript": "I have a high fever of 102 degrees and shivering chills that started last night.",
            "expected_room": "20",
            "expected_dept": "Fever & Acute Infection Clinic",
            "symptom_category": "fever",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },
        {
            "id": "syn_fever_02",
            "transcript": "I am burning up with sweats, chills, and viral flu-like body aches.",
            "expected_room": "20",
            "expected_dept": "Fever & Acute Infection Clinic",
            "symptom_category": "fever",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },
        {
            "id": "syn_fever_03",
            "transcript": "My temperature has been high for three days and I feel feverish all over.",
            "expected_room": "20",
            "expected_dept": "Fever & Acute Infection Clinic",
            "symptom_category": "fever",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },

        # --- ORTHOPEDICS (Room 105) ---
        {
            "id": "syn_ortho_01",
            "transcript": "I twisted my ankle playing basketball and now it is swollen and I cannot put weight on it.",
            "expected_room": "105",
            "expected_dept": "Orthopedics & Minor Injuries",
            "symptom_category": "orthopedic",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },
        {
            "id": "syn_ortho_02",
            "transcript": "I tripped on the stairs and heard a snap in my wrist, I think it is a broken bone.",
            "expected_room": "105",
            "expected_dept": "Orthopedics & Minor Injuries",
            "symptom_category": "orthopedic",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },

        # --- RESPIRATORY & PULMONOLOGY (Room 45) ---
        {
            "id": "syn_pulm_01",
            "transcript": "I have an asthma flare up with wheezing and my inhaler is not providing relief.",
            "expected_room": "45",
            "expected_dept": "Pulmonology & Respiratory Care",
            "symptom_category": "respiratory",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },

        # --- ENT (Room 15) ---
        {
            "id": "syn_ent_01",
            "transcript": "I have severe ear pain and pressure in my left ear with difficulty swallowing.",
            "expected_room": "15",
            "expected_dept": "Ear, Nose & Throat (ENT)",
            "symptom_category": "ent",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },

        # --- GASTROENTEROLOGY (Room 80) ---
        {
            "id": "syn_gastro_01",
            "transcript": "Severe abdominal cramps and continuous nausea after eating seafood.",
            "expected_room": "80",
            "expected_dept": "Gastroenterology & Abdominal Clinic",
            "symptom_category": "gastro",
            "expected_retrieval": True,
            "is_emergency": False,
            "nurse_clarification_needed": True
        },

        # --- EMERGENCY RESUSCITATION (Room 10) ---
        {
            "id": "syn_emergency_01",
            "transcript": "He is losing massive blood from a deep wound and is starting to pass out.",
            "expected_room": "10",
            "expected_dept": "Emergency Resuscitation & Trauma",
            "symptom_category": "emergency",
            "expected_retrieval": True,
            "is_emergency": True,
            "nurse_clarification_needed": False
        },

        # --- NON-CLINICAL / IRRELEVANT (Expected Retrieval: None / 0 tokens) ---
        {
            "id": "syn_irrelevant_01",
            "transcript": "Hello good morning, could you tell me where the parking validation desk is?",
            "expected_room": None,
            "expected_dept": None,
            "symptom_category": "administrative",
            "expected_retrieval": False,
            "is_emergency": False,
            "nurse_clarification_needed": False
        },
        {
            "id": "syn_irrelevant_02",
            "transcript": "Thank you nurse, I appreciate the help.",
            "expected_room": None,
            "expected_dept": None,
            "symptom_category": "chit_chat",
            "expected_retrieval": False,
            "is_emergency": False,
            "nurse_clarification_needed": False
        },
        {
            "id": "syn_irrelevant_03",
            "transcript": "Is there a cafeteria or coffee shop nearby?",
            "expected_room": None,
            "expected_dept": None,
            "symptom_category": "facility",
            "expected_retrieval": False,
            "is_emergency": False,
            "nurse_clarification_needed": False
        }
    ]
    return cases


def build_and_save_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    directory_file = os.path.join(DATA_DIR, "hospital_directory.json")
    vignettes_file = os.path.join(DATA_DIR, "synthetic_hospital_eval.json")

    directory_data = generate_hospital_directory()
    with open(directory_file, "w", encoding="utf-8") as f:
        json.dump(directory_data, f, indent=2)
    print(f"✅ Generated {directory_file} ({len(directory_data['rooms'])} hospital rooms)")

    eval_data = generate_synthetic_eval_cases()
    with open(vignettes_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"✅ Generated {vignettes_file} ({len(eval_data)} synthetic evaluation test cases)")


if __name__ == "__main__":
    build_and_save_all()
