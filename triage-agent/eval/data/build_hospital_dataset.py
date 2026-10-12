"""
Hospital Directory & Synthetic Vignette Dataset Generator for OmniSpec-Triage.

Generates:
1. eval/data/hospital_directory.json:
   Comprehensive directory of hospital wards, room numbers, sub-room bays, floor locations,
   and clinical symptom triggers (Ward 3 Room 120-A for Cardiac, Ward 1 Room 20-A for Fever, etc.).

2. eval/data/synthetic_hospital_eval.json:
   Synthetic intake vignettes testing selective multi-option room routing,
   reasoning across candidate choices, and non-clinical rejection without context bloat.
"""

import json
import os
from typing import Dict, Any, List

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_hospital_directory() -> Dict[str, Any]:
    directory = {
        "metadata": {
            "facility_name": "St. Jude Metropolitan Hospital & Medical Center",
            "campus": "Main Clinical Tower & Ambulatory Complex",
            "version": "2.0",
            "total_rooms": 32,
        },
        "rooms": [
            # ==========================================
            # WARD 0: EMERGENCY RESUSCITATION & TRAUMA COMPLEX
            # ==========================================
            {
                "room_id": "emergency_10_a",
                "ward": "Ward 0 (Emergency Complex)",
                "room_number": "10-A",
                "sub_room_name": "Resuscitation Bay 1 (Critical Cardiac & Stroke)",
                "department": "Emergency Medicine",
                "floor": "Ground Floor, Red Bay",
                "acuity_tier": "Immediate Red Flag (KTAS 1)",
                "clinical_keywords": [
                    "severe bleeding", "unconscious", "passed out", "stroke", "facial droop",
                    "cannot breathe", "anaphylaxis", "severe trauma", "massive hemorrhage",
                    "unresponsive", "sudden collapse", "no pulse"
                ],
                "routing_guidance": "Immediate code-red resuscitation for catastrophic collapse, unresponsive patients, or stroke."
            },
            {
                "room_id": "emergency_10_b",
                "ward": "Ward 0 (Emergency Complex)",
                "room_number": "10-B",
                "sub_room_name": "Acute Polytrauma Bay",
                "department": "Trauma & Resuscitation",
                "floor": "Ground Floor, Red Bay",
                "acuity_tier": "Critical (KTAS 1-2)",
                "clinical_keywords": [
                    "car accident", "severe injury", "crush injury", "open fracture",
                    "stab wound", "gunshot", "deep laceration bleeding profusely", "head trauma"
                ],
                "routing_guidance": "High-impact accidents, severe physical trauma, and heavy uncontrolled bleeding."
            },
            {
                "room_id": "emergency_10_c",
                "ward": "Ward 0 (Emergency Complex)",
                "room_number": "10-C",
                "sub_room_name": "Acute Respiratory Arrest Bay",
                "department": "Emergency Pulmonary",
                "floor": "Ground Floor, Red Bay",
                "acuity_tier": "Critical (KTAS 1)",
                "clinical_keywords": [
                    "cannot breathe at all", "turning blue", "gasping for air",
                    "choking", "stridor severe", "acute respiratory failure"
                ],
                "routing_guidance": "Severe acute respiratory arrest, choking, or extreme inability to breathe."
            },

            # ==========================================
            # WARD 1: OUTPATIENT INFECTIOUS & GENERAL MEDICINE
            # ==========================================
            {
                "room_id": "fever_20_a",
                "ward": "Ward 1 (Infectious & General Clinic)",
                "room_number": "20-A",
                "sub_room_name": "Febrile & Isolation Bay",
                "department": "Fever & Acute Infection Clinic",
                "floor": "Ground Floor, West Wing",
                "acuity_tier": "Urgent / Routine (KTAS 3-4)",
                "clinical_keywords": [
                    "high fever", "temperature 102", "temperature 103", "severe chills",
                    "shivering uncontrollably", "burning up", "night sweats", "malaria symptoms", "dengue"
                ],
                "routing_guidance": "Patients presenting with high fever, rigors, or infectious illness requiring isolation."
            },
            {
                "room_id": "fever_20_b",
                "ward": "Ward 1 (Infectious & General Clinic)",
                "room_number": "20-B",
                "sub_room_name": "General Viral & Flu Bay",
                "department": "Fever & Acute Infection Clinic",
                "floor": "Ground Floor, West Wing",
                "acuity_tier": "Routine (KTAS 4-5)",
                "clinical_keywords": [
                    "mild fever", "flu-like symptoms", "body aches with fever", "viral illness",
                    "sweats", "feverish feeling", "low grade temperature"
                ],
                "routing_guidance": "Mild flu-like illness, low-grade temperature, and general seasonal viral presentations."
            },
            {
                "room_id": "general_05_a",
                "ward": "Ward 1 (Infectious & General Clinic)",
                "room_number": "5-A",
                "sub_room_name": "Outpatient Triage & Consultation",
                "department": "General Medicine",
                "floor": "Ground Floor, Main Atrium",
                "acuity_tier": "Routine (KTAS 5)",
                "clinical_keywords": [
                    "general checkup", "feeling weak", "mild malaise", "fatigue",
                    "routine consultation", "feeling unwell without specific pain"
                ],
                "routing_guidance": "Non-specific mild symptoms, constitutional malaise, and initial physician evaluation."
            },
            {
                "room_id": "general_05_b",
                "ward": "Ward 1 (Infectious & General Clinic)",
                "room_number": "5-B",
                "sub_room_name": "Chronic Care & Prescription Renewal",
                "department": "General Medicine",
                "floor": "Ground Floor, Main Atrium",
                "acuity_tier": "Non-Urgent (KTAS 5)",
                "clinical_keywords": [
                    "prescription refill", "blood pressure check routine", "medication renewal",
                    "follow up test results", "lab paperwork"
                ],
                "routing_guidance": "Administrative refills, routine vitals monitoring, and non-symptomatic check-ins."
            },

            # ==========================================
            # WARD 2: PULMONOLOGY & RESPIRATORY CARE
            # ==========================================
            {
                "room_id": "pulmonary_45_a",
                "ward": "Ward 2 (Respiratory & Chest Care)",
                "room_number": "45-A",
                "sub_room_name": "Nebulization & Acute Asthma Bay",
                "department": "Pulmonology & Respiratory Care",
                "floor": "Ground Floor, North Wing",
                "acuity_tier": "Urgent (KTAS 3)",
                "clinical_keywords": [
                    "asthma flare up", "wheezing", "inhaler not working", "bronchospasm",
                    "asthmatic coughing", "chest tightness breathing", "nebulizer"
                ],
                "routing_guidance": "Active asthma symptoms, wheezing, or patients requiring urgent bronchodilator nebulization."
            },
            {
                "room_id": "pulmonary_45_b",
                "ward": "Ward 2 (Respiratory & Chest Care)",
                "room_number": "45-B",
                "sub_room_name": "Chronic Cough & Bronchial Care",
                "department": "Pulmonology & Respiratory Care",
                "floor": "Ground Floor, North Wing",
                "acuity_tier": "Standard (KTAS 4)",
                "clinical_keywords": [
                    "persistent cough", "chronic bronchitis", "bronchitis", "chest congestion",
                    "mild shortness of breath", "a bit shortness of breath", "heavy breathing with mucus"
                ],
                "routing_guidance": "Persistent sub-acute coughs, stable bronchitis, chest congestion, and mild breathing irritation."
            },

            # ==========================================
            # WARD 3: CARDIOLOGY & HEART CARE CENTER
            # ==========================================
            {
                "room_id": "cardiac_120_a",
                "ward": "Ward 3 (Cardiology Center)",
                "room_number": "120-A",
                "sub_room_name": "Chest Pain & Rapid Telemetry Bay",
                "department": "Cardiology",
                "floor": "1st Floor, East Wing",
                "acuity_tier": "Urgent (KTAS 2-3)",
                "clinical_keywords": [
                    "chest pain", "chest tightness", "angina", "chest pressure",
                    "radiating chest discomfort", "heavy chest sensation", "left side chest ache"
                ],
                "routing_guidance": "Active chest pressure, suspected angina, or stable cardiac pain requiring immediate ECG & troponin."
            },
            {
                "room_id": "cardiac_120_b",
                "ward": "Ward 3 (Cardiology Center)",
                "room_number": "120-B",
                "sub_room_name": "Arrhythmia & Palpitation Bay",
                "department": "Cardiology",
                "floor": "1st Floor, East Wing",
                "acuity_tier": "Urgent / Routine (KTAS 3)",
                "clinical_keywords": [
                    "heart fluttering", "heart pounding", "palpitations", "rapid heartbeat",
                    "irregular pulse", "skipped beats", "racing heart when resting", "arrhythmia"
                ],
                "routing_guidance": "Cardiac palpitations, irregular heartbeat rhythms, and non-hypotensive pulse irregularities."
            },
            {
                "room_id": "cardiac_120_c",
                "ward": "Ward 3 (Cardiology Center)",
                "room_number": "120-C",
                "sub_room_name": "Hypertension & Vascular Clinic",
                "department": "Cardiology",
                "floor": "1st Floor, East Wing",
                "acuity_tier": "Standard (KTAS 4)",
                "clinical_keywords": [
                    "high blood pressure", "hypertension spike", "pounding in ears from pressure",
                    "vascular consultation", "cardiac medication check"
                ],
                "routing_guidance": "Elevated blood pressure without acute organ damage or end-stage chest pain."
            },

            # ==========================================
            # WARD 4: ORTHOPEDICS & MUSCULOSKELETAL
            # ==========================================
            {
                "room_id": "ortho_105_a",
                "ward": "Ward 4 (Orthopedics & Fracture Care)",
                "room_number": "105-A",
                "sub_room_name": "Fracture & Splinting Bay",
                "department": "Orthopedics & Minor Injuries",
                "floor": "1st Floor, South Wing",
                "acuity_tier": "Routine / Urgent (KTAS 3-4)",
                "clinical_keywords": [
                    "broken bone", "fracture", "heard a snap", "bone deformity",
                    "cannot bear weight", "broken wrist", "broken finger", "x-ray bone"
                ],
                "routing_guidance": "Suspected bone fractures, visible skeletal deformities, and emergency splinting."
            },
            {
                "room_id": "ortho_105_b",
                "ward": "Ward 4 (Orthopedics & Fracture Care)",
                "room_number": "105-B",
                "sub_room_name": "Sprains & Soft Tissue Bay",
                "department": "Orthopedics & Minor Injuries",
                "floor": "1st Floor, South Wing",
                "acuity_tier": "Routine (KTAS 4-5)",
                "clinical_keywords": [
                    "sprained ankle", "twisted wrist", "knee twist", "swollen ankle",
                    "muscle pull", "joint pain from fall", "sports sprain", "limping"
                ],
                "routing_guidance": "Ligament sprains, tendon strains, twisted ankles, and localized joint swelling without deformity."
            },
            {
                "room_id": "ortho_105_c",
                "ward": "Ward 4 (Orthopedics & Fracture Care)",
                "room_number": "105-C",
                "sub_room_name": "Spine & Chronic Joint Bay",
                "department": "Orthopedics & Minor Injuries",
                "floor": "1st Floor, South Wing",
                "acuity_tier": "Routine (KTAS 5)",
                "clinical_keywords": [
                    "back pain", "lower back ache", "sciatica", "chronic knee pain",
                    "stiff shoulder", "neck spasm", "arthritis pain"
                ],
                "routing_guidance": "Non-traumatic back pain, lumbar strain, sciatica, and chronic arthritic joint complaints."
            },

            # ==========================================
            # WARD 5: EAR, NOSE & THROAT (ENT / OTOLARYNGOLOGY)
            # ==========================================
            {
                "room_id": "ent_15_a",
                "ward": "Ward 5 (ENT & Airway Clinic)",
                "room_number": "15-A",
                "sub_room_name": "Throat, Pharynx & Common Cold Bay",
                "department": "Ear, Nose & Throat (ENT)",
                "floor": "Ground Floor, West Corridor",
                "acuity_tier": "Routine (KTAS 4-5)",
                "clinical_keywords": [
                    "sore throat", "runny nose", "mild runny nose and cough", "tonsils swollen",
                    "difficulty swallowing food", "pharyngitis", "scratchy throat", "throat mucus", "common cold"
                ],
                "routing_guidance": "Sore throats, common cold symptoms, runny nose with cough/mucus, and tonsillar inflammation."
            },
            {
                "room_id": "ent_15_b",
                "ward": "Ward 5 (ENT & Airway Clinic)",
                "room_number": "15-B",
                "sub_room_name": "Otology & Ear Disorders Bay",
                "department": "Ear, Nose & Throat (ENT)",
                "floor": "Ground Floor, West Corridor",
                "acuity_tier": "Routine (KTAS 4)",
                "clinical_keywords": [
                    "ear pain", "ear infection", "blocked ear", "fluid in ear",
                    "ringing in ears", "tinnitus", "loss of hearing", "ear drainage"
                ],
                "routing_guidance": "Acute otitis media, ear canal infections, sudden hearing reduction, and ear discharge."
            },
            {
                "room_id": "ent_15_c",
                "ward": "Ward 5 (ENT & Airway Clinic)",
                "room_number": "15-C",
                "sub_room_name": "Rhinology & Sinus Bay",
                "department": "Ear, Nose & Throat (ENT)",
                "floor": "Ground Floor, West Corridor",
                "acuity_tier": "Routine (KTAS 4-5)",
                "clinical_keywords": [
                    "sinus pressure", "sinus headache", "congested sinuses", "nosebleed",
                    "epistaxis", "loss of smell", "nasal blockage"
                ],
                "routing_guidance": "Acute sinusitis, facial sinus pressure, recurring nosebleeds, and severe nasal congestion."
            },

            # ==========================================
            # WARD 6: GASTROENTEROLOGY & ABDOMINAL HEALTH
            # ==========================================
            {
                "room_id": "gastro_80_a",
                "ward": "Ward 6 (Digestive & Abdominal Health)",
                "room_number": "80-A",
                "sub_room_name": "Acute Gastroenteritis & Rehydration Bay",
                "department": "Gastroenterology",
                "floor": "Ground Floor, East Corridor",
                "acuity_tier": "Urgent / Routine (KTAS 3-4)",
                "clinical_keywords": [
                    "vomiting", "food poisoning", "diarrhea watery", "severe nausea",
                    "cannot keep water down", "stomach cramps vomiting", "gastroenteritis"
                ],
                "routing_guidance": "Acute food poisoning, dehydrating diarrhea, and vomiting requiring antiemetic therapy."
            },
            {
                "room_id": "gastro_80_b",
                "ward": "Ward 6 (Digestive & Abdominal Health)",
                "room_number": "80-B",
                "sub_room_name": "Abdominal Pain Assessment Bay",
                "department": "Gastroenterology",
                "floor": "Ground Floor, East Corridor",
                "acuity_tier": "Urgent (KTAS 3)",
                "clinical_keywords": [
                    "stomach pain", "abdominal cramps", "belly pain", "appendix area pain",
                    "acid reflux severe", "burning stomach ulcer", "gallbladder pain"
                ],
                "routing_guidance": "Localized abdominal pain, suspected appendicitis evaluation, or severe peptic ulcer discomfort."
            },

            # ==========================================
            # WARD 7: NEUROLOGY & NEURO-DIAGNOSTIC CLINIC
            # ==========================================
            {
                "room_id": "neuro_210_a",
                "ward": "Ward 7 (Neurology Tower)",
                "room_number": "210-A",
                "sub_room_name": "Headache & Migraine Infusion Bay",
                "department": "Neurology",
                "floor": "2nd Floor, Central Tower",
                "acuity_tier": "Routine / Urgent (KTAS 3-4)",
                "clinical_keywords": [
                    "migraine severe", "throbbing headache", "light sensitivity eyes",
                    "cluster headache", "aura headache", "tension headache"
                ],
                "routing_guidance": "Severe migraines, photophobia, and intractable tension or cluster headaches."
            },
            {
                "room_id": "neuro_210_b",
                "ward": "Ward 7 (Neurology Tower)",
                "room_number": "210-B",
                "sub_room_name": "Vestibular, Vertigo & Neuropathy Bay",
                "department": "Neurology",
                "floor": "2nd Floor, Central Tower",
                "acuity_tier": "Standard (KTAS 4)",
                "clinical_keywords": [
                    "dizziness", "vertigo room spinning", "loss of balance", "numbness in fingers",
                    "pins and needles feet", "neuropathy tingling", "shaky hands"
                ],
                "routing_guidance": "Peripheral neuropathies, benign positional vertigo, dizziness, and vestibular balance disturbances."
            },

            # ==========================================
            # WARD 8: PEDIATRIC URGENT CARE & CHILD HEALTH
            # ==========================================
            {
                "room_id": "pediatric_25_a",
                "ward": "Ward 8 (Pediatric Pavilion)",
                "room_number": "25-A",
                "sub_room_name": "Infant & Toddler Acute Bay",
                "department": "Pediatric Triage & Care",
                "floor": "Ground Floor, Yellow Wing",
                "acuity_tier": "Urgent (KTAS 2-3)",
                "clinical_keywords": [
                    "baby", "infant", "newborn", "toddler fever", "my 1 year old",
                    "baby crying uncontrollably", "infant not feeding", "croup cough child"
                ],
                "routing_guidance": "Infants, neonates, and toddlers with fever, acute distress, or feeding refusal."
            },
            {
                "room_id": "pediatric_25_b",
                "ward": "Ward 8 (Pediatric Pavilion)",
                "room_number": "25-B",
                "sub_room_name": "Child & Adolescent Outpatient Bay",
                "department": "Pediatric Triage & Care",
                "floor": "Ground Floor, Yellow Wing",
                "acuity_tier": "Routine (KTAS 4-5)",
                "clinical_keywords": [
                    "my 5 year old", "my child", "my son", "my daughter", "school child fever",
                    "pediatric rash", "child tummy ache"
                ],
                "routing_guidance": "General medical complaints and minor injuries in children aged 3 to 17."
            },

            # ==========================================
            # WARD 9: DERMATOLOGY & ALLERGY CLINIC
            # ==========================================
            {
                "room_id": "derma_112_a",
                "ward": "Ward 9 (Dermatology & Allergy)",
                "room_number": "112-A",
                "sub_room_name": "Acute Rash & Urticaria Bay",
                "department": "Dermatology & Skin Clinic",
                "floor": "1st Floor, North Corridor",
                "acuity_tier": "Routine / Urgent (KTAS 3-4)",
                "clinical_keywords": [
                    "skin rash spreading", "hives itchy", "allergic skin reaction", "welts on skin",
                    "swollen skin patches", "itchy rash all over"
                ],
                "routing_guidance": "Widespread urticaria, allergic skin eruptions, and hives without respiratory compromise."
            },
            {
                "room_id": "derma_112_b",
                "ward": "Ward 9 (Dermatology & Allergy)",
                "room_number": "112-B",
                "sub_room_name": "Infectious & General Skin Bay",
                "department": "Dermatology & Skin Clinic",
                "floor": "1st Floor, North Corridor",
                "acuity_tier": "Routine (KTAS 5)",
                "clinical_keywords": [
                    "eczema flare", "skin lesion", "boil", "blisters on skin",
                    "peeling skin", "athlete foot fungal", "shingles rash"
                ],
                "routing_guidance": "Localized dermatitis, chronic eczema, boils, and suspected localized dermatological infections."
            },

            # ==========================================
            # WARD 10: OPHTHALMOLOGY & EYE BAY
            # ==========================================
            {
                "room_id": "eye_130_a",
                "ward": "Ward 10 (Ophthalmology & Eye Bay)",
                "room_number": "130-A",
                "sub_room_name": "Acute Eye Trauma & Foreign Body Bay",
                "department": "Ophthalmology",
                "floor": "1st Floor, West Wing",
                "acuity_tier": "Urgent (KTAS 2-3)",
                "clinical_keywords": [
                    "chemical in eye", "foreign object in eye", "metal in eye",
                    "eye scratch severe", "sudden vision loss", "flashers and floaters in eye"
                ],
                "routing_guidance": "Chemical ocular injury, foreign bodies in eye, and acute vision emergencies."
            },
            {
                "room_id": "eye_130_b",
                "ward": "Ward 10 (Ophthalmology & Eye Bay)",
                "room_number": "130-B",
                "sub_room_name": "Ocular Infection & Conjunctivitis Bay",
                "department": "Ophthalmology",
                "floor": "1st Floor, West Wing",
                "acuity_tier": "Routine (KTAS 4-5)",
                "clinical_keywords": [
                    "pink eye", "conjunctivitis", "eye discharge sticky", "red eye itching",
                    "swollen eyelid", "stye on eyelid", "blurry eye discharge"
                ],
                "routing_guidance": "Bacterial or viral conjunctivitis, eyelid infections, and non-traumatic red eye."
            }
        ]
    }
    return directory


def generate_synthetic_eval_cases() -> List[Dict[str, Any]]:
    cases = [
        # --- CARDIOLOGY (Ward 3: Room 120-A or 120-B) ---
        {
            "id": "syn_cardiac_01",
            "transcript": "I have had this persistent tightness and pressure in my chest since this morning.",
            "expected_room_prefix": "120",
            "expected_dept": "Cardiology",
            "symptom_category": "cardiac",
            "expected_retrieval": True
        },
        {
            "id": "syn_cardiac_02",
            "transcript": "My heart feels like it is pounding out of my chest and fluttering when I sit down.",
            "expected_room_prefix": "120",
            "expected_dept": "Cardiology",
            "symptom_category": "cardiac",
            "expected_retrieval": True
        },

        # --- FEVER & INFECTION (Ward 1: Room 20-A or 20-B) ---
        {
            "id": "syn_fever_01",
            "transcript": "I have a high fever of 102 degrees and shivering chills that started last night.",
            "expected_room_prefix": "20",
            "expected_dept": "Fever & Acute Infection Clinic",
            "symptom_category": "fever",
            "expected_retrieval": True
        },
        {
            "id": "syn_fever_02",
            "transcript": "I am burning up with sweats, chills, and viral flu-like body aches.",
            "expected_room_prefix": "20",
            "expected_dept": "Fever & Acute Infection Clinic",
            "symptom_category": "fever",
            "expected_retrieval": True
        },

        # --- ENT / COLD / SORE THROAT (Ward 5: Room 15-A, 15-B, 15-C) ---
        {
            "id": "syn_ent_01",
            "transcript": "I have a mild runny nose and cough with a bit of mucus for 3 days.",
            "expected_room_prefix": "15",
            "expected_dept": "Ear, Nose & Throat (ENT)",
            "symptom_category": "ent",
            "expected_retrieval": True
        },
        {
            "id": "syn_ent_02",
            "transcript": "Severe ear pain and pressure in my left ear with blockage.",
            "expected_room_prefix": "15",
            "expected_dept": "Ear, Nose & Throat (ENT)",
            "symptom_category": "ent",
            "expected_retrieval": True
        },

        # --- ORTHOPEDICS (Ward 4: Room 105-A, 105-B) ---
        {
            "id": "syn_ortho_01",
            "transcript": "I twisted my ankle playing basketball and now it is swollen and I cannot put weight on it.",
            "expected_room_prefix": "105",
            "expected_dept": "Orthopedics & Minor Injuries",
            "symptom_category": "orthopedic",
            "expected_retrieval": True
        },
        {
            "id": "syn_ortho_02",
            "transcript": "I tripped on the stairs and heard a snap in my wrist, I think it is a broken bone.",
            "expected_room_prefix": "105",
            "expected_dept": "Orthopedics & Minor Injuries",
            "symptom_category": "orthopedic",
            "expected_retrieval": True
        },

        # --- RESPIRATORY & PULMONOLOGY (Ward 2: Room 45-A, 45-B) ---
        {
            "id": "syn_pulm_01",
            "transcript": "I have an asthma flare up with wheezing and my inhaler is not providing relief.",
            "expected_room_prefix": "45",
            "expected_dept": "Pulmonology & Respiratory Care",
            "symptom_category": "respiratory",
            "expected_retrieval": True
        },

        # --- GASTROENTEROLOGY (Ward 6: Room 80-A, 80-B) ---
        {
            "id": "syn_gastro_01",
            "transcript": "Severe abdominal cramps and continuous vomiting after eating seafood.",
            "expected_room_prefix": "80",
            "expected_dept": "Gastroenterology",
            "symptom_category": "gastro",
            "expected_retrieval": True
        },

        # --- EMERGENCY RESUSCITATION (Ward 0: Room 10-A, 10-B) ---
        {
            "id": "syn_emergency_01",
            "transcript": "He is losing massive blood from a deep wound and is starting to pass out.",
            "expected_room_prefix": "10",
            "expected_dept": "Emergency",
            "symptom_category": "emergency",
            "expected_retrieval": True
        },

        # --- NON-CLINICAL / IRRELEVANT (Expected Retrieval: None / 0 tokens) ---
        {
            "id": "syn_irrelevant_01",
            "transcript": "Hello good morning, could you tell me where the parking validation desk is?",
            "expected_room_prefix": None,
            "expected_dept": None,
            "symptom_category": "administrative",
            "expected_retrieval": False
        },
        {
            "id": "syn_irrelevant_02",
            "transcript": "Thank you nurse, I appreciate the help.",
            "expected_room_prefix": None,
            "expected_dept": None,
            "symptom_category": "chit_chat",
            "expected_retrieval": False
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
    print(f"✅ Generated {directory_file} ({len(directory_data['rooms'])} detailed sub-rooms across 11 hospital wards)")

    eval_data = generate_synthetic_eval_cases()
    with open(vignettes_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"✅ Generated {vignettes_file} ({len(eval_data)} synthetic evaluation test cases)")


if __name__ == "__main__":
    build_and_save_all()
