# disease_prediction.py
import os
import re
import joblib
import pandas as pd
from difflib import get_close_matches

# ---------------- Config & paths ----------------
MODEL_PATH = "rf_disease_model.pkl"
SYMPTOM_LIST_PATH = "symptom_list.pkl"
CLASS_NAMES_PATH = "class_names.pkl"
MED_CSV_PATH = "disease_med.csv"
DATASET_PATH = "dataset_sorted.csv"

# ---------------- Utilities ----------------
def normalize_text_str(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------- Load medication CSV robustly ----------------
medication_map = {}
if os.path.exists(MED_CSV_PATH):
    try:
        med_df = pd.read_csv(MED_CSV_PATH, encoding="utf-8-sig")
        med_df.columns = [c.strip() for c in med_df.columns]
        # Accept either 'Disease' or 'disease'
        if 'Disease' in med_df.columns:
            med_col = 'Disease'
        elif 'disease' in med_df.columns:
            med_col = 'disease'
        else:
            raise KeyError(f"Expected 'Disease' column, found: {list(med_df.columns)}")

        # Determine medication column (Common Medications preferred)
        if 'Common Medications' in med_df.columns:
            med_map_col = 'Common Medications'
        else:
            # fallback to second column or raise
            if len(med_df.columns) >= 2:
                med_map_col = med_df.columns[1]
            else:
                raise KeyError("Could not find a medication column in disease_med.csv")

        med_df['disease_norm'] = med_df[med_col].astype(str).apply(normalize_text_str)
        medication_map = dict(zip(med_df['disease_norm'], med_df[med_map_col].astype(str)))
        print(f"Loaded medication lookup: {len(medication_map)} entries (from {MED_CSV_PATH})")
    except Exception as e:
        print(f"⚠️ Could not load medication data: {e}")
else:
    print(f"⚠️ Medication file not found at {MED_CSV_PATH}. Continuing without medication lookup.")

# ---------------- Load model and artifacts ----------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SYMPTOM_LIST_PATH) or not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError("Model artifacts missing. Run symptoms.py to create rf_disease_model.pkl, symptom_list.pkl, class_names.pkl")

model = joblib.load(MODEL_PATH)
symptoms = joblib.load(SYMPTOM_LIST_PATH)  # list of symptom names (order matters)
classes = joblib.load(CLASS_NAMES_PATH)

# Optional load dataset for overlap scoring
df = None
if os.path.exists(DATASET_PATH):
    try:
        df = pd.read_csv(DATASET_PATH).fillna(0)
    except Exception as e:
        print(f"⚠️ Could not load {DATASET_PATH} for overlap scoring: {e}")

# ---------------- Medication lookup ----------------
def get_medication_for_disease(disease_name: str):
    key = normalize_text_str(disease_name)
    if not key:
        return "No standard medication available"
    if key in medication_map:
        return medication_map[key]
    close = get_close_matches(key, medication_map.keys(), n=1, cutoff=0.55)
    if close:
        return medication_map[close[0]]
    return "No standard medication available"

# ---------------- Overlap scoring ----------------
def symptom_overlap(disease_data, given_symptoms):
    if not given_symptoms or disease_data is None or disease_data.empty:
        return 0.0
    common_cols = [s for s in given_symptoms if s in disease_data.columns]
    if not common_cols:
        return 0.0
    disease_profile = disease_data[common_cols].mean()
    matched_weight = disease_profile.sum()
    return float(matched_weight) / len(given_symptoms)

# ---------------- Prediction ----------------
def predict_disease(selected_symptoms):
    if not selected_symptoms:
        return [("No symptoms provided", 0.0, "No standard medication available")]

    # Normalize and match selected symptoms to available symptom list
    normalized_symptoms = []
    lower_symptom_map = {s.lower(): s for s in symptoms}
    for s in selected_symptoms:
        if s in symptoms:
            normalized_symptoms.append(s)
        else:
            low = str(s).lower()
            if low in lower_symptom_map:
                normalized_symptoms.append(lower_symptom_map[low])
            else:
                close = get_close_matches(low, list(lower_symptom_map.keys()), n=1, cutoff=0.6)
                if close:
                    normalized_symptoms.append(lower_symptom_map[close[0]])

    if not normalized_symptoms:
        return [("No valid symptoms matched to symptom list", 0.0, "No standard medication available")]

    patient_symptoms = {sym: 0 for sym in symptoms}
    for s in normalized_symptoms:
        patient_symptoms[s] = 1

    X_new = pd.DataFrame([patient_symptoms], columns=symptoms)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support probability predictions (predict_proba).")

    probs = model.predict_proba(X_new)[0]
    disease_scores = {}
    for idx, disease in enumerate(classes):
        ml_confidence = float(probs[idx])
        overlap_score = 0.0
        if df is not None:
            disease_data = df[df["diseases"] == disease]
            overlap_score = symptom_overlap(disease_data, normalized_symptoms)

        if ml_confidence < 0.1:
            final_score = (0.4 * ml_confidence) + (0.6 * overlap_score)
        else:
            final_score = (0.65 * ml_confidence) + (0.35 * overlap_score)
        disease_scores[disease] = final_score

    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    results = []
    for disease, score in sorted_diseases:
        percentage = round(score * 100, 2)
        meds = get_medication_for_disease(disease)
        if percentage > 0.5:
            results.append((disease, percentage, meds))

    if not results and sorted_diseases:
        disease, score = sorted_diseases[0]
        results.append((disease, round(score * 100, 2), get_medication_for_disease(disease)))

    return results

# ---------------- Validation ----------------
def validate_symptoms(selected_symptoms):
    valid = []
    invalid = []
    lower_symptom_map = {s.lower(): s for s in symptoms}
    for s in selected_symptoms:
        if s in symptoms:
            valid.append(s)
        elif str(s).lower() in lower_symptom_map:
            valid.append(lower_symptom_map[str(s).lower()])
        else:
            invalid.append(s)
    return valid, invalid

# ---------------- CLI / example usage ----------------
if __name__ == "__main__":
    # quick example — replace with your input list
    user_symptoms = ["fever", "cough", "fatigue"]

    valid, invalid = validate_symptoms(user_symptoms)
    if invalid:
        print(f"⚠️ Invalid/unmatched symptoms ignored: {invalid}")

    if valid:
        results = predict_disease(valid)
        print(f"\n🔍 Predictions for symptoms: {valid}\n")
        for rank, (disease, confidence, meds) in enumerate(results, 1):
            print(f"{rank}. {disease}: {confidence:.2f}%")
            print(f"   Common medications: {meds}\n")
    else:
        print("❌ No valid symptoms provided")
