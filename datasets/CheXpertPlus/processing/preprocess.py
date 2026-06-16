import os
import pandas as pd
import numpy as np
import json
import uuid

# CSV produced by Stanford AIMI; place it next to this script or override via env.
CSV_PATH = os.environ.get(
    "CHEX_CSV",
    "../download/chexpertplus/df_chexpert_plus_240401.csv",
)
OUT_DIR = os.environ.get("CHEX_OUT_DIR", ".")
# sample JSON: 
#  {
#     "path_to_image": "train/patient42142/study5/view1_frontal.jpg",
#     "path_to_dcm": "train/patient42142/study5/view1_frontal.dcm",
#     "frontal_lateral": "Frontal",
#     "ap_pa": "AP",
#     "deid_patient_id": "patient42142",
#     "patient_report_date_order": 5,
#     "report": "NARRATIVE:\nChest 1 View, 8-8-2005\n\nHISTORY: 61 years Female, ICU patient\n\nCOMPARISON: 8/8/2005.\n\nIMPRESSION:\n\n1.TRACHEOSTOMY TUBE REMAINS IN PLACE.  RIGHT PICC IS IN STABLE AND STANDARD POSITION.  THE ENTERIC TUBE HAS BEEN REMOVED.\n\n2.NO EVIDENCE OF PNEUMOTHORAX.  SLIGHTLY IMPROVED AERATION OF THE BILATERAL LUNGS WITH NO EVIDENCE OF FOCAL AIR SPACE OPACITIES OR PLEURAL EFFUSIONS.\n\nSUMMARY: 2-ABNORMAL, PREVIOUSLY REPORTED\nI have personally reviewed the images for this examination and agreed with the report transcribed above.\n\nACCESSION NUMBER:\n9959089\nThis report has been anonymized. All dates are offset from the actual dates by a fixed interval associated with the patient.",
#     "section_narrative": "Chest 1 View, 8-8-2005",
#     "section_clinical_history": "61 years Female, ICU patient",
#     "section_history": null,
#     "section_comparison": "8/8/2005.",
#     "section_technique": null,
#     "section_procedure_comments": null,
#     "section_findings": "1.TRACHEOSTOMY TUBE REMAINS IN PLACE.  RIGHT PICC IS IN STABLE AND STANDARD POSITION.  THE ENTERIC TUBE HAS BEEN REMOVED.\n\n2.NO EVIDENCE OF PNEUMOTHORAX.  SLIGHTLY IMPROVED AERATION OF THE BILATERAL LUNGS WITH NO EVIDENCE OF FOCAL AIR SPACE OPACITIES OR PLEURAL EFFUSIONS.",
#     "section_impression": null,
#     "section_end_of_impression": "2-ABNORMAL, PREVIOUSLY REPORTED\nI have personally reviewed the images for this examination and agreed with the report transcribed above.",
#     "section_summary": null,
#     "section_accession_number": "9959089",
#     "age": 62.0,
#     "sex": "Female",
#     "race": "White",
#     "ethnicity": "Non-Hispanic/Non-Latino",
#     "interpreter_needed": "No",
#     "insurance_type": "Private Insurance",
#     "recent_bmi": 22.2,
#     "deceased": "No",
#     "split": "train"
#   }

def fix_capitalization(text):
    """Convert all-caps text to sentence case (capitalize only the start of each sentence)."""
    if pd.isna(text) or not text:
        return text
    text = text.lower()
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    return '. '.join(sentences)


def extract_reason(section_clinical_history):
    """Extract reason from section_clinical_history by finding content after ', '."""
    if pd.isna(section_clinical_history) or not section_clinical_history or section_clinical_history == "nan":
        return None
    return section_clinical_history


def print_split_counts(df):
    """Print the number of samples in train and valid splits."""
    train_count = len(df[df['split'] == 'train'])
    valid_count = len(df[df['split'] == 'valid'])
    print(f"  -> train: {train_count}, valid: {valid_count}")


os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

print(f"Original dataset size: {len(df)}")
print_split_counts(df)

# Drop images with lateral view (keep only Frontal)
df = df[df['frontal_lateral'] == 'Frontal']
print(f"After dropping lateral views: {len(df)}")
print_split_counts(df)

# Drop all images where path_to_image or section_findings is null
df = df.dropna(subset=['path_to_image', 'section_findings'])
print(f"After dropping null path_to_image or section_findings: {len(df)}")
print_split_counts(df)

# Drop duplicates based on path_to_image
df = df.drop_duplicates(subset=['path_to_image'], keep='first')
print(f"After dropping duplicate path_to_image: {len(df)}")
print_split_counts(df)

# Drop duplicates based on section_findings
df = df.drop_duplicates(subset=['section_findings'], keep='first')
print(f"After dropping duplicate section_findings: {len(df)}")
print_split_counts(df)

# Print unique values for race, sex, and ethnicity
print(f"\nUnique values for race: {df['race'].unique().tolist()}")
print(f"Unique values for sex: {df['sex'].unique().tolist()}")
print(f"Unique values for ethnicity: {df['ethnicity'].unique().tolist()}")

# Process data and organize by split
data_by_split = {}

for idx, row in df.iterrows():
    path_to_image = row['path_to_image'].strip().replace("\n", " ").replace("  ", " ")
    frontal_lateral = row['frontal_lateral'].strip().replace("\n", " ").replace("  ", " ")
    ap_pa = row['ap_pa'].strip().replace("\n", " ").replace("  ", " ")
    age = row['age']
    sex = row['sex'].strip().replace("\n", " ").replace("  ", " ")
    race = row['race'].strip().replace("\n", " ").replace("  ", " ")
    section_findings = row['section_findings'].strip().replace("\n", " ").replace("  ", " ")
    split = row['split'].strip().replace("\n", " ").replace("  ", " ")
    section_clinical_history = str(row['section_clinical_history']).strip().replace("\n", " ").replace("  ", " ")
    ethnicity = row['ethnicity'].strip().replace("\n", " ").replace("  ", " ")

    # Extract reason from section_clinical_history
    reason = extract_reason(section_clinical_history)
    if reason is None:
        continue
    reason = reason.strip()

    # Fix capitalization of section_findings and strip
    section_findings = fix_capitalization(section_findings)
    if section_findings:
        section_findings = section_findings.strip()

    # Uncapitalize sex (convert to lowercase)
    sex_lower = sex.strip().lower() if pd.notna(sex) else None
    # Drop sex if "Unknown"
    if sex_lower and sex_lower.lower() == 'unknown':
        sex_lower = None

    # Handle missing values - store as None if missing, strip all strings
    age_str = str(int(age)) if pd.notna(age) else None
    race_str = race.strip() if pd.notna(race) else None
    # Drop race if "Unknown" or "Patient Refused"
    if race_str and race_str in ['Unknown', 'Patient Refused']:
        race_str = None
    ap_pa_str = ap_pa.strip() if pd.notna(ap_pa) else None
    ethnicity_str = ethnicity.strip() if pd.notna(ethnicity) else None
    # Drop ethnicity if "Unknown" or "Patient Refused"
    if ethnicity_str and ethnicity_str in ['Unknown', 'Patient Refused']:
        ethnicity_str = None
    path_to_image_str = path_to_image.strip() if pd.notna(path_to_image) else None

    # Generate a unique ID
    unique_id = str(uuid.uuid4())

    if race_str is not None and race_str not in reason.lower():
        reason = f"Race: {race_str}. " + reason
    if age_str is not None and age_str not in reason.lower():
        reason = f"Age: {age_str}. " + reason
    if sex_lower is not None and sex_lower not in reason.lower():
        reason = f"Sex: {sex_lower}. " + reason
    

    # Form the output structure
    entry = {
        "id": unique_id,
        "indication": reason,
        "findings": section_findings,
        "ethnicity": ethnicity_str,
        "reason": reason,
        "generate_method": "rule-based",
        "race": race_str,
        "sex": sex_lower,
        "age": age_str,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nDescribe the findings of the chest x-ray given the following indications: {reason}\n"
            },
            {
                "from": "gpt",
                "value": section_findings
            }
        ],
        "image": path_to_image_str,
        "view": ap_pa_str
    }

    if split not in data_by_split:
        data_by_split[split] = []
    data_by_split[split].append(entry)

# Save to JSON and JSONL files based on split
for split, data in data_by_split.items():
    # Save as JSON
    json_filename = os.path.join(OUT_DIR, f"chexpert_plus_{split}.json")
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data)} entries to {json_filename}")

    # Save as JSONL
    jsonl_filename = os.path.join(OUT_DIR, f"chexpert_plus_{split}.jsonl")
    with open(jsonl_filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved {len(data)} entries to {jsonl_filename}")

print("\nProcessing complete!")

