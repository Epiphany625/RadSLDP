import json

JSON_PATHS = [
    "../ReXGradient/metadata/train_metadata_view_position.json", 
    "../ReXGradient/metadata/valid_metadata_view_position.json",
    "../ReXGradient/metadata/test_metadata_view_position.json"
]
OUTPUT_PATHS = [
    "../ReXGradient/metadata/rexgradient_train.json",
    "../ReXGradient/metadata/rexgradient_valid.json",
    "../ReXGradient/metadata/rexgradient_test.json",
]
OUTPUT_PATHS_JSONLS = [
    "../ReXGradient/metadata/rexgradient_train.jsonl",
    "../ReXGradient/metadata/rexgradient_valid.jsonl",
    "../ReXGradient/metadata/rexgradient_test.jsonl",
]

ethnic_groups = set()
for i in range(len(JSON_PATHS)):
    JSON_PATH = JSON_PATHS[i]
    OUTPUT_PATH = OUTPUT_PATHS[i]
    OUTPUT_PATH_JSONL = OUTPUT_PATHS_JSONLS[i]
    
    def parse_constant(val):
        if val == "NaN":
            return None
        return val


    with open(JSON_PATH, "r") as f:
        data = json.load(f, parse_constant=parse_constant)


    def normalize_sex(sex):
        if sex == "M":
            return "male"
        if sex == "F":
            return "female"
        return None


    def normalize_age(age_str):
        """
        Convert things like:
        '007M' -> '7 months old'
        '036Y' -> '36 years old'
        Return None if invalid or missing.
        """
        if not age_str:
            return None
        if not isinstance(age_str, str) or len(age_str) < 2:
            return None

        unit = age_str[-1]          # e.g. 'Y' or 'M'
        num_str = age_str[:-1]      # e.g. '007'

        try:
            value = int(num_str)
        except ValueError:
            return None

        if unit == "Y":
            return f"{value} years old"
        elif unit == "M":
            return f"{value} months old"
        elif unit == "W":
            return f"{value} weeks old"
        elif unit == "D":
            return f"{value} days old"
        else:
            return None


    extracted_records = []

    found_findings = set()
    found_uids = set()
    found_impression = set()
    found_indication = set()

    for key, obj in data.items():
        # Mandatory fields with skip logic
        study_uid = obj.get("StudyInstanceUid")
        findings = obj.get("Findings")      # 'Findings' key from your metadata
        impression = obj.get("Impression")
        indication = obj.get("Indication")

        if (study_uid in found_uids) or (findings in found_findings) or (impression in found_impression) or (indication in found_indication):
            continue

        
        ethnic_group = obj.get("EthnicGroup") or None

        found_uids.add(study_uid)
        found_findings.add(findings)
        found_impression.add(impression)
        found_indication.add(indication)
        if ethnic_group is not None:
            ethnic_groups.add(ethnic_group)



        # Skip this object entirely if any of these are missing or None
        if not study_uid or findings is None or impression is None or indication is None:
            continue

        # Optional fields with normalization
        sex = normalize_sex(obj.get("PatientSex"))
        age = normalize_age(obj.get("PatientAge"))
        

        image_paths = obj.get("ImagePath", []) or []
        view_positions = obj.get("ImageViewPosition", []) or []

        # Build reason string with available patient info
        reason = indication
        if sex is not None:
            reason = f"Sex: {sex}. " + reason
        if age is not None:
            reason = f"Age: {age}. " + reason
        if ethnic_group is not None:
            reason = f"Ethnicity: {ethnic_group}. " + reason

        # If images and views are mismatched, skip this study
        if len(image_paths) != len(view_positions):
            continue

        # Base record shared across all images of this study
        base_record = {
            "id": study_uid,
            "indication": indication,
            "findings": findings,
            "reason": reason,
            "impression": impression,
            "history": None,
            "generate_method": "rule-based",
            "ethnicGroup": ethnic_group, 
            "sex": sex,
            "age": age,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nDescribe the findings of the chest x-ray given the following indications: {reason}.\n"
                },
                {
                    "from": "gpt",
                    "value": findings
                }
            ],
        }

        # Create one record per image/view pair
        for image_path, view_position in zip(image_paths, view_positions):
            if view_position != "PA" and view_position != "AP":
                continue 
            record = base_record.copy()  # shallow copy so each record is independent
            record["image"] = image_path
            record["view"] = view_position
            extracted_records.append(record)
            break # only include one image per ID. 
    
    extracted_records = [
        data for data in extracted_records 
        if data.get("view") in ("AP", "PA")
    ]


    with open(OUTPUT_PATH, "w") as f:
        json.dump(extracted_records, f, indent=2)

    print(f"Saved {len(extracted_records)} records to {OUTPUT_PATH}")

    # convert to JSONL
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    with open(OUTPUT_PATH_JSONL, "w", encoding="utf-8") as f:
        for item in loaded_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"Converted to jsonl: {OUTPUT_PATH_JSONL}")

print(f"ethnic groups: {ethnic_groups}")
