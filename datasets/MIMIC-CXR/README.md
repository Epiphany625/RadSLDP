# MIMIC-CXR

Scripts for preparing the MIMIC-CXR data used by the categorization pipeline.
Mirrors the `ReXGradient/` layout.

- **`download/`** — `download.sh` pulls the MIMIC-CXR-JPG `p10` images from
  PhysioNet (requires a credentialed PhysioNet login). For more than `p10`, see
  the PhysioNet download guide.
- **`processing/`** — restrict the LLaVA-Rad MIMIC-CXR annotation JSONs to the
  `p10` image subset:
  - download the annotations from PhysioNet
    (`llava-rad-mimic-cxr-annotation/1.0.0`),
  - run `filter.sh` (→ `filter_json_by_images.py`) to filter train/dev/test to
    images present in `p10`,
  - `filter_null_reasons_simple.py` drops entries with empty reason text.

> **Access:** you need a PhysioNet credentialed account and the relevant
> data-use agreements before downloading.

> CheXpert+ has no dedicated prep script here — its source CSV is read directly
> by `categorization/build_reason_index.py` (`section_clinical_history` /
> `section_history` fields).

> Paths inside `filter.sh` are hardcoded to a cluster layout; copied verbatim —
> edit `IMAGE_FOLDER` / `DATA_DIR` for your environment.
