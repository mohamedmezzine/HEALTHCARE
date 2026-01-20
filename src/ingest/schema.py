"""Expected schema for MIMIC-III CSV tables."""

import warnings

import pandas as pd

EXPECTED_COLUMNS = {
    "ADMISSIONS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "religion",
        "marital_status",
        "ethnicity",
        "edregtime",
        "edouttime",
        "diagnosis",
        "hospital_expire_flag",
        "has_chartevents_data",
    ],
    "CALLOUT.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "submit_wardid",
        "submit_careunit",
        "curr_wardid",
        "curr_careunit",
        "callout_wardid",
        "callout_service",
        "request_tele",
        "request_resp",
        "request_cdiff",
        "request_mrsa",
        "request_vre",
        "callout_status",
        "callout_outcome",
        "discharge_wardid",
        "acknowledge_status",
        "createtime",
        "updatetime",
        "acknowledgetime",
        "outcometime",
        "firstreservationtime",
        "currentreservationtime",
    ],
    "CAREGIVERS.csv": [
        "row_id",
        "cgid",
        "label",
        "description",
    ],
    "CHARTEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "itemid",
        "charttime",
        "storetime",
        "cgid",
        "value",
        "valuenum",
        "valueuom",
        "warning",
        "error",
        "resultstatus",
        "stopped",
    ],
    "CPTEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "costcenter",
        "chartdate",
        "cpt_cd",
        "cpt_number",
        "cpt_suffix",
        "ticket_id_seq",
        "sectionheader",
        "subsectionheader",
        "description",
    ],
    "D_CPT.csv": [
        "row_id",
        "category",
        "sectionrange",
        "sectionheader",
        "subsectionrange",
        "subsectionheader",
        "codesuffix",
        "mincodeinsubsection",
        "maxcodeinsubsection",
    ],
    "D_ICD_DIAGNOSES.csv": [
        "row_id",
        "icd9_code",
        "short_title",
        "long_title",
    ],
    "D_ICD_PROCEDURES.csv": [
        "row_id",
        "icd9_code",
        "short_title",
        "long_title",
    ],
    "D_ITEMS.csv": [
        "row_id",
        "itemid",
        "label",
        "abbreviation",
        "dbsource",
        "linksto",
        "category",
        "unitname",
        "param_type",
        "conceptid",
    ],
    "D_LABITEMS.csv": [
        "row_id",
        "itemid",
        "label",
        "fluid",
        "category",
        "loinc_code",
    ],
    "DATETIMEEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "itemid",
        "charttime",
        "storetime",
        "cgid",
        "value",
        "valueuom",
        "warning",
        "error",
        "resultstatus",
        "stopped",
    ],
    "DIAGNOSES_ICD.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "seq_num",
        "icd9_code",
    ],
    "DRGCODES.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "drg_type",
        "drg_code",
        "description",
        "drg_severity",
        "drg_mortality",
    ],
    "ICUSTAYS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "dbsource",
        "first_careunit",
        "last_careunit",
        "first_wardid",
        "last_wardid",
        "intime",
        "outtime",
        "los",
    ],
    "INPUTEVENTS_CV.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "charttime",
        "itemid",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "storetime",
        "cgid",
        "orderid",
        "linkorderid",
        "stopped",
        "newbottle",
        "originalamount",
        "originalamountuom",
        "originalroute",
        "originalrate",
        "originalrateuom",
        "originalsite",
    ],
    "INPUTEVENTS_MV.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "starttime",
        "endtime",
        "itemid",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "storetime",
        "cgid",
        "orderid",
        "linkorderid",
        "ordercategoryname",
        "secondaryordercategoryname",
        "ordercomponenttypedescription",
        "ordercategorydescription",
        "patientweight",
        "totalamount",
        "totalamountuom",
        "isopenbag",
        "continueinnextdept",
        "cancelreason",
        "statusdescription",
        "comments_editedby",
        "comments_canceledby",
        "comments_date",
        "originalamount",
        "originalrate",
    ],
    "LABEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "value",
        "valuenum",
        "valueuom",
        "flag",
    ],
    "MICROBIOLOGYEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "chartdate",
        "charttime",
        "spec_itemid",
        "spec_type_desc",
        "org_itemid",
        "org_name",
        "isolate_num",
        "ab_itemid",
        "ab_name",
        "dilution_text",
        "dilution_comparison",
        "dilution_value",
        "interpretation",
    ],
    "NOTEEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "chartdate",
        "charttime",
        "storetime",
        "category",
        "description",
        "cgid",
        "iserror",
        "text",
    ],
    "OUTPUTEVENTS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "charttime",
        "itemid",
        "value",
        "valueuom",
        "storetime",
        "cgid",
        "stopped",
        "newbottle",
        "iserror",
    ],
    "PATIENTS.csv": [
        "row_id",
        "subject_id",
        "gender",
        "dob",
        "dod",
        "dod_hosp",
        "dod_ssn",
        "expire_flag",
    ],
    "PRESCRIPTIONS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "startdate",
        "enddate",
        "drug_type",
        "drug",
        "drug_name_poe",
        "drug_name_generic",
        "formulary_drug_cd",
        "gsn",
        "ndc",
        "prod_strength",
        "dose_val_rx",
        "dose_unit_rx",
        "form_val_disp",
        "form_unit_disp",
        "route",
    ],
    "PROCEDUREEVENTS_MV.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "starttime",
        "endtime",
        "itemid",
        "value",
        "valueuom",
        "location",
        "locationcategory",
        "storetime",
        "cgid",
        "orderid",
        "linkorderid",
        "ordercategoryname",
        "secondaryordercategoryname",
        "ordercategorydescription",
        "isopenbag",
        "continueinnextdept",
        "cancelreason",
        "statusdescription",
        "comments_editedby",
        "comments_canceledby",
        "comments_date",
    ],
    "PROCEDURES_ICD.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "seq_num",
        "icd9_code",
    ],
    "SERVICES.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "transfertime",
        "prev_service",
        "curr_service",
    ],
    "TRANSFERS.csv": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "dbsource",
        "eventtype",
        "prev_careunit",
        "curr_careunit",
        "prev_wardid",
        "curr_wardid",
        "intime",
        "outtime",
        "los",
    ],
}


def expected_columns(table_name):
    """Return expected column list for a table name."""
    try:
        return EXPECTED_COLUMNS[table_name]
    except KeyError as exc:
        raise KeyError(f"Unknown table name: {table_name}") from exc


def validate_columns(columns, table_name):
    """Validate columns for a table; returns (missing, extra, non_lowercase)."""
    expected = list(expected_columns(table_name))
    expected_set = set(expected)
    actual_set = set(columns)
    missing = [col for col in expected if col not in actual_set]
    extra = [col for col in columns if col not in expected_set]
    non_lowercase = [col for col in columns if col != col.lower()]
    return missing, extra, non_lowercase


def assert_valid_columns(columns, table_name):
    """Raise ValueError if columns do not match expected schema."""
    missing, extra, non_lowercase = validate_columns(columns, table_name)
    if missing or extra or non_lowercase:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        if non_lowercase:
            parts.append(f"non_lowercase={non_lowercase}")
        details = ", ".join(parts)
        raise ValueError(f"Schema mismatch for {table_name}: {details}")


def coerce_datetime_columns(df):
    """Safely parse datetime/date columns with errors='coerce'."""
    for col in df.columns:
        if col.endswith("time") or col.endswith("date"):
            df[col] = df[col].str.strip() if df[col].dtype == object else df[col]
            df[col] = df[col].replace({"": pd.NA}) if df[col].dtype == object else df[col]
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df


def warn_deidentified_dob(df, table_name):
    """Warn when DOB values look de-identified (e.g., far in the past)."""
    if table_name != "PATIENTS.csv" or "dob" not in df.columns:
        return
    dob = pd.to_datetime(df["dob"], errors="coerce", utc=False)
    if dob.isna().all():
        return
    year = dob.dt.year
    suspicious = year < 1900
    if suspicious.any():
        count = int(suspicious.sum())
        warnings.warn(
            f"{table_name}: {count} DOB values look de-identified (year < 1900).",
            RuntimeWarning,
        )
