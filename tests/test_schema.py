from src.ingest.schema import assert_valid_columns, validate_columns


def test_validate_columns_detects_missing_and_extra():
    columns = ["row_id", "subject_id", "hadm_id", "extra_col"]
    missing, extra, non_lowercase = validate_columns(columns, "ADMISSIONS.csv")
    assert "admittime" in missing
    assert extra == ["extra_col"]
    assert non_lowercase == []


def test_assert_valid_columns_passes_on_exact_match():
    columns = [
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
    ]
    assert_valid_columns(columns, "ADMISSIONS.csv")
