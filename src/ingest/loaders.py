"""CSV loaders with schema validation."""

import pandas as pd

from .schema import assert_valid_columns


def load_csv(path, table_name, **read_csv_kwargs):
    """Load a CSV file and validate its columns."""
    df = pd.read_csv(path, **read_csv_kwargs)
    assert_valid_columns(df.columns, table_name)
    return df

