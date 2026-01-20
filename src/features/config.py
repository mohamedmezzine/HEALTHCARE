"""Configuration for state features.

Defines which vital signs and lab values to include in the state space.
Based on clinical importance and data availability in MIMIC-III.
"""

from dataclasses import dataclass
from typing import Dict, List


# Vital sign item IDs from D_ITEMS
# Identified from notebooks/02_vitals_and_labs_analysis.ipynb
VITAL_FEATURES = {
    "heart_rate": {
        "itemids": [211, 220045],  # Heart Rate
        "normal_range": (60, 100),
        "units": "bpm",
        "fill_method": "forward",
    },
    "sbp": {  # Systolic Blood Pressure
        "itemids": [51, 442, 455, 6701, 220179, 220050],
        "normal_range": (90, 140),
        "units": "mmHg",
        "fill_method": "forward",
    },
    "dbp": {  # Diastolic Blood Pressure
        "itemids": [8368, 8440, 8441, 8555, 220180, 220051],
        "normal_range": (60, 90),
        "units": "mmHg",
        "fill_method": "forward",
    },
    "mbp": {  # Mean Blood Pressure
        "itemids": [456, 52, 6702, 443, 220052, 220181, 225312],
        "normal_range": (70, 105),
        "units": "mmHg",
        "fill_method": "forward",
    },
    "resp_rate": {  # Respiratory Rate
        "itemids": [615, 618, 220210, 224690],
        "normal_range": (12, 20),
        "units": "breaths/min",
        "fill_method": "forward",
    },
    "spo2": {  # Oxygen Saturation
        "itemids": [646, 220277],
        "normal_range": (95, 100),
        "units": "%",
        "fill_method": "forward",
    },
    "temperature": {  # Temperature (Celsius)
        "itemids": [223761, 678],  # Fahrenheit items: 223762, 676
        "normal_range": (36.5, 37.5),
        "units": "°C",
        "fill_method": "forward",
    },
    "gcs": {  # Glasgow Coma Scale
        "itemids": [198, 226755, 227013],  # Total GCS
        "normal_range": (13, 15),
        "units": "points",
        "fill_method": "forward",
    },
}

# Lab test item IDs from D_LABITEMS
LAB_FEATURES = {
    "lactate": {
        "itemids": [50813],  # Lactate
        "normal_range": (0.5, 2.0),
        "units": "mmol/L",
        "fill_method": "forward",
        "importance": "high",  # Key sepsis marker
    },
    "creatinine": {
        "itemids": [50912],  # Creatinine
        "normal_range": (0.7, 1.2),
        "units": "mg/dL",
        "fill_method": "forward",
        "importance": "high",  # Kidney function
    },
    "bun": {  # Blood Urea Nitrogen
        "itemids": [51006],
        "normal_range": (7, 20),
        "units": "mg/dL",
        "fill_method": "forward",
        "importance": "medium",
    },
    "wbc": {  # White Blood Cell Count
        "itemids": [51300, 51301],
        "normal_range": (4.5, 11.0),
        "units": "K/uL",
        "fill_method": "forward",
        "importance": "high",  # Infection marker
    },
    "hemoglobin": {
        "itemids": [50811, 51222],
        "normal_range": (12.0, 16.0),
        "units": "g/dL",
        "fill_method": "forward",
        "importance": "medium",
    },
    "platelets": {
        "itemids": [51265],
        "normal_range": (150, 400),
        "units": "K/uL",
        "fill_method": "forward",
        "importance": "medium",
    },
    "sodium": {
        "itemids": [50824, 50983],
        "normal_range": (136, 145),
        "units": "mEq/L",
        "fill_method": "forward",
        "importance": "medium",
    },
    "potassium": {
        "itemids": [50822, 50971],
        "normal_range": (3.5, 5.0),
        "units": "mEq/L",
        "fill_method": "forward",
        "importance": "high",  # Critical for cardiac function
    },
    "bicarbonate": {
        "itemids": [50882],
        "normal_range": (22, 28),
        "units": "mEq/L",
        "fill_method": "forward",
        "importance": "medium",
    },
    "glucose": {
        "itemids": [50809, 50931],
        "normal_range": (70, 140),
        "units": "mg/dL",
        "fill_method": "forward",
        "importance": "medium",
    },
}


@dataclass
class StateFeatureConfig:
    """Configuration for state feature extraction."""

    # Which features to include
    vital_features: List[str] = None
    lab_features: List[str] = None

    # Aggregation settings
    vital_aggregation: str = "last"  # last, mean, median
    lab_aggregation: str = "last"  # last, mean, median

    # Missing data handling
    forward_fill_limit: int = 6  # Max hours to forward-fill (None = unlimited)
    backward_fill: bool = False  # Fill first hours backward from first obs

    # Normalization
    normalize: bool = True
    normalization_method: str = "z-score"  # z-score, min-max, robust

    # Outlier handling
    clip_outliers: bool = True
    outlier_std: float = 5.0  # Clip at mean ± N*std

    def __post_init__(self):
        """Set defaults if not provided."""
        if self.vital_features is None:
            # Use all vitals by default
            self.vital_features = list(VITAL_FEATURES.keys())

        if self.lab_features is None:
            # Use high-importance labs by default
            self.lab_features = [
                name
                for name, config in LAB_FEATURES.items()
                if config.get("importance") == "high"
            ]

    def get_vital_itemids(self) -> Dict[str, List[int]]:
        """Get mapping of vital name to item IDs."""
        return {
            name: VITAL_FEATURES[name]["itemids"]
            for name in self.vital_features
            if name in VITAL_FEATURES
        }

    def get_lab_itemids(self) -> Dict[str, List[int]]:
        """Get mapping of lab name to item IDs."""
        return {
            name: LAB_FEATURES[name]["itemids"]
            for name in self.lab_features
            if name in LAB_FEATURES
        }

    def num_features(self) -> int:
        """Total number of state features."""
        return len(self.vital_features) + len(self.lab_features)


# Default configuration
DEFAULT_CONFIG = StateFeatureConfig(
    vital_features=[
        "heart_rate",
        "sbp",
        "dbp",
        "mbp",
        "resp_rate",
        "spo2",
        "temperature",
        "gcs",
    ],
    lab_features=["lactate", "creatinine", "wbc", "potassium"],
    vital_aggregation="last",
    lab_aggregation="last",
    forward_fill_limit=6,
    normalize=True,
    clip_outliers=True,
)
