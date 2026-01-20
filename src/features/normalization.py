"""Normalization and preprocessing for state features."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from .config import StateFeatureConfig, VITAL_FEATURES, LAB_FEATURES


class FeatureNormalizer:
    """Normalize features using various strategies."""

    def __init__(self, config: StateFeatureConfig = None):
        """
        Initialize normalizer.

        Args:
            config: State feature configuration
        """
        self.config = config
        self.stats = {}  # Store normalization statistics
        self.fitted = False

    def fit(self, features_df: pd.DataFrame, verbose=True):
        """
        Compute normalization statistics from training data.

        Args:
            features_df: DataFrame with feature columns
            verbose: Print statistics

        Returns:
            self
        """
        feature_cols = self._get_feature_columns(features_df)

        if verbose:
            print(f"Computing normalization statistics for {len(feature_cols)} features...")

        for col in feature_cols:
            data = features_df[col].dropna()

            if len(data) == 0:
                self.stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
                continue

            if self.config.normalization_method == "z-score":
                mean = data.mean()
                std = data.std()
                self.stats[col] = {
                    "mean": mean,
                    "std": std if std > 0 else 1.0,
                    "min": data.min(),
                    "max": data.max(),
                }

            elif self.config.normalization_method == "min-max":
                min_val = data.min()
                max_val = data.max()
                self.stats[col] = {
                    "min": min_val,
                    "max": max_val if max_val > min_val else min_val + 1.0,
                    "mean": data.mean(),
                    "std": data.std(),
                }

            elif self.config.normalization_method == "robust":
                # Use median and IQR (robust to outliers)
                median = data.median()
                q25 = data.quantile(0.25)
                q75 = data.quantile(0.75)
                iqr = q75 - q25
                self.stats[col] = {
                    "median": median,
                    "iqr": iqr if iqr > 0 else 1.0,
                    "min": data.min(),
                    "max": data.max(),
                }

        self.fitted = True

        if verbose:
            print("Normalization statistics computed")

        return self

    def transform(self, features_df: pd.DataFrame, clip=None) -> pd.DataFrame:
        """
        Apply normalization to features.

        Args:
            features_df: DataFrame with feature columns
            clip: Optional tuple (min, max) to clip normalized values

        Returns:
            DataFrame with normalized features
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        features_df = features_df.copy()
        feature_cols = self._get_feature_columns(features_df)

        for col in feature_cols:
            if col not in self.stats:
                continue

            stats = self.stats[col]

            if self.config.normalization_method == "z-score":
                # (x - mean) / std
                features_df[col] = (features_df[col] - stats["mean"]) / stats["std"]

            elif self.config.normalization_method == "min-max":
                # (x - min) / (max - min)
                features_df[col] = (features_df[col] - stats["min"]) / (
                    stats["max"] - stats["min"]
                )

            elif self.config.normalization_method == "robust":
                # (x - median) / IQR
                features_df[col] = (features_df[col] - stats["median"]) / stats["iqr"]

            # Clip outliers if requested
            if clip is not None:
                features_df[col] = features_df[col].clip(clip[0], clip[1])
            elif self.config.clip_outliers:
                # Clip at ±N std (or equivalent for other methods)
                limit = self.config.outlier_std
                features_df[col] = features_df[col].clip(-limit, limit)

        return features_df

    def fit_transform(self, features_df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(features_df, verbose=verbose)
        return self.transform(features_df)

    def inverse_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse normalization (for interpretability).

        Args:
            features_df: DataFrame with normalized features

        Returns:
            DataFrame with original scale features
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        features_df = features_df.copy()
        feature_cols = self._get_feature_columns(features_df)

        for col in feature_cols:
            if col not in self.stats:
                continue

            stats = self.stats[col]

            if self.config.normalization_method == "z-score":
                features_df[col] = features_df[col] * stats["std"] + stats["mean"]

            elif self.config.normalization_method == "min-max":
                features_df[col] = (
                    features_df[col] * (stats["max"] - stats["min"]) + stats["min"]
                )

            elif self.config.normalization_method == "robust":
                features_df[col] = features_df[col] * stats["iqr"] + stats["median"]

        return features_df

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns to normalize."""
        # Identify columns that are features (not metadata)
        exclude_cols = {"icustay_id", "hour", "hour_start", "hour_end", "hadm_id"}

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and not col.endswith("_raw")
        ]

        return feature_cols

    def get_stats_summary(self) -> pd.DataFrame:
        """Get summary of normalization statistics."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted first")

        summary = []
        for col, stats in self.stats.items():
            summary.append({"feature": col, **stats})

        return pd.DataFrame(summary)


def handle_missing_values(
    features_df: pd.DataFrame,
    strategy="mean",
    feature_means: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fill remaining missing values after forward-fill.

    Args:
        features_df: DataFrame with features (may have NaN)
        strategy: 'mean', 'median', 'zero', or 'normal_value'
        feature_means: Pre-computed means (if None, compute from data)

    Returns:
        (filled_df, fill_values_dict)
    """
    features_df = features_df.copy()

    # Get feature columns
    feature_cols = [
        col
        for col in features_df.columns
        if col
        not in {"icustay_id", "hour", "hour_start", "hour_end", "hadm_id"}
        and not col.endswith("_raw")
    ]

    if feature_means is None:
        feature_means = {}

        for col in feature_cols:
            if strategy == "mean":
                feature_means[col] = features_df[col].mean()
            elif strategy == "median":
                feature_means[col] = features_df[col].median()
            elif strategy == "zero":
                feature_means[col] = 0.0
            elif strategy == "normal_value":
                # Use midpoint of normal range if available
                # Extract feature name (remove trailing numbers/suffixes)
                base_name = col.replace("_norm", "").replace("_filled", "")

                if base_name in VITAL_FEATURES:
                    low, high = VITAL_FEATURES[base_name]["normal_range"]
                    feature_means[col] = (low + high) / 2
                elif base_name in LAB_FEATURES:
                    low, high = LAB_FEATURES[base_name]["normal_range"]
                    feature_means[col] = (low + high) / 2
                else:
                    feature_means[col] = 0.0  # Fallback

    # Fill missing values
    for col in feature_cols:
        if col in feature_means:
            features_df[col] = features_df[col].fillna(feature_means[col])

    return features_df, feature_means


def clip_outliers(features_df: pd.DataFrame, n_std=5.0) -> pd.DataFrame:
    """
    Clip extreme outliers to ±n_std from mean.

    Args:
        features_df: DataFrame with features
        n_std: Number of standard deviations for clipping

    Returns:
        DataFrame with clipped values
    """
    features_df = features_df.copy()

    feature_cols = [
        col
        for col in features_df.columns
        if col
        not in {"icustay_id", "hour", "hour_start", "hour_end", "hadm_id"}
        and not col.endswith("_raw")
    ]

    for col in feature_cols:
        data = features_df[col].dropna()
        if len(data) > 0:
            mean = data.mean()
            std = data.std()
            lower = mean - n_std * std
            upper = mean + n_std * std
            features_df[col] = features_df[col].clip(lower, upper)

    return features_df
