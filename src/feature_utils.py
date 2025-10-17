"""
Utility functions for handling feature names and metadata.
"""

import json
import os
from typing import List, Optional
from pathlib import Path


def load_feature_names_from_json(
    json_path: str = "data/final_column_names.json",
) -> List[str]:
    """
    Load feature names from the final_column_names.json file.

    Args:
        json_path: Path to the JSON file containing column names

    Returns:
        List of feature names

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the JSON file is invalid or empty
    """
    # Convert to Path object for better handling
    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(f"Feature names file not found: {json_path}")

    try:
        with open(json_file, "r") as f:
            feature_names = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in feature names file {json_path}: {e}")

    if not isinstance(feature_names, list):
        raise ValueError(f"Expected list of feature names, got {type(feature_names)}")

    if not feature_names:
        raise ValueError("Feature names list is empty")

    # Validate that all items are strings
    for i, name in enumerate(feature_names):
        if not isinstance(name, str):
            raise ValueError(f"Feature name at index {i} is not a string: {name}")

    print(f"Loaded {len(feature_names)} feature names from {json_path}")
    return feature_names


def get_feature_names_for_model(
    data_dict: dict,
    model_input_shape: tuple,
    fallback_json_path: str = "data/final_column_names.json",
) -> List[str]:
    """
    Get feature names for a model, trying multiple sources in order of preference.

    Args:
        data_dict: Dictionary containing data and metadata (from data loader)
        model_input_shape: Shape of the model input (samples, timesteps, features)
        fallback_json_path: Path to JSON file with feature names as fallback

    Returns:
        List of feature names matching the model's expected number of features

    Raises:
        ValueError: If no valid feature names can be obtained
    """
    expected_n_features = model_input_shape[2]  # (samples, timesteps, features)
    feature_names = None

    # Method 1: Try to get from data_dict column_names + feature_indices
    if "column_names" in data_dict and "feature_indices" in data_dict:
        feature_indices = data_dict.get("feature_indices", [])
        column_names = data_dict.get("column_names", [])

        if feature_indices and column_names:
            try:
                feature_names = [
                    column_names[i] for i in feature_indices if i < len(column_names)
                ]
                print(
                    f"Method 1: Got {len(feature_names)} feature names from data_dict column_names + feature_indices"
                )
            except (IndexError, TypeError) as e:
                print(f"Method 1 failed: {e}")
                feature_names = None

    # Method 2: Try to get from data_dict feature_names directly
    if not feature_names and "feature_names" in data_dict:
        feature_names = data_dict.get("feature_names", None)
        if feature_names:
            print(
                f"Method 2: Got {len(feature_names)} feature names from data_dict feature_names"
            )

    # Method 3: Try to load from JSON file
    if not feature_names:
        try:
            all_feature_names = load_feature_names_from_json(fallback_json_path)

            # If we have feature_indices, use them to select from all features
            if "feature_indices" in data_dict:
                feature_indices = data_dict.get("feature_indices", [])
                if feature_indices:
                    feature_names = [
                        all_feature_names[i]
                        for i in feature_indices
                        if i < len(all_feature_names)
                    ]
                    print(
                        f"Method 3a: Selected {len(feature_names)} feature names from JSON using feature_indices"
                    )

            # If no feature_indices or selection failed, use first N features
            if not feature_names:
                feature_names = all_feature_names[:expected_n_features]
                print(
                    f"Method 3b: Using first {len(feature_names)} feature names from JSON"
                )

        except (FileNotFoundError, ValueError) as e:
            print(f"Method 3 failed: {e}")
            feature_names = None

    # Method 4: Generate generic names as last resort
    if not feature_names:
        feature_names = [f"Feature_{i}" for i in range(expected_n_features)]
        print(f"Method 4: Generated {len(feature_names)} generic feature names")

    # Validate the final result
    if len(feature_names) != expected_n_features:
        print(
            f"Warning: Got {len(feature_names)} feature names but model expects {expected_n_features}"
        )

        if len(feature_names) > expected_n_features:
            # Truncate if we have too many
            feature_names = feature_names[:expected_n_features]
            print(f"Truncated to {len(feature_names)} feature names")
        else:
            # Pad with generic names if we have too few
            while len(feature_names) < expected_n_features:
                feature_names.append(f"Feature_{len(feature_names)}")
            print(f"Padded to {len(feature_names)} feature names")

    # Final validation
    if len(feature_names) != expected_n_features:
        raise ValueError(
            f"Could not obtain correct number of feature names. "
            f"Expected {expected_n_features}, got {len(feature_names)}"
        )

    print(
        f"Final feature names ({len(feature_names)}): {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}"
    )
    return feature_names


def validate_feature_names(feature_names: List[str], expected_count: int) -> bool:
    """
    Validate that feature names are properly formatted and have the expected count.

    Args:
        feature_names: List of feature names to validate
        expected_count: Expected number of features

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(feature_names, list):
        print(f"Feature names must be a list, got {type(feature_names)}")
        return False

    if len(feature_names) != expected_count:
        print(f"Expected {expected_count} feature names, got {len(feature_names)}")
        return False

    for i, name in enumerate(feature_names):
        if not isinstance(name, str):
            print(f"Feature name at index {i} is not a string: {name}")
            return False
        if not name.strip():
            print(f"Feature name at index {i} is empty or whitespace")
            return False

    return True
