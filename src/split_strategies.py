"""
Enhanced data splitting strategies for air pollutant prediction models.

This module provides configurable data splitting strategies including:
- Chronological splits with flexible year ranges
- Random splits with configurable ratios
- Regional splits with geographic boundaries
- Rolling origin splits for time series validation
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

from .data_loader import (
    chronological_split,
    random_split,
    regional_split,
    rolling_origin_splits,
    FIELD_NAMES,
)


class SplitStrategy(Enum):
    """Enumeration of available data splitting strategies."""

    CHRONOLOGICAL = "chronological"
    RANDOM = "random"
    REGIONAL = "regional"
    ROLLING_ORIGIN = "rolling_origin"


@dataclass
class ChronologicalSplitConfig:
    """Configuration for chronological data splitting."""

    train_years: Optional[List[int]] = None
    val_years: Optional[List[int]] = None
    test_years: Optional[List[int]] = None
    train_start: int = 2001
    train_end: int = 2012
    val_year: int = 2013
    test_start: int = 2014
    test_end: Optional[int] = 2015
    year_column_index: int = 2


@dataclass
class RandomSplitConfig:
    """Configuration for random data splitting."""

    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42


@dataclass
class RegionalSplitConfig:
    """Configuration for regional data splitting."""

    test_region_bbox: Tuple[
        float, float, float, float
    ]  # (min_lon, min_lat, max_lon, max_lat)
    lat_col_name: str = "lat"
    lon_col_name: str = "lon"
    train_val_split_method: str = "chronological"  # or "random"
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42


@dataclass
class RollingOriginSplitConfig:
    """Configuration for rolling origin data splitting."""

    initial_train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    step_years: int = 1
    year_column_index: int = 2


@dataclass
class SplitConfiguration:
    """Complete configuration for data splitting."""

    strategy: SplitStrategy
    chronological: Optional[ChronologicalSplitConfig] = None
    random: Optional[RandomSplitConfig] = None
    regional: Optional[RegionalSplitConfig] = None
    rolling_origin: Optional[RollingOriginSplitConfig] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.strategy == SplitStrategy.CHRONOLOGICAL and self.chronological is None:
            self.chronological = ChronologicalSplitConfig()
        elif self.strategy == SplitStrategy.RANDOM and self.random is None:
            self.random = RandomSplitConfig()
        elif self.strategy == SplitStrategy.REGIONAL and self.regional is None:
            raise ValueError("Regional split requires regional configuration")
        elif (
            self.strategy == SplitStrategy.ROLLING_ORIGIN
            and self.rolling_origin is None
        ):
            self.rolling_origin = RollingOriginSplitConfig()


class DataSplitter:
    """Enhanced data splitter with configurable strategies."""

    def __init__(self, config: SplitConfiguration):
        """
        Initialize the data splitter with a configuration.

        Args:
            config: Split configuration specifying strategy and parameters
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the split configuration."""
        if self.config.strategy == SplitStrategy.REGIONAL:
            if self.config.regional is None:
                raise ValueError(
                    "Regional split strategy requires regional configuration"
                )

            bbox = self.config.regional.test_region_bbox
            if len(bbox) != 4:
                raise ValueError(
                    "test_region_bbox must have 4 values (min_lon, min_lat, max_lon, max_lat)"
                )

            min_lon, min_lat, max_lon, max_lat = bbox
            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError(
                    f"Invalid bounding box: min_lon ({min_lon}) must be < max_lon ({max_lon}) "
                    f"and min_lat ({min_lat}) must be < max_lat ({max_lat})"
                )

        if self.config.strategy == SplitStrategy.RANDOM:
            if self.config.random is None:
                raise ValueError("Random split strategy requires random configuration")

            train_frac = self.config.random.train_frac
            val_frac = self.config.random.val_frac

            if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
                raise ValueError("train_frac and val_frac must be within (0,1)")

            if train_frac + val_frac >= 1:
                raise ValueError(
                    "train_frac + val_frac must be < 1 to leave room for test set"
                )

    def split(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data according to the configured strategy.

        Args:
            data: Input data array to split

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        print(f"Applying {self.config.strategy.value} split strategy...")

        if self.config.strategy == SplitStrategy.CHRONOLOGICAL:
            return self._chronological_split(data)
        elif self.config.strategy == SplitStrategy.RANDOM:
            return self._random_split(data)
        elif self.config.strategy == SplitStrategy.REGIONAL:
            return self._regional_split(data)
        elif self.config.strategy == SplitStrategy.ROLLING_ORIGIN:
            return self._rolling_origin_split(data)
        else:
            raise ValueError(f"Unsupported split strategy: {self.config.strategy}")

    def _chronological_split(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply chronological split."""
        config = self.config.chronological

        # Log the configuration being used
        if config.train_years is not None:
            print(f"  Using year lists mode:")
            print(f"    Training years: {config.train_years}")
            print(f"    Validation years: {config.val_years}")
            print(f"    Test years: {config.test_years}")
        else:
            print(f"  Using range mode:")
            print(f"    Training: {config.train_start}-{config.train_end}")
            print(f"    Validation: {config.val_year}")
            print(f"    Test: {config.test_start}-{config.test_end}")

        return chronological_split(
            data,
            year_column_index=config.year_column_index,
            train_years=config.train_years,
            val_years=config.val_years,
            test_years=config.test_years,
            train_start=config.train_start,
            train_end=config.train_end,
            val_year=config.val_year,
            test_start=config.test_start,
            test_end=config.test_end,
        )

    def _random_split(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random split."""
        config = self.config.random

        print(f"  Train fraction: {config.train_frac}")
        print(f"  Validation fraction: {config.val_frac}")
        print(f"  Test fraction: {1 - config.train_frac - config.val_frac:.3f}")
        print(f"  Random seed: {config.seed}")

        return random_split(
            data,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            seed=config.seed,
        )

    def _regional_split(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply regional split."""
        config = self.config.regional

        print(f"  Test region bounding box: {config.test_region_bbox}")
        print(f"  Train/val split method: {config.train_val_split_method}")

        kwargs = {}
        if config.train_val_split_method == "random":
            kwargs.update(
                {
                    "train_frac": config.train_frac,
                    "val_frac": config.val_frac,
                    "seed": config.seed,
                }
            )

        return regional_split(
            data,
            test_region_bbox=config.test_region_bbox,
            lat_col_name=config.lat_col_name,
            lon_col_name=config.lon_col_name,
            train_val_split_method=config.train_val_split_method,
            **kwargs,
        )

    def _rolling_origin_split(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply rolling origin split (returns first split for compatibility)."""
        config = self.config.rolling_origin

        print(f"  Initial training years: {config.initial_train_years}")
        print(f"  Validation years: {config.val_years}")
        print(f"  Test years: {config.test_years}")
        print(f"  Step years: {config.step_years}")

        # Get all splits and return the first one for compatibility
        splits = rolling_origin_splits(
            data,
            year_column_index=config.year_column_index,
            initial_train_years=config.initial_train_years,
            val_years=config.val_years,
            test_years=config.test_years,
            step_years=config.step_years,
        )

        if not splits:
            raise ValueError("No valid rolling origin splits generated")

        print(f"  Generated {len(splits)} rolling splits, using first split")
        return splits[0]

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as a dictionary for logging."""
        config_dict = {
            "strategy": self.config.strategy.value,
        }

        if self.config.chronological:
            config_dict["chronological"] = asdict(self.config.chronological)
        if self.config.random:
            config_dict["random"] = asdict(self.config.random)
        if self.config.regional:
            config_dict["regional"] = asdict(self.config.regional)
        if self.config.rolling_origin:
            config_dict["rolling_origin"] = asdict(self.config.rolling_origin)

        return config_dict

    def save_config(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        config_dict = self.get_config_dict()

        # Helper function to convert numpy types to JSON-serializable types
        def make_json_serializable(obj):
            """Recursively convert numpy types to JSON-serializable types."""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: make_json_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj

        serializable_config = make_json_serializable(config_dict)

        with open(path, "w") as f:
            json.dump(serializable_config, f, indent=2)

        print(f"Split configuration saved to {path}")

    @classmethod
    def load_config(cls, path: Union[str, Path]) -> "DataSplitter":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        strategy = SplitStrategy(config_dict["strategy"])

        # Reconstruct configuration objects
        chronological = None
        if "chronological" in config_dict:
            chronological = ChronologicalSplitConfig(**config_dict["chronological"])

        random = None
        if "random" in config_dict:
            random = RandomSplitConfig(**config_dict["random"])

        regional = None
        if "regional" in config_dict:
            regional_dict = config_dict["regional"]
            # Convert bbox list back to tuple if needed
            if isinstance(regional_dict["test_region_bbox"], list):
                regional_dict["test_region_bbox"] = tuple(
                    regional_dict["test_region_bbox"]
                )
            regional = RegionalSplitConfig(**regional_dict)

        rolling_origin = None
        if "rolling_origin" in config_dict:
            rolling_origin = RollingOriginSplitConfig(**config_dict["rolling_origin"])

        config = SplitConfiguration(
            strategy=strategy,
            chronological=chronological,
            random=random,
            regional=regional,
            rolling_origin=rolling_origin,
        )

        return cls(config)


def create_chronological_split_config(
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    train_start: int = 2001,
    train_end: int = 2012,
    val_year: int = 2013,
    test_start: int = 2014,
    test_end: Optional[int] = 2015,
    year_column_index: int = 2,
) -> SplitConfiguration:
    """
    Create a chronological split configuration.

    Args:
        train_years: Specific years for training (overrides start/end if provided)
        val_years: Specific years for validation
        test_years: Specific years for testing
        train_start: Start year for training range
        train_end: End year for training range
        val_year: Single validation year
        test_start: Start year for test range
        test_end: End year for test range (None means all years >= test_start)
        year_column_index: Index of year column in data

    Returns:
        Complete split configuration
    """
    chronological_config = ChronologicalSplitConfig(
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        train_start=train_start,
        train_end=train_end,
        val_year=val_year,
        test_start=test_start,
        test_end=test_end,
        year_column_index=year_column_index,
    )

    return SplitConfiguration(
        strategy=SplitStrategy.CHRONOLOGICAL,
        chronological=chronological_config,
    )


def create_random_split_config(
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> SplitConfiguration:
    """
    Create a random split configuration.

    Args:
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        Complete split configuration
    """
    random_config = RandomSplitConfig(
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    return SplitConfiguration(
        strategy=SplitStrategy.RANDOM,
        random=random_config,
    )


def create_regional_split_config(
    test_region_bbox: Tuple[float, float, float, float],
    train_val_split_method: str = "chronological",
    lat_col_name: str = "lat",
    lon_col_name: str = "lon",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> SplitConfiguration:
    """
    Create a regional split configuration.

    Args:
        test_region_bbox: Bounding box for test region (min_lon, min_lat, max_lon, max_lat)
        train_val_split_method: Method to split non-test data ("chronological" or "random")
        lat_col_name: Name of latitude column
        lon_col_name: Name of longitude column
        train_frac: Fraction for training (used with random train/val split)
        val_frac: Fraction for validation (used with random train/val split)
        seed: Random seed

    Returns:
        Complete split configuration
    """
    regional_config = RegionalSplitConfig(
        test_region_bbox=test_region_bbox,
        lat_col_name=lat_col_name,
        lon_col_name=lon_col_name,
        train_val_split_method=train_val_split_method,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    return SplitConfiguration(
        strategy=SplitStrategy.REGIONAL,
        regional=regional_config,
    )


def create_rolling_origin_split_config(
    initial_train_years: int = 5,
    val_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
    year_column_index: int = 2,
) -> SplitConfiguration:
    """
    Create a rolling origin split configuration.

    Args:
        initial_train_years: Initial number of years for training
        val_years: Number of years for validation
        test_years: Number of years for testing
        step_years: Step size between splits
        year_column_index: Index of year column in data

    Returns:
        Complete split configuration
    """
    rolling_config = RollingOriginSplitConfig(
        initial_train_years=initial_train_years,
        val_years=val_years,
        test_years=test_years,
        step_years=step_years,
        year_column_index=year_column_index,
    )

    return SplitConfiguration(
        strategy=SplitStrategy.ROLLING_ORIGIN,
        rolling_origin=rolling_config,
    )


# Convenience functions for common split configurations
def get_default_chronological_config() -> SplitConfiguration:
    """Get default chronological split configuration."""
    return create_chronological_split_config()


def get_default_random_config() -> SplitConfiguration:
    """Get default random split configuration."""
    return create_random_split_config()


def parse_split_config_from_args(args) -> SplitConfiguration:
    """
    Parse split configuration from command line arguments.

    This function expects args to have attributes like:
    - split_strategy: str
    - test_region: Optional[str]
    - train_years: Optional[str] (comma-separated)
    - val_years: Optional[str] (comma-separated)
    - test_years: Optional[str] (comma-separated)
    - train_frac: Optional[float]
    - val_frac: Optional[float]
    - seed: Optional[int]
    """
    strategy_name = getattr(args, "split_strategy", "chronological").lower()

    if strategy_name == "regional" or getattr(args, "test_region", None):
        # Regional split
        if not hasattr(args, "test_region") or not args.test_region:
            raise ValueError("Regional split requires --test-region argument")

        try:
            min_lon, min_lat, max_lon, max_lat = map(float, args.test_region.split(","))
            test_region_bbox = (min_lon, min_lat, max_lon, max_lat)
        except ValueError:
            raise ValueError(
                "test_region must be in format 'min_lon,min_lat,max_lon,max_lat'"
            )

        train_val_method = getattr(args, "train_val_split_method", "chronological")

        return create_regional_split_config(
            test_region_bbox=test_region_bbox,
            train_val_split_method=train_val_method,
            train_frac=getattr(args, "train_frac", 0.8),
            val_frac=getattr(args, "val_frac", 0.1),
            seed=getattr(args, "seed", 42),
        )

    elif strategy_name == "random":
        # Random split
        return create_random_split_config(
            train_frac=getattr(args, "train_frac", 0.8),
            val_frac=getattr(args, "val_frac", 0.1),
            seed=getattr(args, "seed", 42),
        )

    elif strategy_name == "rolling_origin":
        # Rolling origin split
        return create_rolling_origin_split_config(
            initial_train_years=getattr(args, "initial_train_years", 5),
            val_years=getattr(args, "val_years", 1),
            test_years=getattr(args, "test_years", 1),
            step_years=getattr(args, "step_years", 1),
            year_column_index=getattr(args, "year_column", 2),
        )

    else:
        # Chronological split (default)
        train_years = None
        val_years = None
        test_years = None

        # Parse year lists if provided
        if hasattr(args, "train_years") and args.train_years:
            train_years = [int(y.strip()) for y in args.train_years.split(",")]

        if hasattr(args, "val_years") and args.val_years:
            val_years = [int(y.strip()) for y in args.val_years.split(",")]

        if hasattr(args, "test_years") and args.test_years:
            test_years = [int(y.strip()) for y in args.test_years.split(",")]

        return create_chronological_split_config(
            train_years=train_years,
            val_years=val_years,
            test_years=test_years,
            train_start=getattr(args, "train_start", 2001),
            train_end=getattr(args, "train_end", 2012),
            val_year=getattr(args, "val_year", 2013),
            test_start=getattr(args, "test_start", 2014),
            test_end=getattr(args, "test_end", 2015),
            year_column_index=getattr(args, "year_column", 2),
        )
