"""Generic helper classes and functions that do not belong to any particular
component of the parser.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ParsingFailure(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"{message}")
        return


class StructureError(Exception):
    """Raised when a parameter fails schema validation. A shape problem means the
    parameter's structure is incongruent with the rest of the parametrization; a
    type problem means its content type is wrong. A failure to parse the YAML
    itself raises a plain ValueError instead.
    """

    _SOURCES = {
        "rules_general": "general rules",
        "rules_special": "special rules",
        "alphabet": "alphabet",
        "dialect": "dialect",
    }

    @classmethod
    def shape(cls, param: str, source: str, problem: str) -> StructureError:
        """Builds an error for a structural (shape) problem."""
        return cls(
            f"Incongruent parametrization: {param} from "
            f"{cls._SOURCES.get(source, source)} {problem}"
        )

    @classmethod
    def wrong_type(cls, param: str, source: str, problem: str) -> StructureError:
        """Builds an error for a content-type problem."""
        return cls(
            f"Wrong parameter type: parameter {param} from "
            f"{cls._SOURCES.get(source, source)} {problem}"
        )


def is_int_list(value: object) -> bool:
    """True if value is a list of integers (booleans excluded)."""
    return isinstance(value, list) and all(
        isinstance(x, int) and not isinstance(x, bool) for x in value
    )


def lemb_rank_sizes(struct_level: list[int]) -> list[int]:
    """Expected entry counts for each compound-length rank of a structure level:
    one per node-count level, the k-th holding 2**(dichotomies up to k) entries.
    Zero-dichotomy ranks add no level, so a degenerate level keeps just the root.
    """
    sizes, total = [1], 0
    for d in struct_level:
        if d > 0:
            total += d
            sizes.append(2**total)
    return sizes


def parse_value(value: str) -> str | None:
    """Parses a string as YAML when possible (so numbers, lists, mappings and
    quoted strings are handled, consistently with the parameter files), otherwise
    returns it unchanged. Non-string values pass through untouched.
    """
    if not isinstance(value, str):
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def resource_path(relative_path: str) -> str:
    """Resolves a path against the application's base directory, accounting for
    the temporary extraction directory created by PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", None) or os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def read_yaml(path: str) -> dict:
    """Loads and returns the object parsed from the YAML file at the given
    (resource-resolved) path.
    """
    full = Path(resource_path(path))
    try:
        with full.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError, yaml.YAMLError:
        logger.exception(f"Failed to read YAML from {full}")
        raise


def binary(number: int, width: int = 0) -> str:
    """Returns the binary digits of number, zero-padded to the given width."""
    return f"{number:b}".rjust(width, "0")


def digits(string: str) -> list[int]:
    """Converts a string of digit characters into a list of integers."""
    return [int(c) for c in string]


def concat(sequence: list) -> str:
    """Joins a sequence into a single string with no separator."""
    return "".join(str(x) for x in sequence)
