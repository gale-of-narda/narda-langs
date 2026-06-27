"""Generic, parser-agnostic helpers shared across the parser scripts.

Nothing here depends on the parsing logic itself: these are small utilities for
value coercion, resource resolution, YAML I/O, and binary/digit conversions that
were previously duplicated across the parser modules.
"""

import os
import sys
import yaml
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def parse_value(value):
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


def resource_path(relative_path):
    """Resolves a path against the application's base directory, accounting for
    the temporary extraction directory created by PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def read_yaml(path):
    """Loads and returns the object parsed from the YAML file at the given
    (resource-resolved) path.
    """
    full = Path(resource_path(path))
    try:
        with full.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        logger.exception(f"Failed to read YAML from {full}")
        raise


def binary(number, width=0):
    """Returns the binary digits of number, zero-padded to the given width."""
    return f"{number:b}".rjust(width, "0")


def digits(string):
    """Converts a string of digit characters into a list of integers."""
    return [int(c) for c in string]


def concat(sequence):
    """Joins a sequence into a single string with no separator."""
    return "".join(str(x) for x in sequence)
