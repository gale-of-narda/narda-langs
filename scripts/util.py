"""Generic helper classes and functions that do not belong to any particular
component of the parser.
"""

import logging
import os
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeGuard

logger = logging.getLogger(__name__)


class ParsingFailure(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"{message}")
        return


def is_int(value: object) -> TypeGuard[int]:
    """True if value is an integer (booleans excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def flatten(nested: object) -> Iterator[Any]:
    """Yields the non-list leaf values of an arbitrarily nested list."""
    if isinstance(nested, list):
        for item in nested:
            yield from flatten(item)
    else:
        yield nested


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


def tree_node_count(struct_level: list[int]) -> int:
    """Total number of nodes in a tree of the given structure: the root plus
    every successive rank of dichotomy splits, i.e. the sum of the per-rank node
    counts (see lemb_rank_sizes).
    """
    return sum(lemb_rank_sizes(struct_level))


def parse_value(value: str) -> object:
    """Parses a string as a TOML value when possible (so numbers, arrays, inline
    tables and quoted strings are handled, consistently with the parameter
    files), otherwise returns it unchanged. Non-string values pass through
    untouched.
    """
    if not isinstance(value, str):
        return value
    try:
        return tomllib.loads(f"_ = {value}")["_"]
    except tomllib.TOMLDecodeError:
        return value


def resource_path(relative_path: str) -> str:
    """Resolves a path against the application's base directory, accounting for
    the temporary extraction directory created by PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", None) or os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def read_toml(path: str) -> dict:
    """Loads and returns the object parsed from the TOML file at the given
    (resource-resolved) path.
    """
    full = Path(resource_path(path))
    try:
        with full.open("rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        logger.exception(f"Failed to read TOML from {full}")
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
