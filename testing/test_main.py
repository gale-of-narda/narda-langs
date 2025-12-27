import json
import pytest

from pathlib import Path
from typing import List

from scripts.parser_procedure import Parser


def tuple_to_list(stances) -> List[List[List[int]]]:
    return [[[pos for pos in comp] for comp in st] for st in stances]


p0 = Parser(level=0)
p1 = Parser(level=1)

path = Path("testing/testing.json")
with path.open("r", encoding="utf-8") as f:
    tests = json.load(f)

good_sentences = [tuple([pair[0], pair[1]]) for pair in tests["Sentences"]["good"]]
good_words = [tuple([pair[0], pair[1]]) for pair in tests["Words"]["good"]]


@pytest.mark.parametrize("string, mapping", good_sentences)
def test_good_sentences(string, mapping):
    p0.parse(string)
    stances_list = tuple_to_list(p0.buffer.mapping.stances)
    assert stances_list == mapping


@pytest.mark.parametrize("string, mapping", good_words)
def test_good_words(string, mapping):
    p1.parse(string)
    stances_list = tuple_to_list(p1.buffer.mapping.stances)
    assert stances_list == mapping


@pytest.mark.parametrize("bad_string", tests["Sentences"]["bad"])
def test_bad_sentences(bad_string):
    assert not p0.parse(bad_string)


@pytest.mark.parametrize("bad_string", tests["Words"]["bad"])
def test_bad_words(bad_string):
    assert not p1.parse(bad_string)
