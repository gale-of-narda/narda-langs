import json
import pytest
from parser_procedure import Parser
from pathlib import Path

p0 = Parser()
p0.load_params(level=0)
p1 = Parser()
p1.load_params(level=1)

path = Path("testing.json")
with path.open("r", encoding="utf-8") as f:
    tests = json.load(f)

# Testing sentences
@pytest.mark.parametrize("good_string", tests["Sentences"]["good"])
def test_good_sentences(good_string):
    assert p0.parse(good_string)

@pytest.mark.parametrize("bad_string", tests["Sentences"]["bad"])
def test_bad_sentences(bad_string):
    assert not p0.parse(bad_string)

# Testing words
@pytest.mark.parametrize("good_string", tests["Words"]["good"])
def test_good_words(good_string):
    assert p1.parse(good_string)

@pytest.mark.parametrize("bad_string", tests["Words"]["bad"])
def test_bad_words(bad_string):
    assert not p1.parse(bad_string)