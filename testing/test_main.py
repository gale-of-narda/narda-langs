import json
import pytest

from pathlib import Path

from scripts.parser_procedure import Parser

parser_words = Parser(start_level=0, end_level=0)
parser_sentences = Parser(start_level=1, end_level=1)

path = Path("testing/testing.json")
with path.open("r", encoding="utf-8") as f:
    tests = json.load(f)

good_sentences = [tuple([pair[0], pair[1]]) for pair in tests["Sentences"]["good"]]
good_words = [tuple([pair[0], pair[1]]) for pair in tests["Words"]["good"]]


@pytest.mark.parametrize("good_string, mapping", good_sentences)
def test_good_sentences(good_string, mapping):
    parser_sentences.process(good_string)
    stances = [str(stance) for stance in parser_sentences.get_stances()]
    assert stances == mapping


@pytest.mark.parametrize("good_string, mapping", good_words)
def test_good_words(good_string, mapping):
    parser_words.process(good_string)
    stances = [str(stance) for stance in parser_words.get_stances()]
    assert stances == mapping


@pytest.mark.parametrize("bad_string", tests["Sentences"]["bad"])
def test_bad_sentences(bad_string):
    assert not parser_sentences.process(bad_string)


@pytest.mark.parametrize("bad_string", tests["Words"]["bad"])
def test_bad_words(bad_string):
    assert not parser_words.process(bad_string)
