import json
import pytest

from pathlib import Path

from scripts.parser_procedure import Processor

prc_words = Processor(max_level=0)
prc_sentences = Processor(max_level=1)
prc_statements = Processor()

path = Path("testing/testing.json")
with path.open("r", encoding="utf-8") as f:
    tests = json.load(f)

good_statements = [tuple([i[0], i[1], i[2]]) for i in tests["Statements"]["good"]]
good_sentences = [tuple([i[0], i[1]]) for i in tests["Sentences"]["good"]]
good_words = [tuple([i[0], i[1]]) for i in tests["Words"]["good"]]


@pytest.mark.parametrize("good_string, mapping_0, mapping_1", good_statements)
def test_good_statements(good_string, mapping_0, mapping_1):
    prc_statements.process(good_string)
    stances_0 = [str(stance) for stance in prc_statements.get_stances(lvl=0)]
    stances_1 = [str(stance) for stance in prc_statements.get_stances(lvl=1)]
    assert stances_0 == mapping_0 and stances_1 == mapping_1


@pytest.mark.parametrize("bad_string", tests["Statements"]["bad"])
def test_bad_statements(bad_string):
    assert not prc_statements.process(bad_string)


@pytest.mark.parametrize("good_string, mapping", good_sentences)
def test_good_sentences(good_string, mapping):
    prc_sentences.process(good_string)
    stances = [str(stance) for stance in prc_sentences.get_stances(lvl=1)]
    assert stances == mapping


@pytest.mark.parametrize("bad_string", tests["Sentences"]["bad"])
def test_bad_sentences(bad_string):
    assert not prc_sentences.process(bad_string)


@pytest.mark.parametrize("good_string, mapping", good_words)
def test_good_words(good_string, mapping):
    prc_words.process(good_string)
    stances = [str(stance) for stance in prc_words.get_stances(lvl=0)]
    assert stances == mapping


@pytest.mark.parametrize("bad_string", tests["Words"]["bad"])
def test_bad_words(bad_string):
    assert not prc_words.process(bad_string)
