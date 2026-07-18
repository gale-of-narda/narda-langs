import copy
import csv
import logging
import os
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel
from rich.console import Group

from scripts import ui
from scripts.parser_dataclasses import (
    Alphabet,
    Dialect,
    Feature,
    GeneralRules,
    ParsingResult,
    SpecialRules,
    Stance,
    Symbol,
    Token,
    Type,
)
from scripts.parser_entities import Dichotomy, Element, Mapping, Mask, Tree
from scripts.util import (
    ParsingFailure,
    concat,
    parse_value,
    read_toml,
    resource_path,
)

logger = logging.getLogger(__name__)


class Processor:
    """Orchestrates all language operations."""

    def __init__(self, max_level: int | None = None, path: str = "") -> None:
        self.max_level: int | None = max_level
        self.loader: Loader = Loader(self, path)
        self.loader.reload()
        return

    def _build(self) -> None:
        """(Re)creates the rule dataclasses and parsing components from the
        loader's current raw parameters. The loader handles parameter state and
        rolls back on failure; this method only turns parameters into objects.
        """
        self.alphabet = self.loader.build_alphabet()
        self.grules = self.loader.build_grules(self.alphabet)
        self.srules = self.loader.build_srules(self.alphabet)
        self.dialect = self.loader.build_dialect(
            self.loader.load_features(), self.loader.load_types()
        )
        self.levels = range(len(self.grules.struct))
        self.mapper = Mapper(self)
        self.masker = Masker(self)
        self.interpreter = Interpreter(self)
        self.streamer = Streamer(self)
        return

    def process(
        self, instr: str, verbose: bool = False, limit_alphabet: bool = False
    ) -> ParsingResult:
        """Parses the input string, applies the parsing to the trees, and
        reports the outcome as a ParsingResult (see docs/result.md). Each
        property starts false and is set true when its condition is reached:
        intelligibility inside `feed`, grammaticality once `feed` returns
        without error, and interpretability once every tree is interpreted with
        no feature left unresolved.

        When limit_alphabet is set, a non-alphabetic character fails the parse
        the moment the stream reaches it: intelligibility is set false (the
        only case that does so). Otherwise such characters are ignored.
        """
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.info(f"Parsing '{instr}'")
        # Resetting the parameters
        self.result = ParsingResult()
        self.mapping = Mapping(self.levels)
        self.trees = [[] for _ in self.levels]
        self.masker.construct()
        # Parsing the string
        try:
            self.streamer.feed(instr, limit_alphabet)
        except ParsingFailure:
            logger.info(f"Failed to parse '{instr}'")
            self.result.grammaticality = False
            return self.result
        logger.info(f"Successfully parsed '{instr}'")
        self.result.grammaticality = True
        # Applying the obtained mapping to the trees
        interpretable = True
        for lvl in range(self.last_level + 1):
            if lvl < len(self.levels) - 1:
                elems = [e.preheader.content for e in self.mapping.elems[lvl + 1]]
            else:
                elems = [self.mapping.elems[lvl]]
            for es in elems:
                tree = Tree(self.grules.struct[lvl], lvl)
                self.trees[lvl].append(tree)
                self.interpreter.apply(es, tree)
                self.interpreter.determine_ctype(tree)
                self.interpreter.determine_ptype(tree)
                interpretable = self.interpreter.interpret(tree) and interpretable
        self.result.interpretability = interpretable
        return self.result

    def get_stances(self, lvl: int = -1) -> list[Stance]:
        """Produces the list of stances of the elements of the given level."""
        return [e.stance for e in self.mapping.elems[lvl]]

    def restore(
        self,
        lvl: int | None = None,
        num: int | None = None,
        show_neutrals: bool = False,
    ) -> str:
        """Restores the tokenized input from the saved mapping by concatenating
        the token content of its elements on the given level (the highest parsed
        level by default), separating the elements of each level and wrapping
        complexes in embedders as defined by the alphabet's guiding glyphs. With a
        number, only that single element of the level is restored. Neutral fillers
        the parser inserted are omitted unless show_neutrals is set. Returns an
        empty string when no content is saved.
        """
        if not hasattr(self, "mapping"):
            return ""
        if lvl is None:
            lvl = self.last_level
        return self.mapping.restore(
            lvl, self.alphabet.separators, self.alphabet.embedders, num, show_neutrals
        )

    @property
    def last_level(self) -> int:
        if self.max_level is not None:
            return self.max_level
        else:
            return max(self.levels)


class Loader:
    """Owns the raw parameter state: reads the parameter files from disk, lets
    callers query and modify individual parameters, and builds the alphabet,
    general and special rules, and dialect dataclasses used by the parser.

    Every mutation rebuilds the owning processor and rolls back on failure, so a
    malformed parameter never leaves the parser in a broken state.
    """

    # Standard parameter sets mapped to their file names (under PARAM_DIR).
    _FILES = {
        "alphabet": "alphabet.toml",
        "rules_general": "rules_general.toml",
        "rules_special": "rules_special.toml",
        "dialect": "dialect.toml",
    }
    PARAM_DIR = "params"

    def __init__(self, prc: Processor, path: str = "") -> None:
        self.prc = prc
        self.path = path
        # Raw parameter data keyed by standard file name; the source of truth
        # from which the dataclasses and components are (re)built.
        self.params: dict[str, dict] = {}
        return

    def reload(self) -> None:
        """Loads every parameter from the standard destination and rebuilds."""
        self.params = self.load_all_raw()
        self.prc._build()
        return

    def load(self, path: str) -> str:
        """Loads parameters from the given path. A directory becomes the new base
        destination for all parameters; a file replaces only the matching
        parameter set (its name must be a standard one). Returns a description
        of what was loaded.
        """
        if os.path.isdir(path):
            self.path = path if path.endswith(("/", "\\")) else path + os.sep
            self.reload()
            return f"all parameters from '{path}'"
        if os.path.isfile(path):
            name, data = self.read_file(path)
            self._transact(lambda: self.params.__setitem__(name, data))
            return f"the '{name}' parameters from '{path}'"
        raise ValueError(f"No such file or directory: '{path}'")

    # Parameter sets that may only be loaded from a file, never set in place.
    _LOAD_ONLY = ("alphabet", "dialect")

    def set(self, name: str, value: str) -> None:
        """Sets the parameter with the given top-level name to the given value.
        A string value is parsed as a TOML value when possible (so numbers,
        arrays and inline tables are handled), otherwise kept as a plain string.

        Alphabet and dialect parameters cannot be set; they may only be loaded
        from a file.
        """
        fname, key = self._locate(name)
        if fname in self._LOAD_ONLY:
            raise ValueError(
                f"{fname.capitalize()} parameters cannot be set, only loaded "
                f"from a file ('{key}')"
            )
        self._transact(lambda: self.params[fname].__setitem__(key, parse_value(value)))
        return

    def reset(self, name: str) -> None:
        """Reloads the named parameter from its standard destination file."""
        fname, key = self._locate(name)
        fresh = self.load_raw(fname)
        if key not in fresh:
            raise ValueError(f"'{key}' is not present in the standard '{fname}' file")
        self._transact(lambda: self.params[fname].__setitem__(key, fresh[key]))
        return

    def get(self, name: str) -> None:
        """Returns the current value of the named parameter."""
        fname, key = self._locate(name)
        return self.params[fname][key]

    def _locate(self, name: str) -> tuple[str, str]:
        """Finds which standard file holds the named top-level parameter."""
        for fname, data in self.params.items():
            if name in data:
                return fname, name
        known = sorted(k for d in self.params.values() for k in d)
        raise ValueError(f"Unknown parameter '{name}'. Known: {', '.join(known)}")

    def _transact(self, mutate: Callable) -> None:
        """Applies a parameter mutation and rebuilds, rolling the parameters back
        and re-raising if validation (now performed by the pydantic models during
        the build) or the build itself fails.
        """
        snapshot = copy.deepcopy(self.params)
        try:
            mutate()
            self.prc._build()
        except Exception:
            self.params = snapshot
            self.prc._build()
            raise
        return

    def load_raw(self, name: str) -> dict:
        """Loads the raw data of a standard parameter set from the base path."""
        rel = f"{self.PARAM_DIR}/{self._FILES[name]}"
        return read_toml(self.path + rel)

    def load_all_raw(self) -> dict[str, dict]:
        """Loads the raw data of every standard parameter set."""
        return {name: self.load_raw(name) for name in self._FILES}

    def read_file(self, path: str) -> tuple[str, dict]:
        """Loads a single parameter file by an arbitrary path. Its base name
        must match one of the standard files, otherwise an error is raised.
        """
        base = os.path.basename(path)
        name = next((n for n, f in self._FILES.items() if f == base), None)
        if name is None:
            allowed = ", ".join(self._FILES.values())
            raise ValueError(
                f"'{base}' is not a standard parameter file (expected one of: "
                f"{allowed})"
            )
        return name, read_toml(path)

    def load_features(self) -> list[Feature]:
        """Loads and validates the features describing the functions and their
        arguments from features.tsv against the Feature schema. Every feature
        belongs to a language level; its blank properties match any input.
        """
        return self.load_tsv("features.tsv", Feature)

    def load_tsv[M: BaseModel](self, filename: str, model: type[M]) -> list[M]:
        """Reads a TSV parameter file and validates every non-empty row against
        the given pydantic model, returning the validated instances. Malformed
        rows raise pydantic's ValidationError.

        This is a standalone, schema-driven validator for tabular parameters,
        the row-oriented counterpart of load_mapping; it scales to any TSV
        parameter file by supplying a matching model.
        """
        path = Path(resource_path(self.path + f"{self.PARAM_DIR}/{filename}"))
        rows: list[M] = []
        with path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                if not any(str(v).strip() for v in line.values() if v):
                    continue  # skip blank rows
                rows.append(model.model_validate(line))
        return rows

    def load_mapping[M: BaseModel](
        self, name: str, model: type[M], context: dict | None = None
    ) -> M:
        """Validates the loaded raw parameter `name` against the given pydantic
        model and returns the validated instance. Optional context supplies the
        cross-parameter data (e.g. terminal-slot count, content characters) that
        the model's validators need. Malformed data raises pydantic's
        ValidationError.

        The mapping-parameter counterpart of load_tsv. It validates
        self.params[name] (populated by load_all_raw), so the parameter stays
        transactional: set/reset/get/load keep working through self.params.
        """
        return model.model_validate(self.params[name], context=context)

    def load_types(self) -> list[Type]:
        """Loads and validates the composition and permutation types from
        types.tsv against the Type schema.
        """
        return self.load_tsv("types.tsv", Type)

    def build_alphabet(self) -> Alphabet:
        """Builds and validates the alphabet through the pydantic pipeline,
        supplying the number of language levels so the guiding groups can be
        checked.
        """
        levels = len(self.params["rules_general"]["struct"])
        return self.load_mapping("alphabet", Alphabet, {"levels": levels})

    def build_grules(self, alphabet: Alphabet) -> GeneralRules:
        """Builds and validates the general rules through the pydantic pipeline,
        supplying the prebuilt alphabet so the index-list `perms` are rebuilt into
        mask strings from the content classes.
        """
        return self.load_mapping("rules_general", GeneralRules, {"alphabet": alphabet})

    def build_srules(self, alphabet: Alphabet) -> SpecialRules:
        """Builds and validates the special rules through the pydantic pipeline,
        supplying the terminal-slot count (2 ** bottom-level dichotomies) and the
        prebuilt alphabet so the index-list `tperms`/`tneuts` are rebuilt into
        strings (membership strings and single characters) from the content
        classes.
        """
        struct = self.params["rules_general"]["struct"]
        context = {"slots": 2 ** sum(struct[0]), "alphabet": alphabet}
        return self.load_mapping("rules_special", SpecialRules, context)

    def build_dialect(self, features: list[Feature], types: list[Type]) -> Dialect:
        """Builds and validates the dialect through the pydantic pipeline,
        supplying the level/rank structure for the composition-type checks, then
        attaches the separately loaded feature and type description lists. The
        features are grouped by language level, keeping the file order within
        each level; a feature with a level outside the structure is rejected.
        """
        struct = self.params["rules_general"]["struct"]
        dialect = self.load_mapping("dialect", Dialect, {"struct": struct})
        grouped: list[list[Feature]] = [[] for _ in struct]
        for f in features:
            if not 0 <= f.lvl < len(struct):
                raise ValueError(f"Feature level {f.lvl} is out of range: {f!r}")
            grouped[f.lvl].append(f)
        dialect.features = grouped
        dialect.types = types
        return dialect


class Streamer:
    """Takes an input string and creates a stream of tokens to be parsed,
    dispatching content tokens to element parsing and guiding tokens to the
    depth and level structure operations.

    Parsing is strictly incremental: the input is consumed in a single forward
    pass, one character at a time. Only `feed` ever holds the raw string; it
    is converted to a character iterator at once, and every pipeline stage is
    a generator over the previous one. A stage may buffer only what the cursor
    has not yet emitted (pending insertions in front of it) and may consult
    state recorded while parsing earlier characters, but it can never re-read
    consumed input or operate on the string as a whole.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc = prc
        return

    def feed(self, instr: str, limit_alphabet: bool = False) -> bool:
        """Processes the stream of tokens according to their alphabetic
        parameters. Content tokens are parsed as language elements, guiding
        tokens steer the depth and level structure of the parsing. When
        limit_alphabet is set, a non-alphabetic character fails the parse as
        unintelligible instead of being dropped.
        """
        held: tuple[Token, int] | None = None
        for t in self._tokenize(iter(instr), limit_alphabet):
            if self.prc.max_level is None or t.base.level <= self.prc.max_level:
                if t.base.asubcat == "guiding":
                    self._steer(t)
                if t.base.asubcat == "content" or t.base.aclass == "wildcard":
                    held = self._pair_swapper(t, held)
        if held is not None:
            raise ParsingFailure("Swapper left unclosed by the end of input")
        # Reaching completion means every character was intelligible: the whole
        # input tokenized and steered without raising.
        self.prc.result.intelligibility = True
        self._complete()
        return True

    def _steer(self, t: Token) -> None:
        """Routes a guiding token to the structure operations: a separator
        seals its level, a popper closes and a pusher opens a complex
        embedding, each sealing the levels below first. Tokens steering a level
        beyond the parsing maximum are ignored, since its stack is not built.
        """
        lvl = t.base.level
        if lvl > self.prc.last_level:
            return
        if t.base.aclass == "separator":
            self._seal(lvl)
        elif t.is_popper(lvl) and self.prc.mapping.cur_dpt[lvl] > 0:
            if lvl > 0 and self.prc.mapping.get_interval(lvl - 1):
                self._seal(lvl - 1)
            self._close_complexes(lvl, single=True)
        elif t.is_pusher(lvl):
            if lvl > 0:
                self._seal(lvl - 1)
            self._open_complex(lvl)
        return

    def _pair_swapper(
        self, t: Token, held: tuple[Token, int] | None
    ) -> tuple[Token, int] | None:
        """Parses an unswapped content token right away. A token opening a
        swapper pair is held back and parsed after the token that closes the
        pair, which must be the next swapped token in the stream.
        """
        swapper = t.swapper
        if swapper is None:
            self._parse_content(t)
            return held
        if held is None:
            openings = self.prc.alphabet.openings
            if swapper not in openings:
                raise ParsingFailure("Closing swapper with no open pair")
            return t, openings[swapper]
        token, pair = held
        if self.prc.alphabet.closings.get(swapper) != pair:
            raise ParsingFailure("Misplaced swapper")
        self._parse_content(t)
        self._parse_content(token)
        return None

    def _parse_content(self, t: Token) -> None:
        """Parses a single content (or wildcard) token as a language element,
        accounting for early and late breakers.
        """
        self._account_breakers(t, late=False)
        self._add(Element(t, level=0))
        self._account_breakers(t, late=True)
        return

    def _tokenize(self, chars: Iterator[str], limit_alphabet: bool) -> Iterator[Token]:
        """Groups the symbolized characters of the input stream into tokens:
        a base symbol starts a token and the following modifiers of its class
        attach to it. A trailing separator token is dropped.
        """
        t: Token | None = None
        s: Symbol | None = None
        for s in self._symbolize(
            self._elongate(self._normalize(chars)), limit_alphabet
        ):
            if s.acat == "Base":
                if t is not None:
                    yield t
                t = Token(base=s)
            elif s.acat == "Modifier" and t is not None and s.aclass == t.base.aclass:
                t.modifiers.append(s)
        if t is not None and s is not None and s.aclass != "separator":
            yield t

    def _symbolize(
        self, chars: Iterator[tuple[int, str]], limit_alphabet: bool
    ) -> Iterator[Symbol]:
        """Wraps every alphabetic character into a symbol carrying its
        alphabet parameters and stream position. A non-alphabetic character is
        dropped, or, when limit_alphabet is set, fails the parse on the spot
        as unintelligible — the only case that sets intelligibility to false.
        """
        lookup = self.prc.alphabet.lookup
        for i, ch in chars:
            params = lookup.get(ch)
            if params is None:
                if limit_alphabet:
                    self.prc.result.intelligibility = False
                    raise ParsingFailure(f"Encountered a non-alphabetic character {ch}")
                continue
            yield replace(params, order=i)

    def _elongate(self, chars: Iterator[tuple[int, str]]) -> Iterator[tuple[int, str]]:
        """Replaces a content character that repeats the previous one of its
        class with the class elongator and drops the further repeats.
        """
        elongators = self.prc.alphabet.elongators
        lookup = self.prc.alphabet.lookup
        prev: str | None = None
        cache: str | None = None  # a content character replaced by an elongator
        for i, ch in chars:
            if cache is not None:
                if ch == cache:
                    continue
                cache = None
            info = lookup.get(ch)
            cls = info.aclass if info and info.asubcat == "content" else None
            if cls is not None and cls in elongators and ch == prev:
                cache = ch
                ch = elongators[cls]
            prev = cache if cache is not None else ch
            yield i, ch

    def _normalize(self, chars: Iterator[str]) -> Iterator[tuple[int, str]]:
        """NFD-decomposes and lowercases the character stream and applies the
        free substitutions, yielding every character with its stream position.
        NFD is applied per combining sequence: characters are buffered until
        the next starter arrives, so the cursor only ever expands characters
        in front of itself and never re-reads consumed input.
        """
        free = self.prc.alphabet.free
        i = 0

        def flush(buffer: str) -> Iterator[tuple[int, str]]:
            nonlocal i
            for ch in (c.lower() for c in unicodedata.normalize("NFD", buffer)):
                for sub in free.get(ch, ch):
                    yield i, sub
                    i += 1

        buffer = ""
        for ch in chars:
            if unicodedata.combining(ch) == 0 and buffer:
                yield from flush(buffer)
                buffer = ""
            buffer += ch
        if buffer:
            yield from flush(buffer)

    def _add(self, e: Element) -> None:
        """Stances the given element at the current depth of its level,
        closes and fits it, and appends it to the corresponding stack.
        """
        e.stance = Stance(depth=self.prc.mapping.cur_dpt[e.level])
        self.prc.mapper.close(e)
        self._fit(e)
        self.prc.mapping.get_stack(e.level).append(e)
        return

    def _fit(self, e: Element) -> None:
        """Passes the given element to the mapper method that determines its
        dichotomic stance.
        """
        lvl = e.level
        # Skipping levels beyond the set maximum
        if self.prc.max_level is not None and self.prc.max_level < lvl:
            return
        # Elements with lists of elements as content must have heads
        if not e.molar:
            num_lvl = lvl - 1 if lvl > e.content[0].level else lvl
            e.set_head(self.prc.grules.heads[num_lvl], fallback=True)
        self.prc.mapper.determine_stance(e)
        dpt = e.stance.depth
        logger.debug(f"[L{lvl}|D{dpt}] Fit '{e}' with stance {e.stance}")
        return

    def _account_breakers(self, t: Token, late: bool) -> None:
        """Scans the modifiers of the given token for late or early breakers
        and increases the current breaker values as needed.
        """
        lvl = t.base.level
        dpt = self.prc.mapping.cur_dpt[lvl]
        for mod in t.modifiers:
            if mod.asubcat == "breaker" and mod.quality == int(late):
                self.prc.mapping.cur_breaks[lvl][dpt] += mod.index + 1
                brk = self.prc.mapping.cur_breaks[lvl][dpt]
                logger.debug(f"[L{lvl}|D{dpt}] Breaker count increased to {brk}")
        return

    def _seal(self, lvl: int) -> None:
        """Wraps up the given level together with the levels it builds on:
        starting from the lowest level with a pending interval, closes the
        open complexes and wraps the interval of every level bottom-up, so
        that the given level continues from completed lower-level elements.
        """
        m = self.prc.mapping
        if lvl > self.prc.last_level:
            return
        floor = lvl
        while floor > 0 and m.get_interval(floor - 1):
            floor -= 1
        for level in range(floor, lvl + 1):
            if m.cur_dpt[level] > 0:
                self._close_complexes(level)
            self._wrap_interval(level)
        return

    def _wrap_interval(self, lvl: int) -> None:
        """Wraps the pending interval of elements at the given level into an
        element of the higher level, adds it to the corresponding stack, and
        starts a new interval.
        """
        interval = self.prc.mapping.get_interval(lvl)
        if interval and lvl < len(self.prc.levels) - 1:
            self._add(Element(interval, level=lvl + 1))
            self.prc.masker.construct(lvl)
            self.prc.mapping.update_interval(lvl)
        return

    def _open_complex(self, lvl: int) -> None:
        """Increases the current depth of complex embedding at the given
        level by opening a new element buffer on its stack.
        """
        m = self.prc.mapping
        m.get_stack(lvl).append([])
        m.cur_dpt[lvl] += 1
        if m.cur_dpt[lvl] >= len(m.cur_breaks[lvl]):
            m.cur_breaks[lvl].append(0)
        dpt = m.cur_dpt[lvl]
        logger.debug(f"[L{lvl}|D{dpt - 1}] Depth at level {lvl} increased to {dpt}")
        return

    def _close_complexes(self, lvl: int, single: bool = False) -> None:
        """Closes the open complex embeddings at the given level, one (single)
        or all of them: wraps the deepest pending complex into an element,
        fits it in place of its content, and decreases the current depth.
        """
        m = self.prc.mapping
        if m.cur_dpt[lvl] <= 0:
            raise ParsingFailure("Attempted to pop to a negative depth")
        while m.cur_dpt[lvl] > 0:
            # Closing only operates on the latest item in the element buffer,
            # which must also be a list of elements
            parent = content = m.elems[lvl]
            for _ in range(m.cur_dpt[lvl]):
                if not content:
                    raise ParsingFailure("Attempted to close an empty complex")
                if not isinstance(content[-1], list):
                    return
                if len(content[-1]) == 0:
                    raise ParsingFailure("Attempted to wrap an empty complex")
                parent, content = content, content[-1]

            e = Element(content, level=lvl)
            e.stance = Stance(depth=m.cur_dpt[lvl] - 1)
            self.prc.mapper.close(e)
            self.prc.masker.construct(lvl, m.cur_dpt[lvl])
            m.cur_breaks[lvl][m.cur_dpt[lvl]] = 0
            parent[-1] = e
            self._fit(e)
            m.cur_dpt[lvl] -= 1

            dpt = m.cur_dpt[lvl]
            logger.debug(f"[L{lvl}|D{dpt + 1}] Depth at level {lvl} decreased to {dpt}")
            if single:
                break
        return

    def _complete(self) -> None:
        """Wraps up the stream by closing and sealing every level bottom-up.
        Necessary in case the corresponding tokens were omitted by the end
        of the input string.
        """
        m = self.prc.mapping
        for level in range(self.prc.last_level + 1):
            if m.cur_dpt[level] > 0:
                self._close_complexes(level)
            self._wrap_interval(level)
        return


class Masker:
    """Creates, holds, and manipulates the dichotomies and masks as needed
    by the parsing procedure.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc: Processor = prc
        # Level > Depth > Rank > Dichotomy
        self.masks: list[list[list[list[Dichotomy]]]] = [
            [] for _ in range(len(self.prc.grules.struct))
        ]
        self.construct()
        return

    def construct(self, level: int | None = None, depth: int | None = None) -> None:
        """Creates the hierarchy of dichotomies loaded with mask pairs.
        If level and depth are given, create or recreate the corresponding
        particular set of dichotomies.
        """
        struct = self.prc.grules.struct
        perms = self.prc.grules.perms
        revs = self.prc.grules.revs
        dembs = self.prc.grules.dembs
        wilds = self.prc.grules.wilds
        rets = self.prc.grules.rets
        skips = self.prc.grules.skips
        splits = self.prc.grules.splits
        lembs = self.prc.grules.lembs
        tneuts = self.prc.srules.tneuts

        for lvl in range(len(struct)):
            # Skip the irrelevant and reset the masks on relevant levels
            if level is not None and level != lvl:
                continue
            if depth is None:
                self.masks[lvl] = [[]]
            else:
                while len(self.masks[lvl]) < (depth or 0) + 1:
                    self.masks[lvl].append([])
                self.masks[lvl][depth or 0] = []

            # Define which dichotomies are last on rank & which are non-binary
            dich_num = sum(struct[lvl])
            lds = [r == len(range(s)) - 1 for s in struct[lvl] for r in range(s)][::-1]
            nbs = [r != 0 for s in struct[lvl] for r in range(s)][::-1]

            ranks = []
            for d in range(dich_num):
                dichs = []
                for p in range(0, len(perms[lvl][d]), 2):
                    left_perms = perms[lvl][d][p : p + 2][0]
                    right_perms = perms[lvl][d][p : p + 2][1]
                    left_mask = Mask(left_perms, dich_num - d, p)
                    right_mask = Mask(right_perms, dich_num - d, p + 1)

                    left_mask.rev = revs[lvl][d][p : p + 2][0]
                    right_mask.rev = revs[lvl][d][p : p + 2][1]
                    left_mask.demb = dembs[lvl][d][p : p + 2][0]
                    right_mask.demb = dembs[lvl][d][p : p + 2][1]
                    left_mask.wild = wilds[lvl][d][p : p + 2][0]
                    right_mask.wild = wilds[lvl][d][p : p + 2][1]

                    # Compound embedding only for the last dichs on rank
                    rlembs, rlds = lembs[lvl][::-1][d], lds[d]
                    left_mask.lemb = rlembs[p : p + 2][0] if rlds else 0
                    right_mask.lemb = rlembs[p : p + 2][1] if rlds else 0
                    left_mask.depth = depth or 0
                    right_mask.depth = depth or 0

                    # Neutral elements only for the lowest terminal masks
                    if d == 0 and lvl == 0:
                        left_mask.tneuts = tneuts[p : p + 2][0]
                        right_mask.tneuts = tneuts[p : p + 2][1]

                    dich = Dichotomy(level=lvl, d=dich_num - d - 1, nb=nbs[d])
                    dich.terminal = d == 0
                    dich.left, dich.right = [left_mask, right_mask]
                    dich.rev = bool(min(revs[lvl][d][p : p + 2]) if d < 1 else 0)
                    dich.ret = rets[lvl][d]
                    dich.skip = skips[lvl][d]
                    dich.split = splits[lvl][d]
                    dichs.append(dich)

                ranks.append(dichs)

            self.masks[lvl][depth or 0] = ranks[::-1]

        return

    def _find_dichs(self, num_key: list[int], depth: int, lvl: int) -> list[Dichotomy]:
        """Returns dichotomies whose keys start with the given one."""
        if len(self.masks[lvl]) <= depth:
            self.construct(lvl, depth)
        dichs = [mp for r in self.masks[lvl][depth] for mp in r]
        out = []
        for dich in dichs:
            if num_key == dich.num_key[: len(num_key)]:
                out.append(dich)
        return out

    def get_dichs(self, stance: Stance, lvl: int) -> list[Dichotomy]:
        """Returns the dichotomy with the key defined by the stance as well as
        every dichotomy downstream of it.
        """
        dichs = self._find_dichs(stance.pos, stance.depth, lvl)
        if dichs:
            return dichs
        else:
            raise ValueError(f"Could not find dichotomy by stance {stance}")

    def set_dichotomy(
        self, dich: Dichotomy, comp: tuple[int | None, int], level: int
    ) -> None:
        """Records the given tuple of pos and rep to the pointed mask.
        If non-terminal, resets dichotomies downstream of the other mask.
        If rep is increased, also resets those downstream the pointed mask.
        """
        pointer = dich.pointer
        assert pointer is not None  # the pointer is set before recording a fit
        target_mask = dich.masks[pointer]
        other_mask = dich.masks[1 - pointer]
        if not dich.terminal:
            self.reset_dichotomies(level, other_mask.depth, other_mask.num_key)
            if target_mask.rep < comp[1]:
                self.reset_dichotomies(level, target_mask.depth, target_mask.num_key)
        target_mask.pos, target_mask.rep = comp
        return

    def reset_dichotomies(
        self,
        level: int,
        depth: int,
        num_key: list[int] | None = None,
        total: bool = False,
    ) -> None:
        """Sets the pointers of dichotomies with and downstream of the given key
        to None. Used to reset the masks of one branch when the pointer is set
        to the other, as well as to prepare for parsing the next element.
        """
        stance = Stance(pos=num_key or [], rep=[], depth=depth)
        dichs = self.get_dichs(stance, level)
        for dich in dichs:
            dich.pointer = None
            for mask in dich.masks:
                mask.rep = 0
                if total:
                    mask.freeze = False
        return


class Mapper:
    """Performs the parsing procedure. Splits the input string into language elements
    and produces a mapping of the elements to the dichotomic masks.

    Successful termination of the parsing procedure is the criterion of grammaticality.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc: Processor = prc
        return

    def determine_stance(self, e: Element) -> bool:
        """Produces the stance for the given element by deciding the
        dichotomies of its level one by one.
        """
        for _ in range(sum(self.prc.grules.struct[e.level])):
            dich = self.prc.masker.get_dichs(e.stance, e.level)[0]
            if not self._decide_dichotomy(e, dich):
                raise ParsingFailure(f"Failed to parse '{e}'")
        return True

    def close(self, e: Element) -> None:
        """Closes the dichotomies corresponding to the given element's
        content and applies special rules to validate it.
        """
        if e.molar:
            return
        if not self._close_dichotomies(e):
            raise ParsingFailure("Failed to close the dichotomies")

        if not self._fill_necessary_terminals(e):
            raise ParsingFailure("Failed to fill the necessary terminals")

        if not self._validate_mapping(e):
            raise ParsingFailure("Failed to validate the mapping")

        return

    def _decide_dichotomy(self, e: Element, dich: Dichotomy) -> bool:
        """Produces the decision for the given element and dichotomy, linking
        the former to either first or second mask of the latter.
        """
        lvl, dpt = e.level, e.stance.depth
        comp0, comp1 = (
            dich.masks[0].compare(e, dich.split),
            dich.masks[1].compare(e, dich.split),
        )
        # A positive breaker count skips to the second mask permanently
        if self.prc.mapping.cur_breaks[lvl][dpt] > 0:
            self.prc.mapping.cur_breaks[lvl][dpt] -= 1
            if not comp1:
                return False
            dich.masks[0].freeze = True
            fit = 1
        else:
            fit = self._choose_fit(dich, comp0, comp1)
            # If nothing fits a non-binary dich, shift its mappings and retry
            if fit is None and dich.nb:
                self._shift_nonbinary_mappings(dich, invert=True, force=True)
                comp0, comp1 = (
                    dich.masks[0].compare(e, dich.split),
                    dich.masks[1].compare(e, dich.split),
                )
                fit = self._choose_fit(dich, comp0, comp1)
            if fit is None:
                logger.warning(f"[L{lvl}|D{dpt}] Could not decide {dich} for '{e}'")
                return False

        # Attempt closure if the obtained fit flips the pointer to 1 permanently
        # Makes sense only for return-restricted, non-terminal dichs
        if dich.ret and not dich.terminal and (dich.pointer or 0) != fit:
            if not self._close_dichotomies(e, dich):
                logger.warning(f"[L{lvl}|D{dpt}] Could not close {dich}")
                return False

        self._commit_fit(e, dich, fit, comp0 if fit == 0 else comp1)
        return True

    def _choose_fit(
        self,
        dich: Dichotomy,
        comp0: tuple[int | None, int] | None,
        comp1: tuple[int | None, int] | None,
    ) -> int | None:
        """Selects the mask of the dichotomy that the given comparisons allow
        fitting to, or None if neither mask both permits and fits.
        """
        # Returning to the first mask of a pointed ret-dichotomy is forbidden;
        # skipping to the second mask is only allowed right after the first
        allow0 = dich.pointer != 1 or not dich.ret
        allow1 = dich.pointer == 0 or not dich.skip
        if allow0 and comp0 and allow1 and comp1:
            # Choose the first mask unless it is the only one increasing rep
            if comp1[1] == dich.masks[1].rep and comp0[1] > dich.masks[0].rep:
                return 1
            return 0
        if allow0 and comp0:
            return 0
        if allow1 and comp1:
            return 1
        return None

    def _commit_fit(
        self,
        e: Element,
        dich: Dichotomy,
        fit: int,
        comp: tuple[int | None, int] | None,
    ) -> None:
        """Records the fit: points the dichotomy at the chosen mask, applies
        the comparison movement to it, and appends the resulting pos and rep
        digits to the element's stance.
        """
        dich.pointer = fit
        old_mask = f"{dich.masks[fit]}"

        assert comp is not None  # the chosen mask had a fit
        self.prc.masker.set_dichotomy(dich, comp, e.level)
        pos = fit if not dich.rev else 1 - fit
        rep = dich.masks[fit].rep

        new_mask = f"{dich.masks[fit]}"
        num_strings = ["1st", "2nd"]
        lvl, dpt = e.level, e.stance.depth
        content = f"[L{lvl}|D{dpt}] Fitting '{e}' to the {num_strings[fit]}"
        if dich.split:
            logger.debug(f"{content} mask {new_mask}")
        else:
            logger.debug(f"{content} mask {old_mask} → {new_mask}")

        e.stance.pos.append(pos)
        e.stance.rep.append(rep)
        return

    def _close_dichotomies(self, e: Element, dich: Dichotomy | None = None) -> bool:
        """For dichotomies downstream of the given dichotomy, performs
        the finalizing operations: shift the mappings for the non-binary ones,
        add neutral elements as needed for the terminal ones.

        If no dichotomy is given, starts from the last fitted branch
        of the topmost dichotomy for the element's content.
        """
        if dich is None:
            content = e.content
            level, depth = content[0].level, content[0].stance.depth
            dich = self.prc.masker.get_dichs(Stance(depth=depth), level)[0]
        else:
            level, depth = dich.level, dich.depth
        res = True
        stance = Stance(pos=dich.masks[dich.pointer or 0].num_key, depth=depth)
        dichs = self.prc.masker.get_dichs(stance, level)
        invert = bool(dich.pointer or 0)

        for cdich in dichs:
            if cdich.nb:
                res = res and self._shift_nonbinary_mappings(cdich, invert=invert)
            # Only the bottom level fills empty terminals with neutrals (its
            # necessary ones, and those with an occupied sibling); higher levels
            # enforce necessity by rejection in `_fill_necessary_terminals`.
            if cdich.terminal and cdich.level == 0:
                res = res and self._fill_necessary(e, cdich)
                res = res and self._fill_siblings(e, cdich)

        return res

    def _shift_nonbinary_mappings(
        self, dich: Dichotomy, invert: bool = False, force: bool = False
    ) -> bool:
        """Shifts the mappings of the elements in the current stack from the
        first to the second mask of the dichotomy (or vice versa if invert is
        True) continuously slot by slot as long as the shift produces a valid
        mapping.

        A shift is only possible if the target mask has an empty slot.
        Equivalent mappings are shifted together or not at all, compounds
        from closest to farthest to the target mask, and only if they fit
        (given lembs and perms).
        """
        level = dich.level
        mapping = self.prc.mapping
        elems = mapping.get_stack(level, interval=True)

        mask_from, mask_to = dich.masks if not invert else dich.masks[::-1]
        matches_from = mapping.enumerate_elems(mask_from.num_key, elems, dich.d)
        matches_to = mapping.enumerate_elems(mask_to.num_key, elems, dich.d)

        # Skip the shift to a non-empty mask unless forced
        num_occupied = len(matches_to)
        if not force and num_occupied > 0:
            return True

        # Setting the pointer to activate the target mask
        dich.pointer = 1 - int(invert)

        # Perform the shift slot by slot
        slots_shifted, elems_shifted = 0, 0
        num = mask_to.lemb + 1 - num_occupied
        items = reversed(matches_from) if not dich.rev else matches_from
        for n, slot in enumerate(items):
            slot_in_process = False
            if num > 0:
                for i in matches_from[slot]:
                    if self._shift_element(elems[i], dich, mask_to, n):
                        slot_in_process = True
                        elems_shifted += 1
                    elif slot_in_process:
                        # A slot moves whole: a member failing after another
                        # member has moved fails the shift
                        logger.warning(f"=> No place to shift {elems[i]} along {dich}")
                        return False
                    else:
                        # The first member of a slot failing ends the shift
                        return True
                num -= 1
            slots_shifted += 1

        # Reset the dichotomies downstream of the mask &
        # subtract the shifted elements from the mask
        if elems_shifted > 0:
            self.prc.masker.reset_dichotomies(level, mask_from.depth, mask_from.num_key)
            mask_from.subtract(elems_shifted, slots_shifted)

        return True

    def _shift_element(self, e: Element, dich: Dichotomy, mask: Mask, n: int) -> bool:
        """Refits the element into the n-th slot of the given mask, flipping
        the pos digit of its stance at the dichotomy.
        """
        old_stance = str(e.stance)
        new_stance = e.stance.copy()
        new_stance.pos[dich.d] = mask.num_key[-1]
        new_stance.rep[dich.d] = mask.rep + n
        if not self._fit_element(e, new_stance, dich.d):
            return False
        dpt = e.stance.depth
        logger.debug(
            f"[L{dich.level}|D{dpt}] Shifted {e} from {old_stance} to {new_stance}"
        )
        return True

    def _fill_necessary_terminals(self, e: Element) -> bool:
        """Rejects the parse if a necessary terminal is left empty in any branch
        of the tree, not only the one the content entered. Runs once at close,
        above the bottom level only: those levels configure no neutral fillers,
        so a missing necessary element cannot be repaired. The bottom level
        fills its necessary terminals in place during closure (`_fill_necessary`).
        """
        content = e.content
        level, depth = content[0].level, content[0].stance.depth
        if level == 0:
            return True
        for dich in self.prc.masker.get_dichs(Stance(depth=depth), level):
            if not dich.terminal:
                continue
            _, _, hits = self._enumerate_sides(dich)
            for mask, occupied in zip(dich.masks, hits, strict=True):
                if mask.necessities[0] and not occupied:
                    raise ParsingFailure(
                        f"Empty necessary terminal {mask} on level {level}"
                    )
        return True

    def _enumerate_sides(
        self, dich: Dichotomy
    ) -> tuple[list[Element], list[list[int]], list[dict[str, list[int]]]]:
        """Returns the current interval elements together with the num keys
        of the dichotomy's masks and the elements enumerated under each key.
        """
        elems = self.prc.mapping.get_stack(dich.level, interval=True)
        keys = [dich.masks[i].num_key for i in range(2)]
        hits = [self.prc.mapping.enumerate_elems(k, elems, dich.d) for k in keys]
        return elems, keys, hits

    def _fill_necessary(self, e: Element, dich: Dichotomy) -> bool:
        """Inserts a neutral element into an empty necessary terminal of the
        given bottom-level dichotomy, placed after the elements preceding the
        branch.
        """
        elems, keys, hits = self._enumerate_sides(dich)
        to_fill = []
        for i, mask in enumerate(dich.masks):
            if mask.necessities[0] and not hits[i]:
                prs = self.prc.mapping.enumerate_elems(keys[i], elems, dich.d, True)
                if prs:
                    last_index = [idx for slot in prs.values() for idx in slot][-1]
                    op_stance = Stance(
                        pos=keys[1 - i],
                        rep=[0] * len(keys[1 - i]),
                        depth=mask.depth,
                    )
                    to_fill.append(([last_index], mask, op_stance))
        return self._neutralize(e, to_fill, dich.level, dich.depth, compensate=False)

    def _fill_siblings(self, e: Element, dich: Dichotomy) -> bool:
        """Inserts a neutral element next to every occupied slot of the given
        dichotomy whose sibling slot on the other mask is empty.
        """
        elems, _, hits = self._enumerate_sides(dich)
        if hits[0] and not hits[1]:
            occupied, empty = (0, 1)
        elif hits[1] and not hits[0]:
            occupied, empty = (1, 0)
        else:
            return True
        to_fill = []
        for slot in hits[occupied]:
            slot_hits = hits[occupied][slot]
            slot_mask = dich.masks[empty]
            slot_stances = elems[hits[occupied][slot][0]].stance.copy()
            to_fill.append((slot_hits, slot_mask, slot_stances))
        return self._neutralize(e, to_fill, dich.level, dich.depth, compensate=True)

    def _neutralize(
        self,
        e: Element,
        to_fill: list[tuple[list[int], Mask, Stance]],
        lvl: int,
        dpt: int,
        compensate: bool,
    ) -> bool:
        """Creates and fits a neutral element for every given address (the
        target stance is derived by flipping the given sibling stance) and
        splices each fitted neutral into the stream order, the wrap's content,
        and the element stack.
        """
        elems = self.prc.mapping.get_stack(lvl, interval=True)
        for indices, neut_mask, op_stance in to_fill:
            # Masks without a configured neutral for this depth (any mask
            # outside the lowest level-0 terminals, or a depth beyond the
            # configured list) cannot be filled; skip them rather than crash.
            if (
                not neut_mask.tneuts
                or dpt >= len(neut_mask.tneuts)
                or not neut_mask.tneuts[dpt]
            ):
                continue
            op_stance.pos[-1] = 1 - op_stance.pos[-1]
            t = self.prc.alphabet.get_token(neut_mask.tneuts[dpt])
            neut = Element(t, op_stance, lvl)
            neut.neutral = True
            if not self._fit_element(neut, op_stance, term_only=True):
                logger.warning(f"[L{lvl}|D{dpt}] Could not fit {neut} to {op_stance}")
                return False
            logger.debug(f"[L{lvl}|D{dpt}] Inserting {neut} with stance {op_stance}")

            if compensate:
                slot = op_stance.pos[-1] if not neut_mask.rev else 1 - op_stance.pos[-1]
                index = min(indices) if slot == 0 else max(indices)
            else:
                slot, index = 1, max(indices)
            neut.head.tok.base.order = elems[index].head.tok.base.order + (
                slot - 1 if compensate else 1
            )
            if not e.molar and e.content[0].level != e.level:
                e.content.insert(index + slot, neut)
            stack = self.prc.mapping.get_stack(lvl)
            stack.insert(self.prc.mapping.cur_bdr[lvl] + index + slot, neut)

        return True

    def _fit_element(
        self,
        e: Element,
        stance: Stance | None = None,
        d: int | None = None,
        term_only: bool = False,
        force_mov: bool = False,
    ) -> bool:
        """Records the element if it can be fit with the given stance."""
        if stance is None:
            stance = e.stance
        for p, pos in enumerate(stance.pos):
            if (term_only and p != len(stance.pos) - 1) or (d is not None and p < d):
                continue
            part_stance = stance.copy(p)
            dich = self.prc.masker.get_dichs(part_stance, e.level)[0]
            cur_mask = pos if not dich.rev else 1 - pos

            comp = dich.masks[cur_mask].compare(e, dich.split, force_mov)
            if comp:
                e.stance = stance
                dich.pointer = cur_mask
                self.prc.masker.set_dichotomy(dich, comp, e.level)
            else:
                return False
        return True

    def _validate_mapping(self, e: Element) -> bool:
        """Checks that every stance in the given element's content complies
        with terminal permissions.
        """
        level = 0
        elems = e.content
        cnt = 0
        addr = None
        for el in elems:
            if el.level != 0:
                continue
            cnt = cnt + 1 if el.stance.pos + el.stance.rep[:-1] == addr else 0
            addr = el.stance.pos + el.stance.rep[:-1]

            rev = bool(self.prc.grules.revs[level][0][el.num])
            addrs = [x for x in elems if x.stance.pos + x.stance.rep[:-1] == addr]

            perms = self.prc.srules.tperms[el.num]
            priority = cnt if not rev else len(addrs) - 1 - cnt
            depth_perms = perms[min(el.stance.depth, len(perms) - 1)]
            # A priority beyond the permission list means more elements landed
            # at this address than the slot licenses: the element is unpermitted.
            if priority >= len(depth_perms):
                logger.warning(
                    f"-> No permission slot for priority {priority} at {el.stance}"
                )
                return False
            perm = depth_perms[priority]

            aclass = el.head.tok.base.aclass
            base = str(el.head.tok.base)
            if not any([base in perm or aclass in perm, aclass == "wildcard"]):
                logger.warning(
                    f"-> No permission for '{el.head}'/'{aclass}' at {el.stance}"
                )
                return False

        return True


class Interpreter:
    """Commits the given mapping to the dichotomic tree and provides dialectic
    interpretations to the functions of its nodes with the received arguments.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc = prc
        return

    def apply(self, elems: list[Element], tree: Tree) -> None:
        """Applies the given elements to the given tree using their stances.
        Embeds complexes and compounds in the tree as needed.
        """
        finals = [r + 1 == s for s in tree.struct for r in range(s)]
        # Iterating the elements and setting each
        for e in elems:
            # Accounting for reps
            for j in range(len(e.stance.pos)):
                if finals[j]:
                    base_stance = e.stance.copy(j + 1)
                    base_stance.rep[-1] = 0
                    base_node = tree.get_nodes(base_stance)[-1]
                    if e.stance.rep[j] > len(base_node.compounds):
                        for _ in range(e.stance.rep[j] - len(base_node.compounds)):
                            tree.embed_compound(base_node)

            # Accounting for depth
            if not e.molar and e.content[0].level == e.level:
                base_node = tree.get_nodes(e.stance)[-1]
                tree.embed_complex(base_node)
                self.apply(e.content, base_node.complexes[-1])

            tree.set_element(e)

        return

    def determine_ctype(self, tree: Tree) -> None:
        """Determines the composition type of the element recorded in the tree."""
        population = tree._collect_population()
        # Try different types one by one
        # Types specific for the depth level go first, general types last
        ctypes = self.prc.dialect.ctypes[tree.level]
        if not ctypes:
            ctypes = []
        elif str(tree.depth) in ctypes:
            ctypes = ctypes[str(tree.depth)] or ctypes[""]
        else:
            ctypes = ctypes[""]
        fits = []
        for ctype in ctypes:
            fit = True
            # If the type relates to trees embedded in nodes with the defined stance,
            # first exclude it if the given tree stance does not match it
            if "cpos" in ctypes[ctype]:
                if tree.stance is None or tree.stance.pos != ctypes[ctype]["cpos"]:
                    fit = False
            if "crep" in ctypes[ctype]:
                if tree.stance is None or tree.stance.rep != ctypes[ctype]["crep"]:
                    fit = False
            # Every type has conditions on how many nodes of each rank are populated
            for r, rank in enumerate(ctypes[ctype]["ranks"]):
                for i, perm in enumerate(rank):
                    num = population[r].get(i, 0)
                    conds = [
                        perm not in ("*", "+", "-") and num > int(perm),
                        perm not in ("*", "+", "-") and num == 0,
                        perm == "+" and num == 0,
                        perm == "-" and num != 0,
                    ]
                    if any(conds):
                        fit = False
            if fit:
                fits.append(ctype)

        # Choose the first type that fits
        if fits:
            tree.ctype = fits[0]
        else:
            logger.warning(f"Undefined composition type at level {tree.level}")

        # Do the same for all embedded trees
        for c in tree.complexes:
            self.determine_ctype(c)

        return

    def determine_ptype(self, tree: Tree) -> None:
        """Determines the permutation type of the elements recorded in the tree
        from the presence of swappers on its terminal molar tokens. Illegal
        swapper sequences are already rejected by the streamer, so the swapper
        node indexes are simply matched against the dialect's permutation types.
        """
        swaps = tree._collect_swaps()
        fits = [
            name
            for entry in self.prc.dialect.ptypes
            for name, pdef in entry.items()
            if {i for pair in pdef["swaps"] for i in pair} == swaps
        ]
        # The last fitting type wins (general types are listed first)
        if fits:
            tree.ptype = fits[-1]
        else:
            logger.warning(f"Undefined permutation type at level {tree.level}")

        # Do the same for all embedded trees
        for c in tree.complexes:
            self.determine_ptype(c)

        return

    def interpret(self, tree: Tree) -> bool:
        """Inscribes interpretations defined in the dialect to the tree nodes
        depending on their content, then interprets the embedded trees. Returns
        whether every interpretable node resolved to a feature; wildcards are
        exempt and never counted, since no feature describes them.
        """
        resolved = True
        # Find features for the tree nodes and their compounds
        nodes = [n for n in tree.all_nodes if n.terminal and n.content]
        for node in nodes:
            if not node.content:
                continue
            e = node.content[0]
            if e.header.tok.base.aclass == "wildcard":
                continue
            feature = self.prc.dialect.get_feature(
                tree.level,
                e.header.tok.base.index,
                e.header.tok.base.aclass,
                node.stance,
                tree.stance,
                tree.ctype,
            )
            if not feature:
                resolved = False
                continue
            node.feature = feature
        # Interpret the embedded trees
        for c in tree.complexes:
            resolved = self.interpret(c) and resolved
        return resolved

    def describe(
        self,
        tree: Tree | int | None = None,
        verbose: bool = False,
        rich: bool = False,
        _prefix: str = "·",
        _depth: int = 0,
    ) -> object | None:
        """Summarises the interpreted features of the tree.

        If rich is True, returns a Rich Group of Tables.
        Otherwise logs the summary via the logger and returns None.
        """
        if tree is None:
            tree = self.prc.trees[0][0]
        elif isinstance(tree, int):
            tree = self.prc.trees[0][tree]

        nodes = tree.get_interpretable_nodes()
        featureless = [n for n in nodes if not n.feature]
        featured = [n for n in nodes if n.feature]

        if rich:
            table = ui.feature_table(featured, featureless, verbose)
            renderables: list = [
                ui.type_table(tree, self.prc.dialect.types, verbose),
                table,
            ]
            for c in tree.complexes:
                renderables.append(
                    self.describe(c, verbose=verbose, rich=True, _depth=_depth + 1)
                )
            return Group(*renderables)
        else:
            logger.info(f"{_prefix} {tree.ptype} {tree.ctype} '{tree.working_string}'")
            for node in featured:
                feature = node.feature
                if feature is None:
                    continue
                msg = "%s> '%s' — %s: %s"
                args = [
                    _prefix,
                    node.content[0],
                    feature.function_name,
                    feature.argument_name,
                ]
                if verbose:
                    msg += " — %s"
                    args.append(feature.argument_description)
                logger.info(msg, *args)
            if featureless:
                logger.info(
                    f"{_prefix}>> Features lacking interpretation: "
                    f"{', '.join(str(n) for n in featureless)}"
                )
            for c in tree.complexes:
                self.describe(c, verbose=verbose, rich=False, _prefix=_prefix + "·")

    def draw_tree(
        self,
        tree: Tree | int | None = None,
        features: bool = False,
        all_nodes: bool = False,
    ) -> str:
        """Prints out the given dichotomic tree with mapped elements."""
        if tree is None:
            level = self.prc.trees[0]
        elif isinstance(tree, int):
            level = self.prc.trees[tree]

        if len(level) == 0:
            return "No tree to draw"
        else:
            tree = level[-1]

        st = str(tree)

        if tree.stance:
            st += f" at {tree.stance}"
        st += "\n"
        st += tree.draw(features=features, all_nodes=all_nodes)

        return st

    def gloss(
        self,
        tree: Tree | int | None = None,
        verbose: bool = False,
    ) -> Group:
        """Iterates the terminal nodes of the tree and replaces the representations
        of their contents with the gloss strings defined by their features. The
        gloss table is preceded by a table of the tree's composition and
        permutation types.
        """
        if tree is None:
            tree = self.prc.trees[0][-1]
        elif isinstance(tree, int):
            tree = self.prc.trees[0][tree]

        items = tree.get_interpretable_nodes(complexes=True)
        tokens = self._build_gloss_tokens(items)

        return Group(
            ui.type_table(tree, self.prc.dialect.types, verbose),
            ui.gloss_table(tokens),
        )

    def _build_gloss_tokens(self, items: list) -> list[tuple[str, str]]:
        """Converts a flat list of nodes (and nested lists for complexes) into
        a list of (form, gloss) pairs.
        """
        tokens: list[tuple[str, str]] = []
        form_parts: list[str] = []
        gloss_parts: list[str] = []

        def flush() -> None:
            if form_parts:
                tokens.append(("-".join(form_parts), "-".join(gloss_parts)))
                form_parts.clear()
                gloss_parts.clear()

        for item in items:
            if isinstance(item, list):
                flush()
                sub_tokens = self._build_gloss_tokens(item)
                sub_form = "-".join(f for f, _ in sub_tokens)
                sub_gloss = "-".join(g for _, g in sub_tokens)
                tokens.append((f"[{sub_form}]", f"[{sub_gloss}]"))
            else:
                form = concat(item.content)
                gloss = (
                    item.feature.argument_gloss
                    if item.feature and item.feature.argument_gloss
                    else form
                )
                # An item carrying its own gloss starts a fresh token group;
                # otherwise it is hyphenated into the current one.
                if item.feature and item.feature.argument_gloss:
                    flush()
                form_parts.append(form)
                gloss_parts.append(gloss)

        flush()
        return tokens
