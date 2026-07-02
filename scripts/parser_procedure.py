import copy
import csv
import logging
import os
import unicodedata
from collections.abc import Callable, Generator
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
    SpecialRules,
    Stance,
    Token,
    Type,
)
from scripts.parser_entities import Dichotomy, Element, Mapping, Mask, Tree
from scripts.util import (
    ParsingFailure,
    concat,
    parse_value,
    read_yaml,
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
        self.grules = self.loader.build_grules()
        self.srules = self.loader.build_srules()
        self.dialect = self.loader.build_dialect(
            self.loader.load_features(), self.loader.load_types()
        )
        self.levels = range(len(self.grules.struct))
        self.mapper = Mapper(self)
        self.masker = Masker(self)
        self.interpreter = Interpreter(self)
        self.streamer = Streamer(self)
        return

    def process(self, instr: str, verbose: bool = False) -> bool:
        """Parses the input string and applies the parsing to the trees."""
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.info(f"Parsing '{instr}'")
        # Resetting the parameters
        self.mapping = Mapping(self.levels)
        self.trees = [[] for _ in self.levels]
        self.streamer.instr = instr
        self.masker.construct()
        # Parsing the string
        try:
            self.streamer.feed()
        except ParsingFailure:
            logger.info(f"Failed to parse '{instr}'")
            return False
        logger.info(f"Successfully parsed '{instr}'")
        # Applying the obtained mapping to the trees
        for lvl in self.levels:
            if lvl < len(self.levels) - 1:
                elems = [e.preheader.content for e in self.mapping.elems[lvl + 1]]
            else:
                elems = [self.mapping.elems[lvl]]
            for es in elems:
                tree = Tree(self.grules.struct[lvl], lvl)
                self.trees[lvl].append(tree)
                self.interpreter.apply(es, tree)
                if self.max_level is None or self.max_level >= lvl:
                    self.interpreter.determine_ctype(tree)
                    self.interpreter.determine_ptype(tree)
                    self.interpreter.interpret(tree)
        return True

    def get_stances(self, lvl: int = -1) -> list[Stance]:
        """Produces the list of stances of the elements of the given level."""
        return [e.stance for e in self.mapping.elems[lvl]]

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
        "alphabet": "alphabet.yaml",
        "rules_general": "rules_general.yaml",
        "rules_special": "rules_special.yaml",
        "dialect": "dialect.yaml",
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
        A string value is parsed as YAML when possible (so numbers, lists and
        mappings are handled), otherwise kept as a plain string.

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
        return read_yaml(self.path + rel)

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
        return name, read_yaml(path)

    def load_features(self) -> list[Feature]:
        """Loads and validates the features describing the functions and their
        arguments from features.tsv against the Feature schema. A feature with a
        content type is bound to it; the rest are untyped.
        """
        return self.load_tsv("features.tsv", Feature)

    def load_tsv[M: BaseModel](self, filename: str, model: type[M]) -> list[M]:
        """Reads a TSV parameter file and validates every non-empty row against
        the given pydantic model, returning the validated instances. Malformed
        rows raise pydantic's ValidationError.

        This is a standalone, schema-driven validator for tabular parameters,
        independent of the YAML shape checks in _validate; it scales to any TSV
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

    def load_yaml[M: BaseModel](
        self, name: str, model: type[M], context: dict | None = None
    ) -> M:
        """Validates the loaded raw parameter `name` against the given pydantic
        model and returns the validated instance. Optional context supplies the
        cross-parameter data (e.g. terminal-slot count, content characters) that
        the model's validators need. Malformed data raises pydantic's
        ValidationError.

        The mapping-parameter counterpart of load_tsv and the pydantic
        replacement for the _validate_* shape checks. It validates
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
        return self.load_yaml("alphabet", Alphabet, {"levels": levels})

    def build_grules(self) -> GeneralRules:
        """Builds and validates the general rules through the pydantic pipeline.
        The rules validate against their own structure, so no context is needed.
        """
        return self.load_yaml("rules_general", GeneralRules)

    def build_srules(self) -> SpecialRules:
        """Builds and validates the special rules through the pydantic pipeline,
        supplying the terminal-slot count (2 ** bottom-level dichotomies) and the
        alphabet's content characters so SpecialRules can enforce the
        cross-parameter content checks.
        """
        struct = self.params["rules_general"]["struct"]
        content = self.params["alphabet"]["bases"]["content"]
        chars = {c for group in content.values() for s in group for c in s}
        context = {"slots": 2 ** sum(struct[0]), "content_chars": chars}
        return self.load_yaml("rules_special", SpecialRules, context)

    def build_dialect(self, features: list[Feature], types: list[Type]) -> Dialect:
        """Builds and validates the dialect through the pydantic pipeline,
        supplying the level/rank structure for the composition-type checks, then
        attaches the separately loaded feature and type description lists.
        """
        struct = self.params["rules_general"]["struct"]
        dialect = self.load_yaml("dialect", Dialect, {"struct": struct})
        dialect.features = features
        dialect.types = types
        return dialect


class Streamer:
    """Takes an input string and creates a stream of tokens wrapped into
    elements to be parsed, depth and level structure accounted.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc = prc
        self.instr: str = ""
        self.e: Element
        self.t: Token
        return

    def _tokenize(self) -> Generator:
        """Performs the splitting, alphabetic filtering, symbolization,
        and tokenization of the input string.
        """
        t: Token | None = None
        prev: str | None = None  # previous character, for detecting repeats
        cache: str | None = None  # a content character replaced by an elongator
        # Precomposed characters are decomposed (NFC)
        linstr = [ch.lower() for ch in unicodedata.normalize("NFD", self.instr)]
        for i, ch in enumerate(linstr):
            # Replacing a character with its free substitution
            free = self.prc.alphabet.free
            if ch in free:
                sub = free[ch]
                # If the replacement consists of more than one symbol,
                # add the additional symbols to the string
                for j, extra in enumerate(sub[1:]):
                    linstr.insert(i + j + 1, extra)
                ch = sub[0]
            # Elongating: a content character repeating the previous one of its
            # class is replaced by its elongator. While the same character keeps
            # repeating it is dropped; any other character clears the cache.
            elongators = self.prc.alphabet.elongators
            info = self.prc.alphabet.lookup.get(ch)
            cls = info.aclass if info and info.asubcat == "content" else None
            if cache is not None:
                if ch == cache:
                    continue
                cache = None
            if cls is not None and cls in elongators and ch == prev:
                cache = ch
                ch = elongators[cls]
            prev = cache if cache is not None else ch
            # Wrapping the char into a symbol if it is alphabetic
            if ch not in self.prc.alphabet.lookup:
                continue
            params = self.prc.alphabet.lookup[ch]
            s = replace(params, order=i)
            # Wrapping the base symbol into a token (if base)
            # or modifying the previous one (if modifier)
            # The token is yielded as the next base symbol is read
            # or at the end of the string
            if isinstance(t, Token):
                if s.acat == "Base":
                    yield t
                elif s.acat == "Modifier" and s.aclass == t.base.aclass:
                    t.modifiers.append(s)
            if s.acat == "Base":
                t = Token(base=s)

        if isinstance(t, Token) and s.aclass != "separator":
            yield t

    def feed(self) -> bool:
        """Processes the stream of tokens according to their alphabetic parameters.
        Content tokens are parsed as language elements, others set the parameters
        of the processor.
        """
        stream = self._tokenize()
        openings = self.prc.alphabet.openings
        closings = self.prc.alphabet.closings
        cache: tuple[Token, int] | None = None
        # Exhausting the stream
        while True:
            t = next(stream, None)
            # When the last token is reached, do the closing operations
            if t is None:
                if cache is not None:
                    raise ParsingFailure("Swapper left unclosed by the end of input")
                self.complete()
                return True
            self.t = t
            lvl = t.base.level
            if t.base.asubcat == "guiding":
                # Separating intervals of elements
                if t.base.aclass == "separator":
                    self.separate(lvl)
                # Decreasing depth of complex embedding
                elif t.is_popper(lvl) and self.prc.mapping.cur_dpt[t.base.level] > 0:
                    self.pop(lvl)
                # Increasing depth of complex embedding
                elif t.is_pusher(lvl):
                    self.push(lvl)
            # Parsing content tokens while accounting for early & late breakers
            if t.base.asubcat == "content" or t.base.aclass == "wildcard":
                swapper = t.swapper
                if swapper is None:
                    self._parse_content(t)
                elif cache is None:
                    # Nothing held: the swapper must open a pair
                    if swapper not in openings:
                        raise ParsingFailure("Closing swapper with no open pair")
                    cache = (t, openings[swapper])
                else:
                    # A token is held: only its closing complement may follow
                    held, pair = cache
                    if closings.get(swapper) != pair:
                        raise ParsingFailure("Misplaced swapper")
                    self._parse_content(t)
                    self._parse_content(held)
                    cache = None

    def _parse_content(self, t: Token) -> None:
        """Parses a single content (or wildcard) token as a language element,
        accounting for early and late breakers.
        """
        self.t = t
        self.account_breaker(late=False)
        self.e = Element(t, level=0)
        self.add()
        self.account_breaker(late=True)
        return

    def add(self) -> None:
        """Adds the current element to stack, parses and closes it."""
        lvl = self.e.level
        dpt = self.prc.mapping.cur_dpt[self.e.level]
        self.e.stance = Stance(depth=dpt)
        self.prc.mapper.close(self.e)
        self.parse()
        self.prc.mapping.get_stack(lvl).append(self.e)
        return

    def separate(self, lvl: int, deep: bool = False) -> None:
        """Wraps the interval of elements limited by the current border position
        into an element of the higher level and adds it to corresponding stack.
        """
        for level in self.prc.levels:
            if (level == lvl or deep) and level <= self.prc.last_level:
                # If the current depth of embedding is above zero on the same level,
                # pop until it reaches zero
                if level > 0:
                    self.pop(level, deep=True)
                interval = self.prc.mapping.get_interval(level)
                if interval and level < len(self.prc.levels) - 1:
                    self.e = Element(interval, level=level + 1)
                    self.add()
                    self.prc.masker.construct(level)
                    self.prc.mapping.update_interval(level)
        return

    def pop(self, lvl: int, deep: bool = False) -> None:
        """Decreases the current depth of complex embedding, wraps the current
        stack interval into an element of the same level and parses it.
        """
        # Perform the separation on the previous level if it hasn't already been
        if lvl > 0 and self.prc.mapping.get_interval(lvl - 1):
            self.separate(lvl - 1)
        while self.prc.mapping.cur_dpt[lvl] > 0:
            # Popping only operates on the latest item in the element buffer,
            # which must also be a list of elements
            content = self.prc.mapping.elems[lvl]
            for _ in range(self.prc.mapping.cur_dpt[lvl]):
                if not isinstance(content[-1], list):
                    return
                elif len(content[-1]) == 0:
                    del content[-1]
                    return
                content = content[-1]

            e = Element(content, level=lvl)
            e.stance = Stance(depth=self.prc.mapping.cur_dpt[lvl] - 1)
            self.e = e
            self.prc.mapper.close(e)
            self.prc.masker.construct(lvl, self.prc.mapping.cur_dpt[lvl])
            self.prc.mapping.cur_breaks[lvl][self.prc.mapping.cur_dpt[lvl]] = 0

            target = self.prc.mapping.elems[lvl]
            for _ in range(self.prc.mapping.cur_dpt[lvl] - 1):
                target = target[-1]
            target[-1] = e

            self.parse()
            self.prc.mapping.cur_dpt[lvl] -= 1

            dpt = self.prc.mapping.cur_dpt[lvl]
            logger.debug(f"[L{lvl}|D{dpt + 1}] Depth at level {lvl} decreased to {dpt}")

            if not deep:
                break

        return

    def push(self, lvl: int) -> None:
        """Increases the current depth of complex embedding.
        Performs separation on the previous level if necessary.
        """
        if lvl > 0:
            self.separate(lvl - 1)
        self.prc.mapping.get_stack(lvl).append([])
        self.prc.mapping.cur_dpt[lvl] += 1
        if self.prc.mapping.cur_dpt[lvl] >= len(self.prc.mapping.cur_breaks[lvl]):
            self.prc.mapping.cur_breaks[lvl].append(0)
        dpt = self.prc.mapping.cur_dpt[lvl]
        logger.debug(f"[L{lvl}|D{dpt - 1}] Depth at level {lvl} increased to {dpt}")
        return

    def complete(self) -> None:
        """Wraps up the stream by performing final popping and separation.
        Necessary in case the corresponding tokens were omitted by the end
        of the input string.
        """
        for level in self.prc.levels:
            if level <= self.prc.last_level:
                self.pop(level, deep=True)
        self.separate(self.prc.last_level, deep=True)
        return

    def account_breaker(self, late: bool) -> None:
        """Scans the modifiers of the current token for late or early breakers
        and increases the current breaker values as needed.
        """
        lvl = self.t.base.level
        dpt = self.prc.mapping.cur_dpt[lvl]
        for mod in self.t.modifiers:
            if mod.asubcat == "breaker" and mod.quality == int(late):
                self.prc.mapping.cur_breaks[lvl][dpt] += mod.index + 1
                brk = self.prc.mapping.cur_breaks[lvl][dpt]
                logger.debug(f"[L{lvl}|D{dpt}] Breaker count increased to {brk}")
        return

    def parse(self) -> None:
        """Passes the current element to the mapper method that determines
        its dichotomic stance.
        """
        lvl = self.e.level
        # Skipping levels beyond the set maximum
        if self.prc.max_level is not None and self.prc.max_level < lvl:
            return
        # Elements with lists of elements as content must have heads
        if not self.e.molar:
            num_lvl = lvl - 1 if lvl > self.e.content[0].level else lvl
            self.e.set_head(self.prc.grules.heads[num_lvl], fallback=True)
        self.prc.mapper.determine_stance(self.e)
        dpt = self.e.stance.depth
        logger.debug(f"[L{lvl}|D{dpt}] Fit '{self.e}' with stance {self.e.stance}")
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
        self.e: Element
        return

    def _decide_dichotomy(self, dich: Dichotomy, forbid_shift: bool = False) -> bool:
        """Produces the decision for the current element and dichotomy, linking
        the former to either first or second mask of the latter.
        """
        lvl, dpt = self.e.level, self.e.stance.depth
        P = f"[L{lvl}|D{dpt}]"
        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any((dich.pointer != 1, not dich.ret)),
            any((dich.pointer == 0, not dich.skip)),
        ]
        # Results of fit for the masks
        comp0, comp1 = (
            dich.masks[0].compare(self.e, dich.split),
            dich.masks[1].compare(self.e, dich.split),
        )
        fit = None
        # Skip to the second mask if the breaker count is positive
        if self.prc.mapping.cur_breaks[lvl][dpt] > 0:
            self.prc.mapping.cur_breaks[lvl][dpt] -= 1
            if not comp1:
                return False
            # Breaking is permanent, so fitting to the first mask is now forbidden
            dich.masks[0].freeze = True
            fit = 1

        # Determine the fit the normal way
        if not fit:
            # If both masks are fitting, choose the first one unless
            # it is the only one that increases rep
            if conds[0] and comp0 and conds[1] and comp1:
                if comp1[1] == dich.masks[1].rep and comp0[1] > dich.masks[0].rep:
                    fit = 1
                else:
                    fit = 0
            # First mask fitting (the second one wasn't fit OR ret is not forbidden)
            elif conds[0] and comp0:
                fit = 0
            # Second mask fitting (the first one wasn't fit OR skip is not forbidden)
            elif conds[1] and comp1:
                fit = 1
            # Otherwise and if the dich is non-binary, perform a shift and try again
            elif dich.nb and not forbid_shift:
                self._shift_nonbinary_mappings(dich, invert=True, force=True)
                return self._decide_dichotomy(dich, forbid_shift=True)
            # If all fails
            else:
                logger.warning(f"{P} Could not decide {dich} for '{self.e}'")
                return False

        # Attempt closure if the obtained fit flips the pointer to 1 permanently
        # Makes sense only for return-restricted, non-terminal dichs
        if all([dich.ret, not dich.terminal, (dich.pointer or 0) != fit]):
            closure = self._close_dichotomies(dich)
            if not closure:
                logger.warning(f"{P} Could not close {dich}")
                return False

        dich.pointer = fit
        old_mask = f"{dich.masks[fit]}"

        # Prepare the decision
        comp = comp0 if fit == 0 else comp1
        assert comp is not None  # the chosen mask had a fit
        self.prc.masker.set_dichotomy(dich, comp, lvl)
        pos = fit if not dich.rev else 1 - fit
        rep = dich.masks[fit].rep

        new_mask = f"{dich.masks[fit]}"
        num_strings = ["1st", "2nd"]
        content = f"{P} Fitting '{self.e}' to the {num_strings[fit]}"
        if dich.split:
            logger.debug(f"{content} mask {new_mask}")
        else:
            logger.debug(f"{content} mask {old_mask} → {new_mask}")

        self.e.stance.pos.append(pos)
        self.e.stance.rep.append(rep)

        return True

    def _shift_nonbinary_mappings(
        self, dich: Dichotomy, invert: bool = False, force: bool = False
    ) -> bool:
        """Shifts the mappings of the elements in the given list (or in the
        current stack if none are given) from the first to the second mask
        of the dichotomy (or vice versa if invert is True) continuously slot by slot
        as long as the shift produces a valid mapping.
        """
        # Shift is only possible if the target mask has an empty slot.
        # Equivalent mappings are shifted together or not at all,
        # compounds from closest to farthest to the target mask
        # and only if they fit (given lembs and perms).
        # Get keys for both masks
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
                    old_stance = str(elems[i].stance)
                    new_stance = elems[i].stance.copy()
                    new_stance.pos[dich.d] = mask_to.num_key[-1]
                    new_stance.rep[dich.d] = mask_to.rep + n
                    fit = self._fit_element(elems[i], new_stance, dich.d)
                    if fit:
                        dpt = elems[i].stance.depth
                        logger.debug(
                            f"[L{level}|D{dpt}] Shifted {elems[i]} from "
                            f"{old_stance} to {new_stance}"
                        )
                        slot_in_process = True
                        elems_shifted += 1
                    else:
                        # Terminate successfully if the first elem from slot fails
                        # unsuccessfully if any other elem fails
                        res = not slot_in_process
                        if not res:
                            logger.warning(
                                f"=> No place to shift {elems[i]} along {dich}"
                            )
                        return res
                num -= 1
            slots_shifted += 1

        # Reset the dichotomies downstream of the mask &
        # subtract the shifted elements from the mask
        if elems_shifted > 0:
            self.prc.masker.reset_dichotomies(level, mask_from.depth, mask_from.num_key)
            mask_from.subtract(elems_shifted, slots_shifted)

        return True

    def _fill_empty_terminals(self, dich: Dichotomy) -> bool:
        """Adds neutral elements to match to empty masks within the given dichotomy
        that (1) have a sibling with some elements fit or (2) are necessary.

        Case (1) is applicable only to the zeroth level.
        """
        lvl, dpt = dich.level, dich.depth
        enum_elems = self.prc.mapping.enumerate_elems

        def get_sides() -> tuple[
            list[Element], list[list[int]], list[dict[str, list[int]]]
        ]:
            elems = self.prc.mapping.get_stack(lvl, interval=True)
            nk = [dich.masks[i].num_key for i in range(2)]
            hits = [enum_elems(nk[i], elems, dich.d) for i in range(2)]
            return elems, nk, hits

        # Find and fill empty necessary mask slots
        elems, nk, hits = get_sides()
        # logger.debug(f"Filling {dich} with l={hits[0]},r={hits[1]}")
        to_fill = []
        for i, mask in enumerate(dich.masks):
            if mask.necessities[0] and not hits[i]:
                prs = enum_elems(nk[i], elems, dich.d, True)
                if prs:
                    last_index = [idx for slot in prs.values() for idx in slot][-1]
                    op_stance = Stance(
                        pos=nk[1 - i],
                        rep=[0] * len(nk[1 - i]),
                        depth=mask.depth,
                    )
                    to_fill.append(([last_index], mask, op_stance))
        if not self._neutralize(to_fill, lvl, dpt, compensate=False):
            return False

        # Find and fill empty sibling slots (level 0 only)
        if lvl == 0:
            elems, nk, hits = get_sides()
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
            if not self._neutralize(to_fill, lvl, dpt, compensate=True):
                return False

        return True

    def _neutralize(
        self,
        to_fill: list[tuple[list[int], Mask, Stance]],
        lvl: int,
        dpt: int,
        compensate: bool,
    ) -> bool:
        """Inserts neutral elements at the given addresses for the given level
        and depth.
        """
        # logger.debug(f"Neutralizing on level {lvl} at {to_fill}")
        elems = self.prc.mapping.get_stack(lvl, interval=True)
        for indices, neut_mask, op_stance in to_fill:
            # Masks without a configured neutral for this depth (any mask
            # outside the lowest level-0 terminals, or a depth beyond the
            # configured list) cannot be filled; skip them rather than crash.
            if not neut_mask.tneuts or dpt >= len(neut_mask.tneuts):
                continue
            if not neut_mask.tneuts[dpt]:
                continue
            op_stance.pos[-1] = 1 - op_stance.pos[-1]
            t = self.prc.alphabet.get_token(neut_mask.tneuts[dpt])
            neut = Element(t, op_stance, lvl)
            if not self._fit_element(neut, op_stance, term_only=True):
                logger.warning(f"[L{lvl}|D{dpt}] Could not fit {neut} to {op_stance}")
                return False

            logger.debug(f"[L{lvl}|D{dpt}] Inserting {neut} with stance {op_stance}")

            if compensate:
                slot = op_stance.pos[-1] if not neut_mask.rev else 1 - op_stance.pos[-1]
                insert_index = min(indices) if slot == 0 else max(indices)
            else:
                slot = 1
                insert_index = max(indices)

            base_order = elems[insert_index].head.tok.base.order
            slot_order = slot - 1 if compensate else 1
            neut.head.tok.base.order = base_order + slot_order

            if not self.e.molar and self.e.content[0].level != self.e.level:
                self.e.content.insert(insert_index + slot, neut)

            stack = self.prc.mapping.get_stack(lvl)
            stack.insert(self.prc.mapping.cur_bdr[lvl] + insert_index + slot, neut)

        return True

    def _validate_mapping(self) -> bool:
        """Checks that every stance in the current element's content
        complies with terminal permissions.
        """
        level = 0
        elems = self.e.content
        cnt = 0
        addr = None
        for e in elems:
            if e.level != 0:
                continue
            cnt = cnt + 1 if e.stance.pos + e.stance.rep[:-1] == addr else 0
            addr = e.stance.pos + e.stance.rep[:-1]

            rev = bool(self.prc.grules.revs[level][0][e.num])
            addrs = [e for e in elems if e.stance.pos + e.stance.rep[:-1] == addr]

            perms = self.prc.srules.tperms[e.num]
            priority = cnt if not rev else len(addrs) - 1 - cnt
            depth_perms = perms[min(e.stance.depth, len(perms) - 1)]
            # A priority beyond the permission list means more elements landed
            # at this address than the slot licenses: the element is unpermitted.
            if priority >= len(depth_perms):
                logger.warning(
                    f"-> No permission slot for priority {priority} at {e.stance}"
                )
                return False
            perm = depth_perms[priority]

            aclass = e.head.tok.base.aclass
            base = str(e.head.tok.base)
            if not any([base in perm or aclass in perm, aclass == "wildcard"]):
                logger.warning(
                    f"-> No permission for '{e.head}'/'{aclass}' at {e.stance}"
                )
                return False

        return True

    def _close_dichotomies(self, dich: Dichotomy | None = None) -> bool:
        """For dichotomies downstream of the given dichotomy, performs
        the finalizing operations: shift the mappings for the non-binary ones,
        add neutral elements as needed for the terminal ones.

        If no dichotomy is given, starts from the last fitted branch
        of the topmost dichotomy for the current element.
        """
        if dich is None:
            elems = self.e.content
            level, depth = elems[0].level, elems[0].stance.depth
            dich = self.prc.masker.get_dichs(Stance(depth=depth), level)[0]
        else:
            elems = None
            level, depth = dich.level, dich.depth
        res = True
        stance = Stance(pos=dich.masks[dich.pointer or 0].num_key, depth=depth)
        dichs = self.prc.masker.get_dichs(stance, level)
        invert = bool(dich.pointer or 0)

        for cdich in dichs:
            if cdich.nb:
                res = res and self._shift_nonbinary_mappings(cdich, invert=invert)
            if cdich.terminal:
                res = res and self._fill_empty_terminals(cdich)

        return res

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

    def determine_stance(self, e: Element) -> bool:
        """Sets the given element as current and  produces the stance for it
        by deciding the dichotomies.
        """
        self.e = e
        for _ in range(0, sum(self.prc.grules.struct[e.level])):
            dich = self.prc.masker.get_dichs(e.stance, e.level)[0]
            if not self._decide_dichotomy(dich):
                raise ParsingFailure(f"Failed to parse '{e}'")
        return True

    def close(self, e: Element) -> None:
        """Sets the given element as current, closes the corresponding dichotomy
        and applies special rules to validate its content.
        """
        self.e = e
        if e.molar:
            return
        if not self._close_dichotomies():
            raise ParsingFailure("Failed to close the dichotomies")

        if not self._validate_mapping():
            raise ParsingFailure("Failed to validate the mapping")

        return


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

    def interpret(self, tree: Tree) -> None:
        """Inscribes interpretations defined in the dialect to the tree nodes
        depending on their content.
        """
        # Find features for the tree nodes and their compounds
        nodes = [n for n in tree.all_nodes if n.terminal and n.content]
        for node in nodes:
            if not node.content:
                continue
            e = node.content[0]
            feature = self.prc.dialect.get_feature(
                e.header.tok.base.index,
                e.header.tok.base.aclass,
                node.stance,
                tree.stance,
                tree.ctype,
            )
            if not feature:
                continue
            node.feature = feature
        # Interpret the embedded trees
        for c in tree.complexes:
            self.interpret(c)
        return

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
            tree = self.prc.trees[0][-1]
        elif isinstance(tree, int):
            tree = self.prc.trees[tree][-1]

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
