from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import log
from typing import Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from scripts.util import concat, digits, flatten, is_int, lemb_rank_sizes


class Alphabet(BaseModel):
    """A holder for the characters used in the language.
    Has functions for removing non-alphabetic and representing alphabetic characters.

    Doubles as the schema for params/alphabet.yaml. 'extra' is allowed so that
    the derived character dictionaries built by _build_dicts can be attached to
    the model.
    """

    model_config = ConfigDict(extra="allow")

    bases: dict[str, dict]
    modifiers: dict
    substitutions: dict

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Self:
        """Checks the alphabet's bases, modifiers and substitutions structure,
        then builds the derived character dictionaries. Guiding groups carry one
        member per language level, supplied via the 'levels' validation context.
        A failing check reports the offending group or parameter name.
        """
        levels = (info.context or {}).get("levels")

        def is_member(m: object) -> bool:
            return isinstance(m, str) or (
                isinstance(m, list) and all(isinstance(x, str) for x in m)
            )

        def check_groups(groups: dict, count: int) -> None:
            for name, group in groups.items():
                if not (
                    isinstance(group, list)
                    and len(group) == count
                    and all(is_member(m) for m in group)
                ):
                    raise ValueError(name)

        bases = self.bases
        if "content" not in bases or "guiding" not in bases:
            raise ValueError("bases")
        content = bases["content"]
        if not (
            isinstance(content, dict)
            and content
            and all(isinstance(v, list) for v in content.values())
        ):
            raise ValueError("content")
        guiding = bases["guiding"]
        if not (
            isinstance(guiding, dict)
            and all(k in guiding for k in ("wildcard", "separator", "embedder"))
        ):
            raise ValueError("guiding")
        if levels is not None:
            check_groups(guiding, levels)

        modifiers = self.modifiers
        if "breaker" not in modifiers or "swapper" not in modifiers:
            raise ValueError("modifiers")
        if not isinstance(modifiers["breaker"], dict):
            raise ValueError("breaker")
        # Breakers are defined for the bottom level only: one pair per class.
        check_groups(modifiers["breaker"], 1)
        swappers = modifiers["swapper"]
        if not isinstance(swappers, list):
            raise ValueError("swapper")
        for pair in swappers:
            if not (
                isinstance(pair, list)
                and len(pair) == 2
                and all(
                    isinstance(m, dict)
                    and all(
                        isinstance(k, str) and isinstance(v, str) and len(v) == 1
                        for k, v in m.items()
                    )
                    for m in pair
                )
            ):
                raise ValueError("swapper")

        subs = self.substitutions
        if "free" not in subs or "elongator" not in subs:
            raise ValueError("substitutions")
        free = subs["free"]
        if not (
            isinstance(free, dict)
            and all(
                isinstance(k, str) and len(k) == 1 and isinstance(v, str)
                for k, v in free.items()
            )
        ):
            raise ValueError("free")
        elongator = subs["elongator"]
        classes = bases["content"]
        if not (
            isinstance(elongator, dict)
            and all(
                k in classes and isinstance(v, str) and len(v) == 1
                for k, v in elongator.items()
            )
        ):
            raise ValueError("elongator")

        # Validation done; attach the derived character dictionaries.
        self._build_dicts()
        return self

    def _build_dicts(self) -> None:
        """Adds to the alphabet various dictionaries of characters
        useful on different stages of string parsing.
        """

        self.content = self.bases["content"]
        self.wildcards = self.bases["guiding"]["wildcard"]
        self.separators = self.bases["guiding"]["separator"]
        self.embedders = self.bases["guiding"]["embedder"]
        self.breakers = self.modifiers["breaker"]
        self.swappers = self.modifiers.get("swapper", [])
        # Free substitutions replace single characters; elongators replace a
        # content character that repeats the previous one of the same class.
        self.free = self.substitutions["free"]
        self.elongators = self.substitutions["elongator"]

        # Groups that follow the level/quality/index layout: every base group
        # and the breaker modifiers. Swappers carry their own shape and are
        # indexed separately below.
        d = {"Base": self.bases, "Modifier": {"breaker": self.breakers}}

        # Creating a flat dictionary of characters with all parameters encoded.
        # Each entry is a Symbol template (order 0) copied with the occurrence
        # order when a character is read; see get_token and Streamer._tokenize.
        self.lookup: dict[str, Symbol] = {}
        for cat in d:
            for subcat in d[cat]:
                for aclass in d[cat][subcat]:
                    # A level entry is a string or a list of per-quality strings.
                    group = cast("list[str | list[str]]", d[cat][subcat][aclass])
                    for level, values in enumerate(group):
                        if isinstance(values, list):
                            for q, vals in enumerate(values):
                                for i, val in enumerate(vals):
                                    self.lookup[val] = Symbol(
                                        content=val,
                                        acat=cat,
                                        asubcat=subcat,
                                        aclass=aclass,
                                        level=level,
                                        quality=None if val in self.lookup else q,
                                        index=i,
                                    )
                        else:
                            for i, val in enumerate(values):
                                self.lookup[val] = Symbol(
                                    content=val,
                                    acat=cat,
                                    asubcat=subcat,
                                    aclass=aclass,
                                    level=level,
                                    quality=0,
                                    index=i,
                                )

        # Swappers pair single characters from two content classes; each pair
        # member is indexed under its own class so it attaches to that content.
        for p, pair in enumerate(self.swappers):
            for q, mapping in enumerate(pair):
                for aclass, char in mapping.items():
                    self.lookup[char] = Symbol(
                        content=char,
                        acat="Modifier",
                        asubcat="swapper",
                        aclass=aclass,
                        level=0,
                        quality=None if char in self.lookup else q,
                        index=p,
                    )

        # Each swapper pair maps its opening character (the first member) and its
        # closing character (the second) to the pair's index.
        self.openings = {
            next(iter(p[0].values())): i for i, p in enumerate(self.swappers)
        }
        self.closings = {
            next(iter(p[1].values())): i for i, p in enumerate(self.swappers)
        }

        return

    def get_token(self, st: str) -> Token:
        """Creates a token based on the given base character."""
        spec = self.lookup.get(st)
        if spec is not None and spec.acat == "Base":
            return Token(base=replace(spec))
        raise ValueError(f"Tried to get a token with the illegal character {st}")


class GeneralRules(BaseModel):
    """A holder for the general rules of the language, which deal with the mapping
    of elements to masks in relation to dichotomies.

    Doubles as the schema for params/rules_general.yaml: 'struct' fixes the tree
    shape per level; dichotomy-indexed params hold N entries per level and
    node-indexed ones 2**N, while 'lembs' is rank-indexed. After validation,
    perms/revs/dembs/wilds are unraveled per mask.
    """

    struct: list
    heads: list
    rets: list
    skips: list
    splits: list
    revs: list
    dembs: list
    perms: list
    lembs: list
    wilds: list

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Checks the general rules against the shape and value types implied by
        their own structure. Each rule pairs a group of like-shaped parameters
        with the entries expected per level and the predicate every leaf value
        must satisfy; a failing check reports the offending parameter name.
        """
        struct = self.struct
        if not (
            struct
            and all(
                isinstance(lv, list) and lv and all(is_int(x) and x >= 0 for x in lv)
                for lv in struct
            )
        ):
            raise ValueError("struct")
        levels = len(struct)
        sums = [sum(lv) for lv in struct]  # total dichotomies (N_i) per level

        # (parameters, entries expected per level or None, leaf-value predicate);
        # dichotomy-indexed params hold N entries per level, node-indexed ones 2**N.
        rules: list[tuple[tuple[str, ...], Callable | None, Callable]] = [
            (("heads",), None, lambda x: is_int(x) and x >= 0),
            (
                ("rets", "skips", "splits"),
                lambda n: n,
                lambda x: is_int(x) and x in (0, 1),
            ),
            (("revs", "wilds"), lambda n: 2**n, lambda x: is_int(x) and x in (0, 1)),
            (("dembs",), lambda n: 2**n, lambda x: is_int(x) and x >= -1),
            (("perms",), lambda n: 2**n, lambda x: isinstance(x, str)),
        ]
        for names, length_of, ok in rules:
            for name in names:
                param = getattr(self, name)
                if len(param) != levels:
                    raise ValueError(name)
                for i, sub in enumerate(param):
                    if not isinstance(sub, list) or (
                        length_of is not None and len(sub) != length_of(sums[i])
                    ):
                        raise ValueError(name)
                    if not all(ok(x) for x in flatten(sub)):
                        raise ValueError(name)

        # 'lembs' is rank-indexed: one list per node-count level, the k-th holding
        # 2**(dichotomies up to rank k) entries (zero-dichotomy ranks collapse).
        if len(self.lembs) != levels:
            raise ValueError("lembs")
        for i, ranks in enumerate(self.lembs):
            if not all(isinstance(r, list) for r in ranks):
                raise ValueError("lembs")
            if [len(r) for r in ranks] != lemb_rank_sizes(struct[i]):
                raise ValueError("lembs")
            if not all(is_int(x) and x >= -1 for x in flatten(ranks)):
                raise ValueError("lembs")

        # Validation done; transform the terminal params in place.
        self._unravel_term_params()
        return self

    def _unravel_term_params(self) -> None:
        """Transforms perms, revs, dembs, and wilds for each mask
        based on the general rules where they are defined in a more compact form.
        """
        for lv in range(len(self.struct)):
            self.perms[lv] = [self.perms[lv]]
            self.revs[lv] = [self.revs[lv]]
            self.dembs[lv] = [self.dembs[lv]]
            self.wilds[lv] = [self.wilds[lv]]
            for r in range(int(log(len(self.perms[lv][0]), 2))):
                split_perms, split_revs, split_dembs, split_wilds = [], [], [], []
                for i in range(0, len(self.perms[lv][-1]), 2):
                    rev = min(self.revs[lv][-1][i : i + 2]) if r == 0 else 0
                    demb = max(self.dembs[lv][-1][i : i + 2])
                    wilds = max(self.wilds[lv][-1][i : i + 2])
                    perm = self.perms[lv][-1][i : i + 2]
                    split_perms.append("".join(perm[::-1] if rev else perm))
                    split_revs.append(rev)
                    split_dembs.append(demb)
                    split_wilds.append(wilds)
                self.perms[lv].append(split_perms)
                self.revs[lv].append(split_revs)
                self.dembs[lv].append(split_dembs)
                self.wilds[lv].append(split_wilds)
        return


class SpecialRules(BaseModel):
    """A holder for the special rules of the language, which deal with permitting
    or forbidding certain characters for certain stances post-mapping.

    Doubles as the schema for params/rules_special.yaml: 'tperms' is one entry
    per terminal node, each a per-depth list of permission strings; 'tneuts' is
    one entry per terminal node, each a list of neutral-character strings.
    """

    tperms: list[list[list[str]]]
    tneuts: list[list[str]]

    @model_validator(mode="after")
    def _check_content(self, info: ValidationInfo) -> Self:
        """Enforces the cross-parameter content rules when the validation context
        provides them: 'slots' (2 ** the bottom-level dichotomy count) fixes the
        number of terminal nodes for both tperms and tneuts, and 'content_chars'
        restricts each neutral to empty or a single content character. Without a
        context only the structural shape above is enforced.
        """
        context = info.context or {}
        slots = context.get("slots")
        if slots is not None:
            for name, value in (("tperms", self.tperms), ("tneuts", self.tneuts)):
                if len(value) != slots:
                    raise ValueError(name)
        chars = context.get("content_chars")
        if chars is not None:
            for slot in self.tneuts:
                if any(s != "" and (len(s) != 1 or s not in chars) for s in slot):
                    raise ValueError("tneuts")
        return self


class Feature(BaseModel):
    """A holder for a feature with all the parameters that describe it. Doubles
    as the row schema for params/features.tsv: 'ctype' is optional (blank means
    untyped), the four stance vectors are digit strings decoded into integer
    lists, and the numeric columns are coerced to integers.
    """

    ctype: str | None
    pos: list[int]
    rep: list[int]
    cpos: list[int]
    crep: list[int]
    content_class: str
    priority: int
    index: int
    function_name: str
    argument_gloss: str
    argument_name: str
    argument_description: str

    @field_validator("ctype", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """A blank content type marks an untyped feature."""
        return value or None

    @field_validator("pos", "rep", "cpos", "crep", mode="before")
    @classmethod
    def _decode_digits(cls, value: object) -> object:
        """Stance vectors are stored as digit strings (e.g. '010')."""
        return [int(c) for c in value] if isinstance(value, str) else value

    def __repr__(self) -> str:
        return f"{self.function_name}: {self.argument_name}"


class Type(BaseModel):
    """A composition or permutation type with its descriptive metadata,
    mirroring the description fields of a feature.

    Doubles as the row schema for params/types.tsv: every column is required
    and 'priority' is coerced to an integer.
    """

    type: str
    priority: int
    argument_name: str
    argument_description: str

    def __repr__(self) -> str:
        return f"{self.type}: {self.argument_name}"


class Dialect(BaseModel):
    """A holder for the parameters that guide the interpretation of node features.

    Doubles as the schema for the composition and permutation types in
    params/dialect.yaml; the 'features' and 'types' description lists are loaded
    separately from the TSV files and attached by the Loader.
    """

    ctypes: list
    ptypes: list
    features: list[Feature] = Field(default_factory=list)
    types: list[Type] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Self:
        """Checks the composition types against the level and rank structure
        (supplied via the 'struct' validation context) and the permutation types
        against the swap-pair shape. Each ctype carries a mandatory 'ranks'
        restriction (one string per tree rank) and optional non-negative
        'cpos'/'crep' vectors; each ptype carries a 'swaps' list of integer
        pairs. A failing check reports the offending parameter or type name.
        """
        struct = (info.context or {}).get("struct")
        if struct is not None and len(self.ctypes) != len(struct):
            raise ValueError("ctypes")
        for i, level in enumerate(self.ctypes):
            if not isinstance(level, dict):
                raise ValueError("ctypes")
            ranks = len(lemb_rank_sizes(struct[i])) if struct is not None else None
            for types in level.values():
                if not isinstance(types, dict):
                    raise ValueError("ctypes")
                for tname, tdef in types.items():
                    rank_perms = tdef.get("ranks") if isinstance(tdef, dict) else None
                    if not (
                        isinstance(rank_perms, list)
                        and all(isinstance(r, str) for r in rank_perms)
                    ):
                        raise ValueError(tname)
                    if ranks is not None and len(rank_perms) != ranks:
                        raise ValueError(tname)
                    for opt in ("cpos", "crep"):
                        vec = tdef.get(opt)
                        if vec is not None and not (
                            isinstance(vec, list)
                            and all(is_int(x) and x >= 0 for x in vec)
                        ):
                            raise ValueError(tname)

        for entry in self.ptypes:
            if not isinstance(entry, dict):
                raise ValueError("ptypes")
            for pname, pdef in entry.items():
                swaps = pdef.get("swaps") if isinstance(pdef, dict) else None
                if not (
                    isinstance(swaps, list)
                    and all(
                        isinstance(s, list)
                        and len(s) == 2
                        and all(is_int(x) for x in s)
                        for s in swaps
                    )
                ):
                    raise ValueError(pname)
        return self

    def get_feature(
        self,
        index: int,
        content_class: str,
        stance: Stance,
        cstance: Stance | None,
        ctype: str | None,
    ) -> Feature | None:
        """Returns the feature described by the stances and character index,
        or None if no feature is found. A feature bound to a content type matches
        only when that type equals the supplied one; an untyped feature matches
        regardless of the supplied content type.
        """
        feature_list = [
            f
            for f in self.features
            if (f.ctype or ctype) == ctype and f.content_class == content_class
        ]
        for f in feature_list:
            if all((f.index == index, f.pos == stance.pos, f.rep == stance.rep)):
                if (not cstance and not f.cpos and not f.crep) or (
                    cstance and f.cpos == cstance.pos and f.crep == cstance.rep
                ):
                    return f
        return None


@dataclass
class Stance:
    """A representation of the mapping between the element and the mask."""

    pos: list[int] = field(default_factory=list)
    rep: list[int] = field(default_factory=list)
    depth: int = 0

    def __repr__(self) -> str:
        return f"[{concat(self.pos)}|{concat(self.rep)}|{self.depth}]"

    @staticmethod
    def decode(st: str) -> Stance:
        """Returns a Stance object with pos, rep, and depth
        based on the provided string.
        """
        stance = Stance()
        data = st.strip("[]").split("|")
        if data and data[0].isdigit():
            stance.pos = digits(data[0])
        if len(data) > 1 and data[1].isdigit():
            stance.rep = digits(data[1])
        if len(data) > 2 and data[2].isdigit():
            stance.depth = int(data[2])
        return stance

    @property
    def key(self) -> str:
        """The key of the stance is the binary representation of its position."""
        return concat(self.pos)

    def copy(self, lim: int | None = None) -> Stance:
        """Creates a copy of the stance with pos and rep limited from the right
        by the given index.
        """
        pos = self.pos[:lim] if lim is not None else self.pos[:]
        rep = self.rep[:lim] if lim is not None else self.rep[:]
        depth = self.depth
        return Stance(pos, rep, depth)


@dataclass
class Token:
    """A holder for elementary emic units of the language."""

    base: Symbol
    modifiers: list[Symbol] = field(default_factory=list)

    def __repr__(self) -> str:
        return self.content or "Empty symbol"

    def is_pusher(self, lvl: int) -> bool:
        """Checks if the token acts as a pusher."""
        base = self.base.aclass == "embedder"
        level = self.base.level == lvl
        quality = self.base.quality in (None, 0)
        return all((base, level, quality))

    def is_popper(self, lvl: int) -> bool:
        """Checks if the token acts as a popper."""
        base = self.base.aclass == "embedder"
        level = self.base.level == lvl
        quality = self.base.quality in (None, 1)
        return all((base, level, quality))

    @property
    def content(self) -> str:
        """How the token appears in writing."""
        base = str(self.base)
        mods = "".join([m.content for m in self.modifiers])
        return base + mods

    @property
    def lit(self) -> str:
        """The string representation of the token's base used for matching."""
        return str(self.base).lower()

    @property
    def order(self) -> int:
        """The order of the token is the minimum of those of its symbols."""
        return min([self.base.order] + [m.order for m in self.modifiers])

    @property
    def swapper(self) -> str | None:
        """The swapper character among the token's modifiers, if any."""
        for mod in self.modifiers:
            if mod.asubcat == "swapper":
                return mod.content
        return None


@dataclass
class Symbol:
    """A holder for elementary etic units of the language."""

    content: str = ""
    acat: str = ""
    asubcat: str = ""
    aclass: str = ""
    level: int = 0
    quality: int | None = 0
    index: int = 0
    order: int = 0

    def __repr__(self) -> str:
        return self.content
