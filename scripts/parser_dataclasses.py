from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import log
from typing import ClassVar, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from scripts.util import (
    concat,
    digits,
    flatten,
    is_int,
    lemb_rank_sizes,
    tree_node_count,
)


@dataclass
class ParsingResult:
    """The four properties a string may satisfy as a language element, in the
    order of docs/result.md. Each starts undetermined (None) and is set as its
    condition is decided while the processor runs; `well_formed` is their
    syntactic conjunction.

    The criteria form a dependency chain: an unintelligible string cannot be
    properly parsed, an incomplete mapping cannot be fully interpreted, and a
    statement lacking interpretability even in part cannot be assessed for
    felicity. Failing one criterion therefore fails every criterion after it,
    which `__setattr__` enforces no matter where the failure is recorded.
    """

    intelligibility: bool | None = None
    grammaticality: bool | None = None
    interpretability: bool | None = None
    felicity: bool | None = None

    # The dependency chain, in field order.
    CRITERIA: ClassVar[tuple[str, ...]] = (
        "intelligibility",
        "grammaticality",
        "interpretability",
        "felicity",
    )

    def __setattr__(self, name: str, value: object) -> None:
        """Sets the attribute; a criterion set to False also drags every
        criterion downstream of it in the chain to False.
        """
        object.__setattr__(self, name, value)
        if value is False and name in self.CRITERIA:
            for downstream in self.CRITERIA[self.CRITERIA.index(name) + 1 :]:
                object.__setattr__(self, downstream, False)

    @property
    def well_formed(self) -> bool:
        """True when the string is intelligible, grammatical, and interpretable.
        Felicity is a pragmatic property and lies beyond well-formedness.
        """
        wf = all([self.intelligibility, self.grammaticality, self.interpretability])
        return wf


class Alphabet(BaseModel):
    """A holder for the characters used in the language.
    Has functions for removing non-alphabetic and representing alphabetic characters.

    Doubles as the schema for params/alphabet.toml. 'extra' is allowed so that
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
        if not (isinstance(content, dict) and content):
            raise ValueError("content")
        # Each content class is a pair [structure, characters]: the structure is
        # a list of non-negative integers defining a dichotomic tree, and the
        # characters are single-character strings, one per tree node, so their
        # count must equal the number of nodes the structure defines.
        for cls, spec in content.items():
            if not (
                isinstance(spec, list)
                and len(spec) == 2
                and isinstance(spec[0], list)
                and all(is_int(x) and x >= 0 for x in spec[0])
                and isinstance(spec[1], list)
                and all(isinstance(c, str) and len(c) == 1 for c in spec[1])
                and len(spec[1]) == tree_node_count(spec[0])
            ):
                raise ValueError(cls)
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

        # Creating a flat dictionary of characters with all parameters encoded.
        # Each entry is a Symbol template (order 0) copied with the occurrence
        # order when a character is read; see get_token and Streamer._tokenize.
        self.lookup: dict[str, Symbol] = {}

        # Content characters occupy the nodes of their class tree in list order,
        # so each character's index is its node number; the structures are kept
        # for addressing characters by tree position. Content is bottom-level.
        self.structs: dict[str, list[int]] = {}
        for aclass, (struct, chars) in self.content.items():
            self.structs[aclass] = struct
            for i, val in enumerate(chars):
                self.lookup[val] = Symbol(
                    content=val,
                    acat="Base",
                    asubcat="content",
                    aclass=aclass,
                    level=0,
                    quality=0,
                    index=i,
                )

        # Guiding bases and breaker modifiers follow the level/quality/index
        # layout. Swappers carry their own shape and are indexed separately below.
        d = {
            "Base": {"guiding": self.bases["guiding"]},
            "Modifier": {"breaker": self.breakers},
        }
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

    @property
    def class_names(self) -> list[str]:
        """The content class names, in the order that indexes them."""
        return list(self.content)

    def class_chars(self, cls: int) -> list[str]:
        """The characters of the content class with the given number."""
        return self.content[self.class_names[cls]][1]

    def _resolve_perm(
        self, entry: object, *, special_ok: bool
    ) -> tuple[int | None, list[int] | None, str]:
        """Validates an index-list permission entry against the content classes,
        splitting it into (class number or None if empty, character indices or
        None for the whole class, special-symbol string). The entry is [] for an
        empty permission, [c] for a whole class, or [c, [indices...]] for a set
        of characters; masks may append a "?" or "!" symbol. Raises ValueError on
        any malformed or out-of-range entry.
        """
        if not isinstance(entry, list):
            raise ValueError(f"permission entry must be a list, got {entry!r}")
        if not entry:
            return None, None, ""
        cls = entry[0]
        if not is_int(cls) or not 0 <= cls < len(self.class_names):
            raise ValueError(f"invalid content class {cls!r} in {entry!r}")
        pool = self.class_chars(cls)
        indices: list[int] | None = None
        special = ""
        for extra in entry[1:]:
            if isinstance(extra, list):
                if not all(is_int(i) and 0 <= i < len(pool) for i in extra):
                    raise ValueError(f"character index out of range in {entry!r}")
                indices = cast("list[int]", extra)
            elif special_ok and extra in ("?", "!"):
                special = extra
            else:
                raise ValueError(f"invalid permission element {extra!r} in {entry!r}")
        return cls, indices, special

    def mask_perm(self, entry: object) -> str:
        """Builds a mask permission string (as used in `perms`) from an index-list
        entry: the special symbol, then the class name for a whole class or the
        characters otherwise, parenthesized when there is more than one variant.
        """
        cls, indices, special = self._resolve_perm(entry, special_ok=True)
        if cls is None:
            return ""
        if indices is None:
            return special + self.class_names[cls]
        chars = "".join(self.class_chars(cls)[i] for i in indices)
        return special + (f"({chars})" if len(indices) > 1 else chars)

    def membership_perm(self, entry: object) -> str:
        """Builds a terminal-permission membership string (as used in `tperms`)
        from an index-list entry: the class name for a whole class, or the plain
        concatenation of the selected characters.
        """
        cls, indices, _ = self._resolve_perm(entry, special_ok=False)
        if cls is None:
            return ""
        if indices is None:
            return self.class_names[cls]
        return "".join(self.class_chars(cls)[i] for i in indices)

    def neutral_perm(self, entry: object) -> str:
        """Builds a neutral character (as used in `tneuts`) from an index-list
        entry: the single selected character, or "" for an empty entry.
        """
        cls, indices, _ = self._resolve_perm(entry, special_ok=False)
        if cls is None:
            return ""
        if indices is None or len(indices) != 1:
            raise ValueError(f"a neutral must be a single character, got {entry!r}")
        return self.class_chars(cls)[indices[0]]


class GeneralRules(BaseModel):
    """A holder for the general rules of the language, which deal with the mapping
    of elements to masks in relation to dichotomies.

    Doubles as the schema for params/rules_general.toml: 'struct' fixes the tree
    shape per level; dichotomy-indexed params hold N entries per level and
    node-indexed ones 2**N, while 'lembs' is rank-indexed. 'perms' entries are
    index-list permissions (see Alphabet.mask_perm) rebuilt into mask strings
    from the alphabet supplied in the validation context. After validation,
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
    def _check(self, info: ValidationInfo) -> Self:
        """Checks the general rules against the shape and value types implied by
        their own structure. Each rule pairs a group of like-shaped parameters
        with the entries expected per level and the predicate every leaf value
        must satisfy; a failing check reports the offending parameter name. The
        'perms' entries are rebuilt into mask strings from the alphabet passed in
        the validation context.
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

        # 'perms' holds one index-list permission per terminal node (2**N per
        # level). Rebuild each into a mask string from the content classes of the
        # alphabet supplied in the validation context.
        alphabet = (info.context or {}).get("alphabet")
        if alphabet is None:
            raise ValueError("perms: alphabet context required")
        if len(self.perms) != levels:
            raise ValueError("perms")
        for i, level in enumerate(self.perms):
            if not isinstance(level, list) or len(level) != 2 ** sums[i]:
                raise ValueError("perms")
        self.perms = [[alphabet.mask_perm(e) for e in level] for level in self.perms]

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

    Doubles as the schema for params/rules_special.toml: 'tperms' is one entry
    per terminal node, each a per-depth list of index-list permissions; 'tneuts'
    is one entry per terminal node, each a list of index-list neutrals. Both are
    rebuilt into strings (membership strings and single characters) from the
    alphabet supplied in the validation context.
    """

    tperms: list
    tneuts: list

    @model_validator(mode="after")
    def _check_content(self, info: ValidationInfo) -> Self:
        """Enforces the cross-parameter content rules when the validation context
        provides them: 'slots' (2 ** the bottom-level dichotomy count) fixes the
        number of terminal nodes for both tperms and tneuts, and 'alphabet'
        rebuilds each index-list permission into its string form (a membership
        string for tperms, a single character for tneuts). Without a context only
        the structural shape above is enforced.
        """
        context = info.context or {}
        slots = context.get("slots")
        if slots is not None:
            for name, value in (("tperms", self.tperms), ("tneuts", self.tneuts)):
                if len(value) != slots:
                    raise ValueError(name)
        alphabet = context.get("alphabet")
        if alphabet is not None:
            for node in self.tperms:
                if not isinstance(node, list) or not all(
                    isinstance(depth, list) for depth in node
                ):
                    raise ValueError("tperms")
            self.tperms = [
                [[alphabet.membership_perm(e) for e in depth] for depth in node]
                for node in self.tperms
            ]
            for node in self.tneuts:
                if not isinstance(node, list):
                    raise ValueError("tneuts")
            self.tneuts = [
                [alphabet.neutral_perm(e) for e in node] for node in self.tneuts
            ]
        return self


class Feature(BaseModel):
    """A holder for a feature with all the parameters that describe it. Doubles
    as the row schema for params/features.tsv: 'lvl' is the language level the
    feature belongs to, the four stance vectors are digit strings decoded into
    integer lists, and the numeric columns are coerced to integers. Any matched
    property left blank ('ctype', 'content_class', 'index', or a stance vector)
    leaves the feature unrestricted in that dimension; only the level is always
    required and matched exactly.
    """

    lvl: int
    ctype: str | None
    pos: list[int]
    rep: list[int]
    cpos: list[int]
    crep: list[int]
    content_class: str | None
    priority: int
    index: int | None
    function_name: str
    argument_gloss: str
    argument_name: str
    argument_description: str

    @field_validator("ctype", "content_class", "index", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """A blank property leaves the feature unrestricted in that dimension."""
        return None if value is None or value == "" else value

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
    params/dialect.toml; the 'features' and 'types' description lists are loaded
    separately from the TSV files and attached by the Loader.
    """

    ctypes: list
    ptypes: list
    # Features grouped by language level: features[lvl] holds the features of
    # that level in file order, one (possibly empty) list per level.
    features: list[list[Feature]] = Field(default_factory=list)
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
        level: int,
        index: int,
        content_class: str,
        stance: Stance,
        cstance: Stance | None,
        ctype: str | None,
    ) -> Feature | None:
        """Returns the first feature of the given level that matches the
        stances and character index, or None if no feature is found. A blank
        feature property matches any input in that dimension; a non-blank one
        must equal the input. Only the level is scanned exactly: features of
        other levels are never considered.
        """
        for f in self.features[level]:
            if f.ctype is not None and f.ctype != ctype:
                continue
            if f.content_class is not None and f.content_class != content_class:
                continue
            if f.index is not None and f.index != index:
                continue
            if f.pos and f.pos != stance.pos:
                continue
            if f.rep and f.rep != stance.rep:
                continue
            if f.cpos and (cstance is None or f.cpos != cstance.pos):
                continue
            if f.crep and (cstance is None or f.crep != cstance.rep):
                continue
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