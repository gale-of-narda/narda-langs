from math import log

from typing import Dict
from dataclasses import dataclass, field


@dataclass
class Alphabet:
    """A holder for the characters used in the language.
    Has functions for removing non-alphabetic and representing alphabetic characters.
    """

    bases: Dict[str, Dict]
    modifiers: Dict[str, Dict[str, list[str]]]
    substitutions: Dict[str, str]

    def __post_init__(self) -> None:
        self._build_dicts()
        return

    def _build_dicts(self) -> None:
        """Adds to the alphabet various dictionaries of characters
        useful on different stages of string parsing.
        """

        self.content = self.bases["Content"]
        self.wildcards = self.bases["Guiding"]["Wildcard"]
        self.separators = self.bases["Guiding"]["Separator"]
        self.embedders = self.bases["Guiding"]["Embedder"]
        self.breakers = self.modifiers["Breaker"]

        d = {"Base": self.bases, "Modifier": self.modifiers}

        # Creating a flat dictionary of characters with all parameters encoded
        self.lookup = {}
        for cat in d:
            for subcat in d[cat]:
                for aclass in d[cat][subcat]:
                    for level, values in enumerate(d[cat][subcat][aclass]):
                        if isinstance(values, list):
                            for q, vals in enumerate(values):
                                for i, val in enumerate(vals):
                                    self.lookup[val] = {
                                        "Category": cat,
                                        "Subcategory": subcat,
                                        "Class": aclass,
                                        "Level": level,
                                        "Quality": None if val in self.lookup else q,
                                        "Index": i,
                                    }
                        else:
                            for i, val in enumerate(values):
                                self.lookup[val] = {
                                    "Category": cat,
                                    "Subcategory": subcat,
                                    "Class": aclass,
                                    "Level": level,
                                    "Quality": 0,
                                    "Index": i,
                                }

        return

    def get_token(self, st: str) -> Token:
        """Creates a token based on the given base character."""
        if st in [
            k for k in self.lookup.keys() if self.lookup[k]["Category"] == "Base"
        ]:
            s = Symbol(st, *self.lookup[st].values())
            g = Token(base=s)
            return g
        raise ValueError(f"Tried to get a token with the illegal character {st}")


@dataclass
class GeneralRules:
    """A holder for the general rules of the language, which deal with the mapping
    of elements to masks in relation to dichotomies.
    """

    struct: list[list[int]]
    heads: list[list[int]]
    rets: list[list[int]]
    skips: list[list[int]]
    splits: list[list[int]]
    revs: list[list[int]]
    dembs: list[list[int]]
    perms: list[list[str]]
    lembs: list[list[list[int]]]
    wilds: list[list[list[int]]]

    def __post_init__(self) -> None:
        self._unravel_term_params()
        return

    def _unravel_term_params(self) -> None:
        """Transforms perms, revs, dembs, and wilds for each mask
        based on the general rules where they are defined in a more compact form.
        """
        for lv, s in enumerate(self.struct):
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


@dataclass
class SpecialRules:
    """A holder for the special rules of the language, which deal with permitting
    or forbidding certain characters for certain stances post-mapping.
    """

    tperms: list[list[list[str]]]
    tneuts: list[list[list[str]]]


@dataclass
class Dialect:
    """A holder for the parameters that guide the interpretation of node features."""

    ctypes: list[Dict[str, str]]
    untyped: list[Feature]
    typed: list[Feature]

    def get_feature(
        self,
        index: int,
        content_class: str,
        stance: Stance,
        cstance: Stance | None,
        ctype: str | None,
    ) -> Feature | None:
        """Returns the feature described by the stances and character index,
        or None if no feature is found. If content type is supplied, searches
        the appropriate typed feature list first, then the untyped list
        if no match is found.
        """
        feature_list = [
            f
            for f in self.typed + self.untyped
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
class Feature:
    """A holder for a feature with all the parameters that describe it."""

    # None for untyped, str for typed features
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

    def __repr__(self) -> str:
        return f"{self.function_name}: {self.argument_name}"


@dataclass
class Stance:
    """A representation of the mapping between the element and the mask."""

    pos: list[int] = field(default_factory=lambda: [])
    rep: list[int] = field(default_factory=lambda: [])
    depth: int = 0

    def __repr__(self) -> str:
        pos = "".join(str(p) for p in self.pos)
        rep = "".join(str(r) for r in self.rep)
        depth = self.depth
        return f"[{pos}|{rep}|{depth}]"

    @staticmethod
    def decode(st: str) -> Stance:
        """Returns a Stance object with pos, rep, and depth
        based on the provided string.
        """
        stance = Stance()
        data = st.strip("[]").split("|")
        if data and data[0].isdigit():
            stance.pos = [int(d) for d in data[0]]
        if len(data) > 1 and data[1].isdigit():
            stance.rep = [int(d) for d in data[1]]
        if len(data) > 2 and data[2].isdigit():
            stance.depth = int(data[2])
        return stance

    @property
    def key(self) -> str:
        """The key of the stance is the binary representation of its position."""
        return "".join(str(s) for s in self.pos)

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
    modifiers: list[Symbol] = field(default_factory=lambda: [])

    def __repr__(self) -> str:
        return self.content or "Empty symbol"

    def is_pusher(self) -> bool:
        """Checks if the token acts as a pusher."""
        base = self.base.aclass == "Embedder"
        quality = self.base.quality in (None, 0)
        return min(base, quality)

    def is_popper(self) -> bool:
        """Checks if the token acts as a popper."""
        base = self.base.aclass == "Embedder"
        quality = self.base.quality in (None, 1)
        return min(base, quality)

    def is_wild(self, lv: int = 0) -> bool:
        return False

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


@dataclass
class Symbol:
    """A holder for elementary etic units of the language."""

    content: str = str()
    acat: str = str()
    asubcat: str = str()
    aclass: str = str()
    level: int = 0
    quality: int = 0
    index: int = 0
    order: int = 0

    def __repr__(self) -> str:
        return self.content
