from math import log

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Alphabet:
    """A holder for the characters used in the language.
    Has functions for removing non-alphabetic and representing alphabetic characters.
    """

    bases: Dict[str, Dict]
    modifiers: Dict[str, Dict[str, List[str]]]
    substitutions: Dict[str, str]

    def _build_dicts(self) -> None:
        """Adds to the alphabet various dictionaries of characters
        useful on different stages of string parsing.
        """

        def unpack(d) -> Dict:
            # Creates a flat dictionary of characters with all parameters encoded
            items = {}
            for cat in d:
                for subcat in d[cat]:
                    for aclass in d[cat][subcat]:
                        for level, values in enumerate(d[cat][subcat][aclass]):
                            if isinstance(values, List):
                                for q, vals in enumerate(values):
                                    for i, val in enumerate(vals):
                                        items[val] = {
                                            "Category": cat,
                                            "Subcategory": subcat,
                                            "Class": aclass,
                                            "Level": level,
                                            "Quality": None if val in items else q,
                                            "Index": i,
                                        }
                            else:
                                for i, val in enumerate(values):
                                    items[val] = {
                                        "Category": cat,
                                        "Subcategory": subcat,
                                        "Class": aclass,
                                        "Level": level,
                                        "Quality": 0,
                                        "Index": i,
                                    }
            return items

        self.content = self.bases["Content"]
        self.wildcards = self.bases["Guiding"]["Wildcard"]
        self.separators = self.bases["Guiding"]["Separator"]
        self.embedders = self.bases["Guiding"]["Embedder"]
        self.breakers = self.modifiers["Breaker"]

        d = {"Base": self.bases, "Modifier": self.modifiers}
        self.lookup = unpack(d)

        return

    def prepare(self, st: str) -> str:
        """Removes non-alphabetic characters and makes the replacements."""
        lst = st.lower()
        # Replace the special strings as defined in the alphabet
        reps = self.substitutions
        replaced_string = [reps[ch] if ch in reps else ch for ch in lst]
        replaced_string = "".join(replaced_string)

        # Erase the non-alphabetic strings from the string
        masked = [ch for ch in replaced_string if ch in self.lookup.keys()]

        prepared_string = "".join(masked)

        return prepared_string

    def symbolize(self, prepared_string: str, lv: int) -> List[Symbol]:
        """Tranforms the input string into a list of symbols matched
        with the character parameters in the alphabet.
        """
        symbols = []
        sep = self.separators[lv]
        split_string = prepared_string.split() if sep else prepared_string
        for i, ch in enumerate(split_string):
            if ch in self.lookup:
                d = self.lookup[ch]
                symbol = Symbol(ch, *d.values(), i)
                symbols.append(symbol)
        return symbols

    def graphemize(self, symbols: List[Symbol]) -> List[Grapheme]:
        graphemes = []
        for s in symbols:
            if s.acat == "Base":
                g = Grapheme(base=s)
                graphemes.append(g)
            elif s.acat == "Modifier" and s.aclass == graphemes[-1].aclass:
                graphemes[-1].modifiers.append(s)
        return graphemes

    def get_grapheme(self, st: str) -> Grapheme:
        """Creates a grapheme based on the given base character."""
        if st in [
            k for k in self.lookup.keys() if self.lookup[k]["Category"] == "Base"
        ]:
            s = Symbol(st, *self.lookup[st].values())
            g = Grapheme(base=s)
            return g
        raise ValueError(f"Tried to get a grapheme with the illegal character {st}")


@dataclass
class GeneralRules:
    """A holder for the general rules of the language, which deal with the mapping
    of elements to masks in relation to dichotomies.
    """

    struct: List[List[int]]
    heads: List[List[int]]
    rets: List[List[int]]
    skips: List[List[int]]
    splits: List[List[int]]
    revs: List[List[int]]
    dembs: List[List[int]]
    perms: List[List[str]]
    lembs: List[List[List[int]]]
    wilds: List[List[List[int]]]

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

    tperms: List[List[List[str]]]
    tneuts: List[List[List[str]]]


@dataclass
class Dialect:
    """A holder for the parameters that guide the interpretation of node features."""

    ctypes: List[Dict[str, str]]
    untyped: List[Feature]
    typed: List[Feature]

    def get_feature(
        self,
        index: int,
        content_class: str,
        stance: Stance,
        cstance: Optional[Stance] = None,
        ctype: Optional[str] = None,
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
    pos: List[int]
    rep: List[int]
    cpos: List[int]
    crep: List[int]
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

    pos: List[int] = field(default_factory=lambda: [])
    rep: List[int] = field(default_factory=lambda: [])
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
class Grapheme:
    """A holder for elementary emic units of the language."""

    base: Symbol
    modifiers: List[Symbol] = field(default_factory=lambda: [])

    def __repr__(self) -> str:
        return self.content or "Empty symbol"

    def __str__(self) -> str:
        return repr(self)

    def is_pusher(self, lv: int) -> bool:
        """Checks if the grapheme acts as a pusher at the given level."""
        base = self.base.aclass == "Embedder"
        level = self.base.level == lv
        quality = self.base.quality in (None, 0)
        return min(base, level, quality)

    def is_popper(self, lv: int) -> bool:
        """Checks if the grapheme acts as a popper at the given level."""
        base = self.base.aclass == "Embedder"
        level = self.base.level == lv
        quality = self.base.quality in (None, 1)
        return min(base, level, quality)

    @property
    def content(self) -> str:
        """How the grapheme appears in writing."""
        base = str(self.base)
        mods = "".join([m.content for m in self.modifiers])
        return base + mods
    
    @property
    def lit(self) -> str:
        """The string representation of the grapheme's base used for matching."""
        return str(self.base).upper()

    @property
    def aclass(self) -> str:
        """The class of the grapheme is that of its base symbol."""
        return self.base.aclass

    @property
    def index(self) -> str:
        """The index of the grapheme is that of its base symbol."""
        return self.base.index

    @property
    def order(self) -> int:
        """The order of the grapheme is the minimum of those of its symbols."""
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
    
    def __str__(self) -> str:
        return self.content
