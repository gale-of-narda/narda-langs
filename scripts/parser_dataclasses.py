from math import log

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class Alphabet:
    """A holder for the characters used in the language.
    Has functions for removing non-alphabetic and representing alphabetic characters.
    """

    bases: Dict[str, Dict]
    modifiers: Dict[str, Dict[str, List[str]]]
    substitutions: Dict[str, str]

    def _build_dicts(self, level: int) -> None:
        """Adds to the alphabet various dictionaries of characters
        useful on different stages of string parsing.
        """

        def unpack(d) -> Dict:
            # Creates a flat dictionary of characters with all parameters encoded
            items = {}
            for cat in d:
                for subcat in d[cat]:
                    for aclass in d[cat][subcat]:
                        values = d[cat][subcat][aclass]
                        if isinstance(values, List):
                            for q, vals in enumerate(values):
                                for i, val in enumerate(vals):
                                    items[val] = {
                                        "Category": cat,
                                        "Subcategory": subcat,
                                        "Class": aclass,
                                        "Quality": q,
                                        "Index": i,
                                    }
                        else:
                            for i, val in enumerate(values):
                                items[val] = {
                                    "Category": cat,
                                    "Subcategory": subcat,
                                    "Class": aclass,
                                    "Quality": 0,
                                    "Index": i,
                                }
            return items

        # Removing characters not on the given level
        for cat in self.bases["Guiding"]:
            self.bases["Guiding"][cat] = self.bases["Guiding"][cat][level]
        for cat in self.modifiers:
            for subcat in self.modifiers[cat]:
                self.modifiers[cat][subcat] = self.modifiers[cat][subcat][level]

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
        # Replace the special strings as defined in the alphabet
        reps = self.substitutions
        replaced_string = [reps[ch] if ch in reps else ch for ch in st]
        replaced_string = "".join(replaced_string)

        # Erase the non-alphabetic strings from the string
        masked = [ch for ch in replaced_string if ch in self.lookup.keys()]

        prepared_string = "".join(masked).lower()

        return prepared_string

    def symbolize(self, prepared_string: str) -> List[Symbol]:
        """Tranforms the input string into a list of symbols matched
        with the character parameters in the alphabet.
        """
        symbols = []
        for i, ch in enumerate(prepared_string):
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
                if s.aclass == "Embedder":
                    if s.content == self.embedders[0]:
                        g.is_pusher = True
                    if s.content == self.embedders[1]:
                        g.is_popper = True
                graphemes.append(g)
            elif s.acat == "Modifier" and s.aclass == graphemes[-1].aclass:
                graphemes[-1].modifiers.append(s)
        return graphemes

    def get_grapheme(self, st: str) -> Optional[str]:
        """Creates a grapheme based on the given base character."""

        if st in [
            k for k in self.lookup.keys() if self.lookup[k]["Category"] == "Base"
        ]:
            s = Symbol(st, *self.lookup[st].values())
            g = Grapheme(base=s)
            if s.aclass == "Embedder":
                if s.content == self.embedders[0]:
                    g.is_pusher = True
                if s.content == self.embedders[1]:
                    g.is_popper = True
            return g
        else:
            raise ValueError(
                f"Tried to create a grapheme with an illegal character {st}"
            )


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

    def _unravel_term_params(
        self,
    ) -> Tuple[List[List[List[str]]], List[List[List[int]]], List[List[List[int]]]]:
        """Creates perms, revs, dembs for each mask based on the general rules
        as they are defined in more compact form.
        """
        perms, revs, dembs = [self.perms], [self.revs], [self.dembs]
        for r in range(int(log(len(perms[0]), 2))):
            split_perms, split_revs, split_dembs = [], [], []
            for i in range(0, len(perms[-1]), 2):
                rev = min(revs[-1][i : i + 2]) if r < 1 else 0
                demb = max(dembs[-1][i : i + 2])
                perm = perms[-1][i : i + 2]
                split_perms.append("".join(perm[::-1] if rev else perm))
                split_revs.append(rev)
                split_dembs.append(demb)
            perms.append(split_perms)
            revs.append(split_revs)
            dembs.append(split_dembs)
        return perms, revs, dembs


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
    """A holder for a feature with all the parameters that describe it.
    Ctype is None for untyped, str for typed features.
    """

    ctype: str | None
    pos: List[int]
    rep: List[int]
    cpos: List[int]
    crep: List[int]
    content_class: str
    priority: int
    index: int
    function_name: str
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
    is_pusher: bool = False
    is_popper: bool = False

    def __repr__(self) -> str:
        return self.content or "Empty symbol"

    def __str__(self) -> str:
        return repr(self)

    @property
    def content(self) -> str:
        """How the grapheme appears in writing."""
        base = str(self.base)
        mods = "".join([m.content for m in self.modifiers])
        return base + mods

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

    @property
    def complex_role(self) -> Optional[str]:
        """Whether the grapheme is a pusher or a popper."""
        if self.base.aclass == "Embedder":
            if self.base.quality == 0:
                return "Pusher"
            elif self.base.quality == 1:
                return "Popper"
        return None


@dataclass
class Symbol:
    """A holder for elementary etic units of the language."""

    content: str = str()
    acat: str = str()
    asubcat: str = str()
    aclass: str = str()
    quality: int = 0
    index: int = 0
    order: int = 0

    def __repr__(self) -> str:
        return self.content
