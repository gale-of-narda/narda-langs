from math import log

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class Alphabet:
    """A holder for the characters used in the language.
    Has functions for removing non-alphabetic and representing alphabetic characters.
    """

    content: Dict[str]
    equivalents: Dict[str]
    wildcards: List[str]
    separators: List[str]
    breakers: List[List[str]]
    embedders: List[List[str]]
    bclasses: List[str]

    def prepare(self, st: str) -> str:
        """Removes non-alphabetic characters and makes the replacements."""
        # Replace the special strings as defined in the alphabet
        reps = self.equivalents
        replaced_string = [reps[ch] if ch in reps else ch for ch in st]
        replaced_string = "".join(replaced_string)

        # Erase the non-alphabetic strings from the string
        content = "".join(self.content.values())
        separators = "".join(self.separators)
        breakers = "".join(self.breakers)
        embedders = "".join(self.embedders)
        full_mask = content + separators + breakers + embedders
        to_strip = separators
        masked = [ch for ch in replaced_string if ch in full_mask]

        # Strip separators from both ends
        ss = "".join(masked).strip("".join(to_strip))

        # Remove breakers that follow characters not from the breaking classes
        for i, ch in enumerate(ss):
            if i > 0 and ch in breakers:
                if self.represent(ss[i - 1]) not in self.bclasses:
                    ss = ss[:i] + ss[i + 1 :]

        prepared_string = ss.lower()

        return prepared_string

    def represent(self, ch: str, level: int = 0) -> str:
        """Repalces the input character with its alphabetic representation."""
        if level != 0:
            return ch.upper()
        else:
            content = self.content.items()
            breakers = list(self.breakers)
            embedders = list(self.embedders)
            if any(ch in nc for nc in breakers + embedders):
                return ch
            for key, val in content:
                if ch in val:
                    return key
            raise ValueError(f"No representation for '{ch}' on level {level}")

    def get_index(self, ch: str) -> Tuple(str, int) | None:
        """Searches for the given character in content classes and returns
        the index of first occurence, or None if nothing is found.
        """
        for cc in self.content:
            index = self.content[cc].find(ch)
            if index > 0:
                return cc, index
        return None


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

    def _unravel_term_params(self) -> None:
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

    ctypes: List[Dict[str]]
    untyped: List
    typed: List

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
            if all([f.index == index, f.pos == stance.pos, f.rep == stance.rep]):
                if all([not cstance, not f.cpos, not f.crep]) or all(
                    [f.cpos == cstance.pos and f.crep == cstance.rep]
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
        string = f"{self.function_name}: {self.argument_name}"
        return string


@dataclass
class Stance:
    """A representation of the mapping between the element and the mask."""

    pos: List[int] = field(default_factory=lambda: [])
    rep: List[int] = field(default_factory=lambda: [])
    depth: int = 0

    def __repr__(self) -> str:
        pos = "".join([str(p) for p in self.pos])
        rep = "".join([str(r) for r in self.rep])
        return f"[{pos}|{rep}|{self.depth}]"

    @staticmethod
    def decode(st: str) -> Stance:
        """Returns a Stance object with pos, rep, and depth
        based on the provided string.
        """
        stance = Stance()
        data = st.strip("[]").split("|")
        if len(data) > 0:
            if data[0].isnumeric():
                stance.pos = [int(d) for d in data[0]]
        if len(data) > 1:
            if data[1].isnumeric():
                stance.rep = [int(d) for d in data[1]]
        if len(data) > 2:
            if data[2].isnumeric():
                stance.depth = int(data[2])
        return stance

    @property
    def key(self) -> str:
        """The key of the stance is the binary representation of its position."""
        return "".join([str(s) for s in self.pos])

    def copy(self, lim: int | None = None) -> Stance:
        """Creates a copy of the stance with pos and rep limited from the right
        by the given index.
        """
        pos = [p for p in self.pos][:lim]
        rep = [r for r in self.rep][:lim]
        depth = self.depth
        new_stance = Stance(pos, rep, depth)
        return new_stance
