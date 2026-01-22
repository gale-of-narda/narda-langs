from math import log

from typing import Dict, List, Any, Optional
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
        stripped_string = "".join(masked).strip("".join(to_strip))

        return stripped_string.lower()

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
        for r in range(int(log(len(perms[0])))):
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
class Buffer:
    """A holder for mappings and trees used in the parsing procedure.
    Each level has a list of those with length equal to the number of complexes + 1.
    """

    parsed_string: Optional[str] = None
    mapping: Optional[Any] = None
    tree: Optional[Any] = None


@dataclass
class Dichotomy:
    """A combination of the mask pair, parameters guiding the choice between them,
    and the pointer that records the last choice made.
    """

    d: int = 0
    nb: bool = False
    terminal: bool = False
    rev: Optional[bool] = None
    ret: Optional[bool] = None
    skip: Optional[bool] = None
    split: Optional[bool] = None
    _pointer: Optional[int] = field(default=None, init=False, repr=False)

    def __repr__(self) -> str:
        try:
            return f"{repr(self.masks[0])}—{repr(self.masks[1])}"
        except AttributeError:
            return "(empty dichotomy)"

    @property
    def masks(self) -> List:
        """Returns the masks in the appropriate order."""
        try:
            return [self.left, self.right] if not self.rev else [self.right, self.left]
        except AttributeError:
            return []

    @property
    def pointer(self):
        """Returns which mask was fitted last (possibly None)."""
        return self._pointer

    @pointer.setter
    def pointer(self, value: int | None) -> None:
        """Setting the pointer to the first or second mask activates it
        and deactivates the other mask. Setting it to None deactivates both.

        To activate a mask is to set its position to 0 while setting the position
        of the opposite mask to None and incrementing its rep.

        To deactivate a mask is to set its position to None.
        """
        if value in (0, 1):
            if not self.masks[value].active:
                self.masks[value].pos = 0
            if self.masks[1 - value].active:
                self.masks[1 - value].pos = None
                self.masks[1 - value].rep += 1
        elif value is None:
            for mask in self.masks:
                if mask.active:
                    mask.pos = None
        else:
            raise ValueError(f"Tried to set an illegal pointer value {value}")

        self._pointer = value

        return

    @property
    def key(self) -> str:
        """The key of the dichotomy is the common left substring of the keys
        of its masks.
        """
        if not self.left or not self.right:
            return None
        else:
            key = [
                self.left.key[:i]
                for i, k in enumerate(self.left.key)
                if self.left.key[:i] == self.right.key[:i]
            ]
        return "".join(key[-1])

    @property
    def num_key(self) -> List[int]:
        """Represents the dichotomy key as a list of integers."""
        num_key = [int(k) for k in self.key]
        return num_key

    def record(self, pos: int, rep: int) -> None:
        target_mask = self.masks[self.pointer]
        other_mask = self.masks[1 - self.pointer]
        self.reset_mask(other_mask)
        if target_mask.rep < rep:
            self.reset_mask(target_mask)
        target_mask.pos = pos
        target_mask.rep = rep
        return


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

    def copy(self, lim: int | None = None) -> Stance:
        """Creates a copy of the stance with pos and rep limited from the right
        by the given index.
        """
        pos = [p for p in self.pos][:lim]
        rep = [r for r in self.rep][:lim]
        depth = self.depth
        new_stance = Stance(pos, rep, depth)
        return new_stance
