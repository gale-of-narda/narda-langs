from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Alphabet:
    content: Dict[str]
    equivalents: Dict[str]
    wildcards: List[str]
    separators: List[str]
    breakers: List[List[str]]
    embedders: List[List[str]]


@dataclass
class GeneralRules:
    struct: List[List[int]]
    heads: List[List[int]]
    rets: List[List[int]]
    skips: List[List[int]]
    splits: List[List[int]]
    revs: List[List[int]]
    demb: List[List[int]]
    perms: List[List[str]]
    lemb: List[List[List[int]]]


@dataclass
class SpecialRules:
    tperms: List[List[List[str]]]
    tneuts: List[List[List[str]]]


@dataclass
class Buffer:
    parsed_string: Optional[str] = None
    mapping: Optional[Any] = None
    tree: Optional[Any] = None


@dataclass
class Dichotomy:
    d: int = 0
    nb: bool = False
    terminal: bool = False
    rev: Optional[bool] = None
    ret: Optional[bool] = None
    skip: Optional[bool] = None
    split: Optional[bool] = None
    pointer: Optional[int] = None

    def __repr__(self) -> str:
        try:
            return f"{repr(self.masks[0])}—{repr(self.masks[1])}"
        except AttributeError:
            return "(empty dichotomy)"

    @property
    def masks(self) -> List:
        if not self.left or not self.right:
            return []
        return [self.left, self.right] if not self.rev else [self.right, self.left]

    @property
    def key(self) -> str:
        if not self.left or not self.right:
            return None
        else:
            key = [
                self.left.key[:i]
                for i, k in enumerate(self.left.key)
                if self.left.key[:i] == self.right.key[:i]
            ]
        return "".join(key[-1])

    def set_masks(self, masks: List) -> None:
        self.left = masks[0]
        self.right = masks[1]
        return


@dataclass
class Stance:
    pos: List[int] = field(default_factory=lambda: [])
    rep: List[int] = field(default_factory=lambda: [])
    depth: int = 0

    def __repr__(self) -> str:
        pos = "".join([str(p) for p in self.pos])
        rep = "".join([str(r) for r in self.rep])
        return f"[{pos}|{rep}|{self.depth}]"

    def copy(self, lim: int | None = None) -> Stance:
        pos = [p for p in self.pos][:lim]
        rep = [r for r in self.rep][:lim]
        depth = self.depth
        new_stance = Stance(pos, rep, depth)
        return new_stance
