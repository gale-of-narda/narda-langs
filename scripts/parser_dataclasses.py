from typing import List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Alphabet:
    content: Any
    wildcards: Any
    separators: Any
    breakers: Any
    embedders: Any
    equivalents: Any


@dataclass
class GeneralRules:
    struct: Any
    heads: Any
    rets: Any
    skips: Any
    splits: Any
    perms: Any
    revs: Any
    lemb: Any
    demb: Any


@dataclass
class SpecialRules:
    tperms: Any
    tneuts: Any


@dataclass
class Buffer:
    parsed_string: Optional[str] = None
    mapping: Optional[Any] = None
    tree: Optional[Any] = None


@dataclass
class Dichotomy:
    nb: bool
    cursor: Optional[int] = None

    @property
    def rev(self) -> bool:
        return bool(min(self.left.rev, self.right.rev))

    @property
    def masks(self) -> List:
        return [self.left, self.right] if not self.rev else [self.right, self.left]
    
    def __repr__(self) -> str:
        try:
            return f"{repr(self.masks[0])}—{repr(self.masks[1])}"
        except AttributeError:
            return "(empty dichotomy)"

    def set_masks(self, masks: List) -> None:
        self.left = masks[0]
        self.right = masks[1]
        return

    def switch(self) -> int:
        self.cursor = 1 - self.cursor if self.cursor else 0
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
