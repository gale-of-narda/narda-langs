from typing import List, Any, Optional
from dataclasses import dataclass, field

from scripts.parser_entities import Tree, Mapping


@dataclass
class Alphabet:
    content: Any
    separators: Any
    breakers: Any
    embedders: Any
    equivalents: Any


@dataclass
class GeneralRules:
    struct: Any
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
    mapping: Optional[Mapping] = None
    tree: Tree = Tree([0])


@dataclass
class Stance:
    pos: List[int] = field(default_factory=lambda: [])
    rep: List[int] = field(default_factory=lambda: [])
    depth: int = 0

    def __repr__(self):
        pos = "".join([str(p) for p in self.pos])
        rep = "".join([str(r) for r in self.rep])
        return f"[{pos}|{rep}|{self.depth}]"

    def copy(self, lim: int | None = None) -> Stance:
        pos = [p for p in self.pos][:lim]
        rep = [r for r in self.rep][:lim]
        depth = self.depth
        new_stance = Stance(pos, rep, depth)
        return new_stance
