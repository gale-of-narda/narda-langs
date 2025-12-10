from typing import List, Any, Optional
from dataclasses import dataclass

from scripts.parser_entities import Tree

type Stance = (List[int], List[int])

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
class Mapping:
    chars: List[str]
    stances: List[Stance]

@dataclass
class Buffer:
    parsed_string: Optional[str] = None
    mapping: Optional[Mapping] = None
    tree: Tree = Tree([0])