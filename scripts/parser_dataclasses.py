from typing import Dict, Any, Optional
from dataclasses import dataclass

from scripts.parser_entities import Tree


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


@dataclass
class SpecialRules:
    tperms: Any
    tneuts: Any
    lemb: Any
    demb: Any


@dataclass
class Buffer:
    tree: Tree = Tree([0])
    pst: Optional[str] = None
    mapping: Optional[Dict] = None
