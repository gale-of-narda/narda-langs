from typing import Dict, Tuple, List, Callable, Optional, Any
from collections import deque

from scripts.parser_dataclasses import Stance


class Mapping:
    """A temporary structure holding elements as they are recorded.
    When a list of elements of depth > 0 is fully recorded, it is
    replaced by an element of a higher level.
    """

    def __init__(self, level: int) -> None:
        self.level = level
        self.heads: List[int] = []
        self.elems: List[Element] = []
        self.cur_depth: int = 0
        self.breaks: int = 0
        self.stack = self.elems
        self.holder = None
        return

    def record_element(self, e: Element) -> None:
        """Adds an element to the current iterator."""
        self.stack.append(e)
        return

    def push(self) -> None:
        """Increases depth by one, adds a list and sets it to stack."""
        self.cur_depth += 1
        self.stack.append([])
        self.holder = self.stack
        self.stack = self.stack[-1]
        return

    def pop(self) -> None:
        """Decreases depth by one and collapses the current list into an element."""
        if self.cur_depth == 0:
            print("Tried to set a negative depth")
            return
        self.cur_depth -= 1
        self.stack = self.holder
        self.stack[-1] = Element(self.stack[-1], Stance(), self.level)
        head = self.heads[min(self.cur_depth + 1, len(self.heads) - 1)]
        self.stack[-1].set_head(head)
        self.stack[-1].stance.depth = self.cur_depth
        return

    def enumerate_elems(
        self, num_key: List[int], d: Optional[int] = None
    ) -> Dict | List[int]:
        """Returns a list of stack indices of elements that conform
        to the given key. If d is given, arranges them into a dict of lists
        where the keys are the reps before d.
        """
        matches = {} if d else []
        for i, e in enumerate(self.stack):
            if e.stance.pos[: len(num_key)] == num_key:
                if not d:
                    matches.append(i)
                else:
                    slot = "".join([str(r) for r in e.stance.rep[:d]])
                    if slot not in matches:
                        matches[slot] = [i]
                    else:
                        matches[slot].append(i)
        return matches


class Dichotomy:
    """A combination of the mask pair, parameters guiding the choice between them,
    and the pointer that records the last choice made.
    """

    def __init__(self, d: int = 0, nb: bool = False) -> None:
        self.d: int = d
        self.nb: bool = nb
        self.terminal: bool = False
        self.rev: Optional[bool] = None
        self.ret: Optional[bool] = None
        self.skip: Optional[bool] = None
        self.split: Optional[bool] = None
        self._pointer: Optional[int] = None
        return

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
        """Returns the number of mask was fitted last (possibly None)."""
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


class Mask:
    """A string to be used as the mask for the candidate character undergoing
    dichotomy resolution during parsing.

    Consists of positions to be moved
    through with a cursor, where each position can be optional and/or
    allow for several types of characters rather than one.
    """

    def __init__(
        self, premask: str, rank: int = 0, num: int = 0, depth: int = 0
    ) -> None:
        self.literals, self.optionals = self._decode(premask)
        self.rank, self.num, self.depth = rank, num, depth
        self.tneuts = None
        self.rev = None
        self.lemb = None
        self.demb = None
        self.pos = None
        self.rep = 0
        self.freeze = False
        return

    def __repr__(self) -> str:
        out = ""
        lits = ["".join(lit) for lit in self.literals]
        for n, lit in enumerate(lits):
            opt = "?" if self.optionals[n] else ""
            brd = f"({lit})" if len(lit) > 1 else lit
            und = [f"{b}̲" if self.pos == n and b not in "()" else b for b in brd]
            out += opt + "".join(und)
        return f"'{out}'"

    @property
    def key(self) -> str:
        """The key of the mask is the binary representation of its number."""
        key = f"{self.num:b}".rjust(self.rank, "0")
        return key

    @property
    def num_key(self) -> List[int]:
        """Represents the dichotomy key as a list of integers."""
        num_key = [int(k) for k in self.key]
        return num_key

    @property
    def active(self) -> bool:
        """A mask is active if its current position is not None."""
        return self.pos is not None

    def _decode(self, premask: str) -> Tuple[List[List[str]], List[bool]]:
        """Decodes the terminal permissions defined in the general rules
        to transform them into masks.
        """
        optional, bracketed = False, False
        literals, optionals, group = [], [], []

        for c in premask:
            match c:
                case "?":
                    optional = True
                case "(":
                    bracketed = True
                case ")":
                    optionals.append(optional)
                    bracketed, optional = False, False
                    literals += [group]
                    group = []
                case _:
                    group += c
                    if not bracketed:
                        optionals.append(optional)
                        optional = False
                        literals += [group]
                        group = []

        return literals, optionals

    def move(self, step: int, inplace: bool = False) -> Tuple[int, int]:
        """Changes the position of the cursor in the mask for the given number
        of steps forward. Loops back and increases the repetition counter
        if the mask is cyclical.
        """
        pos = self.pos or 0

        ln = len(self.literals)
        new_pos = (pos + step) % ln
        new_rep = self.rep + (pos + step) // ln

        if inplace:
            self.pos = new_pos
            self.rep = new_rep
            return None
        else:
            return (new_pos, new_rep)

    def match(self, rep_str: str, pos: int = 0, ignore_pos: bool = False) -> bool:
        """Checks if the given string fits the element at the given position."""
        # If freeze is True, fitting to the mask is forbidden
        if self.freeze:
            return False
        if ignore_pos:
            return any([rep_str in m for pos in self.literals for m in pos])
        else:
            target_pos = self.move(pos)[0]
            return any([rep_str in m for m in self.literals[target_pos]])

    def subtract(self, pos_delta: int = 0, rep_delta: int = 0) -> None:
        """Subtracts the given number of rep and pos, limited by zero."""
        self.rep = max(self.rep - rep_delta, 0)
        if self.pos:
            self.pos = max(self.pos - pos_delta, 0)
        return


class Tree:
    """A dichotomic tree defined by the given structure."""

    def __init__(self, struct: List[int]) -> None:
        self.struct = struct
        self.root = Node()
        self.nodes = [self.root]
        self.traverse(self.root, self._populate)
        self.perms = None
        self.ctype = None

    def __repr__(self) -> str:
        return f"T{self.struct}"

    def __str__(self) -> str:
        st = f"{repr(self)}: {self.ctype} \n"
        st += self._draw(self.root)
        return st

    def _populate(self, node: Node, i: int):
        """Realizes the defined structure by creating the appropriate number of nodes
        and setting the parent-child connections between them.
        """
        r = node.rank
        st = [0] + self.struct
        node.num = i
        node.ranknum = node.num - sum([2**s for s in st[: node.rank]])
        node.set_key(self.struct)
        if r < len(self.struct) and self.struct[r] > 0:
            for sibnum in range(0, 2 ** self.struct[r]):
                n = Node(r + 1)
                n.sibnum = sibnum
                n.parent = node
                node.children.append(n)
                self.nodes.append(n)
        return

    def _draw(self, node: Node, depth: int = 0, header: str = "└", top=False) -> str:
        """Draws the structure of the tree to be printed."""
        anc = node.parent
        ancs_last = []
        while anc is not None:
            is_last = anc.parent is None or anc is anc.parent.children[-1]
            ancs_last.append(is_last)
            anc = anc.parent
        ancs_last = list(reversed(ancs_last))

        if len(ancs_last) < depth:
            ancs_last = [True] * (depth - len(ancs_last)) + ancs_last

        prefix = "".join("  " if is_last else "│ " for is_last in ancs_last)

        last_sib = node.parent is None or node is node.parent.children[-1]
        arrow = header if depth == 0 or top or last_sib else "├"

        st = prefix + arrow + "─" + repr(node) + "\n"

        for ch in node.children:
            # if ch.content:
            st += self._draw(ch, depth + 1)
        for cd in node.compounds:
            # if cd.content:
            st += self._draw(cd, depth, "⤷", top=True)
        for cx in node.complexes:
            st += prefix + "  " + "⤷─" + repr(cx) + "\n"
            st += cx._draw(cx.root, depth + 2, top=True)

        return st

    def _get_subtree(self, target: Optional[int] = None) -> Tree:
        """Creates a structural copy of the tree and returns either that copy
        or its node of the given number.
        """
        subtree = Tree(self.struct)
        if target is None:
            return subtree
        else:
            return subtree.nodes[target]

    def traverse(self, subroot: Node, fun: Callable) -> Optional[Any]:
        """Applies the given function to the nodes of the subtree
        that originates from the given subroot.
        """
        queue = deque([subroot])
        i = 0
        while queue:
            node = queue.popleft()
            res = fun(node, i)
            if res is not None:
                return res
            for ch in node.children:
                queue.append(ch)
            i += 1
        return

    def embed_compound(self, node: Node) -> None:
        """Adds a copy of the subtree originating from the node of the given number
        to the mapped compounds list of the node.
        """
        new_node = self._get_subtree(node.num)
        node.compounds.append(new_node)
        return

    def embed_complex(self, node: Node) -> None:
        """Adds a copy of the subtree originating from the root of the tree
        to the mapped complexes list of the node.
        """
        new_tree = self._get_subtree()
        node.complexes.append(new_tree)
        return

    def get_nodes(
        self, stance: Optional[Stance] = None, upstream: bool = False
    ) -> List[Node] | Node:
        """Returns the node addressed by the given stance.
        If upstream = True, also returns its ancestors.
        """
        if stance is None:
            stance = Stance()
        cursor, node, out = 0, self.root, [self.root]
        pos, comps = "".join([str(s) for s in stance.pos]), stance.rep
        struct_sums = [sum(self.struct[: i + 1]) for i, s in enumerate(self.struct)]
        struct_to_loop = [s for s in struct_sums if s <= len(pos)]
        for s in struct_to_loop:
            node = node.children[int(pos[cursor : cursor + s], 2)]
            comp = comps[cursor : cursor + s][-1]
            if comp > 0:
                node = node.compounds[comp - 1]
            cursor += s
            out.append(node)
        return out if upstream else out[-1]

    def set_element(self, e: Element, set_all: bool = True) -> None:
        """Maps the element to the node addressed by its stance.
        If set_all is True, also maps it continuously to parent nodes
        all the way up to the root.
        """
        node = self.get_nodes(e.stance)
        if set_all:
            cursor = 0
            struct = [0] + self.struct[: len(e.stance.pos)]
            for s in struct[: len(e.stance.pos) + 1]:
                node = self.get_nodes(e.stance.copy(cursor + s))
                node.map_element(e)
                cursor += s
        return


class Node:
    """A dichotomic tree node. Connected to its parent and children,
    defined by rank, number, and key.

    Elements can be mapped to nodes either directly as content
    or as a compound or complex elements (each type of mapped element
    is stored separately).
    """

    def __init__(self, rank: int = 0) -> None:
        self.parent: Node | None = None
        self.rank: int = rank
        self.num: int = 0
        self.ranknum: int = 0
        self.sibnum: int = 0
        self.content: List[str] = []
        self.children: List[Node] = []
        self.compounds: List[Node] = []
        self.complexes: List[Tree] = []
        self.key: str = str()

    def __repr__(self) -> str:
        content = [s for s in self.content if isinstance(s, Element)]
        return f"N({self.num}): {''.join(str(s) for s in content)}"

    def __str__(self) -> str:
        return repr(self)

    def set_key(self, struct: List[int]):
        """Computes and saves the key of the node in a binary format."""
        layer = 1
        layers = [1] + [(layer := layer * 2**x) for x in struct]
        d = self.num - sum(layers[: self.rank])
        key = f"{d:b}".rjust(sum(struct[: self.rank]), "0")
        self.key = key if self.num > 0 else str()
        return

    def map_element(self, e: Element) -> None:
        """Adds the given element to the content of the node."""
        self.content.append(e)
        return


class Element:
    """A language element is an alphabetic string assigned a stance
    of a certain level.
    """

    def __init__(
        self, content: str | List[Element], stance: Stance, level: int
    ) -> None:
        self.content = content
        self.stance = stance
        self.level = level
        self.rep: str = str()
        self.complex: bool = isinstance(self.content, List)
        self.head = self
        return

    def __repr__(self) -> str:
        if self.complex:
            return repr(self.head)
        else:
            return self.content

    def __str__(self) -> str:
        return str(repr(self))

    @property
    def num(self) -> int:
        """Returns the number represented in the binary form by the stance."""
        key = "".join([str(s) for s in self.stance.pos])
        num = int(key, 2)
        return num

    def set_head(self, num: int) -> None:
        """Finds the content element with the given binary number
        and sets it as the head.
        """
        for e in self.content:
            if int("".join([str(p) for p in e.stance.pos]), 2) == num:
                self.head = e
                return
        raise Exception(f"Couldn't find head at node {num}")
