from math import log

from typing import Dict, Tuple, List, Optional

from scripts.parser_dataclasses import Stance, Token, Feature


class Mapping:
    """A holder for the current stacks of elements and the parameter values
    involved in parsing.
    """

    def __init__(self, levels: range):
        self.cur_breaks = [[0] for lvl in levels]
        self.cur_dpt = [0 for lvl in levels]
        self.cur_bdr = [0 for lvl in levels]
        self.elems = [[] for lvl in levels]
        
    def get_interval(self, lvl: int) -> list[Element]:
        """Returns the interval on the given level consisting of elements
        that are yet to be wrapped into an element of the higher level.
        """
        elems = self.elems[lvl][self.cur_bdr[lvl] :]
        return elems

    def update_interval(self, lvl: int) -> None:
        """Moves the interval border on the given level to the end of its
        element list.
        """
        self.cur_bdr[lvl] = len(self.elems[lvl])
        return

    def get_stack(self, lvl: int = -1, interval: bool = False) -> list[Element]:
        """Returns the list of elements at the given level
        to which the next element should be appended.
        """
        stack = self.get_interval(lvl) if interval else self.elems[lvl]
        while len(stack) > 0 and isinstance(stack[-1], list):
            stack = stack[-1]
        return stack

    def enumerate_elems(
        self, num_key: list[int], elems: list[Element], d: int
    ) -> Dict[str, list[int]]:
        """Returns a list of indices of the given list of elements that conform
        to the given key, arranging them into a dict of lists where the keys are
        the reps before d.
        """
        matches = {}
        for i, e in enumerate(elems):
            if e.stance.pos[: len(num_key)] == num_key:
                slot = "".join(str(r) for r in e.stance.rep[:d])
                if slot not in matches:
                    matches[slot] = [i]
                else:
                    matches[slot].append(i)
        return matches

class Dichotomy:
    """A combination of the mask pair, parameters guiding the choice between them,
    and the pointer that records the last choice made.
    """

    def __init__(self, level: int = 0, d: int = 0, nb: bool = False) -> None:
        self.level: int = level
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
    def masks(self) -> List[Mask]:
        """Returns the masks in the appropriate order."""
        try:
            return [self.left, self.right] if not self.rev else [self.right, self.left]
        except AttributeError:
            return []

    @property
    def pointer(self) -> Optional[int]:
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
    def key(self) -> Optional[str]:
        """The key of the dichotomy is the common left substring of the keys
        of its masks.
        """
        if not self.left or not self.right:
            return None
        key = [
            self.left.key[:i]
            for i, k in enumerate(self.left.key)
            if self.left.key[:i] == self.right.key[:i]
        ]
        return "".join(key[-1]) if key else ""

    @property
    def num_key(self) -> List[int]:
        """Represents the dichotomy key as a list of integers."""
        num_key = [int(k) for k in self.key]
        return num_key

    @property
    def depth(self) -> int:
        """The depth of the dichotomy is the greatest depth of its masks."""
        return max([m.depth for m in self.masks])

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
        self.wild = None
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

    def compare(
        self,
        e: Element,
        split: bool,
        force_mov: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Produces the movement for the mask required to fit the element
        with its current stance. If no fit is possible, return None.
        """
        # First check if a complex element can be fit
        if all([not e.molar, self.demb is not None, self.demb != -1]):
            if e.stance.depth > self.demb - 1:
                return None

        # Bypass positional matching if split-set fitting is applied
        if split:
            if self.match(e, ignore_pos=True):
                return (self.pos, self.rep)
            else:
                return None

        # Otherwise start going through the literals one-by-one
        singular = len(self.literals) == 1
        incr = 1 if (singular or force_mov) and self.active else 0
        while any(
            (
                # Current string, unless the mask is singular and was already fit
                incr == 0 and not (singular and self.active),
                # Next string, if the mask is singular or was already fit
                incr == 1 and (singular or force_mov or self.active),
                # Next string(s), unless a non-optional string is getting skipped
                incr > 0 and not singular and self.optionals[self.move(incr - 1)[0]],
            )
        ):
            if self.match(e, incr):
                mov = self.move(incr)
                # Check that we aren't adding compounds beyond the restriction
                if mov[1] <= (self.lemb or 0) or self.lemb == -1:
                    return mov
            incr += 1

        return None

    def move(self, step: int) -> Optional[Tuple[int, int]]:
        """Changes the position of the cursor in the mask for the given number
        of steps forward. Loops back and increases the repetition counter
        if the mask is cyclical.
        """
        pos = self.pos or 0

        ln = len(self.literals)
        new_pos = (pos + step) % ln
        new_rep = self.rep + (pos + step) // ln

        return (new_pos, new_rep)

    def match(self, e: Element, pos: int = 0, ignore_pos: bool = False) -> bool:
        """Checks if the given string fits the element at the given position."""
        # If freeze is True, fitting to the mask is forbidden
        aclass, lit = e.header.content.base.aclass, e.header.content.lit
        if self.freeze:
            return False
        if aclass == "Wildcard":
            return self.wild
        if ignore_pos:
            by_class = any(aclass in m for lits in self.literals for m in lits)
            by_val = any(lit in m for lits in self.literals for m in lits)
            return max(by_class, by_val)
        else:
            target_pos = self.move(pos)[0]
            by_class = any(aclass in m for m in self.literals[target_pos])
            by_val = any(lit in m for m in self.literals[target_pos])
            return max(by_class, by_val)

    def subtract(self, pos_delta: int = 0, rep_delta: int = 0) -> None:
        """Subtracts the given number of rep and pos, limited by zero."""
        self.rep = max(self.rep - rep_delta, 0)
        if self.pos is not None:
            self.pos = max(self.pos - pos_delta, 0)
        return


class Element:
    """A language element is an alphabetic string assigned a stance
    of a certain level.
    """

    def __init__(
        self,
        content: Token | List[Token],
        stance: Optional[Stance] = None,
        level: int = 0,
    ) -> None:
        self.content: Token | List[Token] | Element | List[Element] = content
        self.stance: Stance | None = stance
        self.level: int = level
        self.molar: bool = not isinstance(self.content, List)
        self.head: Element = self
        return

    def __repr__(self) -> str:
        return self._represent()

    def _represent(self, top: bool = True) -> str:
        if self.molar:
            return str(self.content)
        else:
            if top:
                out = ""
                for c in self.content:
                    out += c._represent(top=False)
                    if c is self.head:
                        out += "̲"
                return out
            else:
                return self.head._represent(top=False)

    @property
    def view(self):
        """String representation of the tokens in the element's content."""
        return "".join([str(g) for g in self.content])

    @property
    def num(self) -> int:
        """Returns the number represented in the binary form by the stance."""
        key = "".join(str(s) for s in self.stance.pos)
        return int(key, 2)

    @property
    def order(self) -> int:
        """The order of the element is that of its head's content token."""
        return self.head.content.order

    @property
    def header(self) -> Element:
        """The lowest head of the element."""
        source = self
        while source.head is not source:
            source = source.head
        return source

    @property
    def preheader(self) -> Element:
        """The lowest head of the element that is still of the same level."""
        source = self
        while source.head is not source and source.head.level == source.level:
            source = source.head
        return source

    def set_head(self, nums: int | list[int], fallback: bool = False) -> bool:
        """Finds the content element with the given binary number
        and sets it as the head.
        """
        # Simple elements always have themselves as heads
        if self.molar:
            return False
        # Try each permitted stance one by one until the first fitting element
        for num in nums if isinstance(nums, list) else [nums]:
            for e in self.content:
                if int(e.stance.key or "0", 2) == num:
                    self.head = e
                    return True
        # If no elements are found, set the first one
        if fallback:
            self.head = self.content[0]
            return True
        else:
            return False


class Tree:
    """A dichotomic tree defined by the given structure."""

    def __init__(self, struct: List[int], level: int = 0, depth: int = 0) -> None:
        self.struct: List[int] = struct
        self.level: int = level
        self._depth: int = depth
        self.root: Node = Node()
        self.nodes: List[Node] = [self.root]
        self.perms = None
        self.ctype = None
        self.stance = None
        self._populate(self.root)

    def __repr__(self) -> str:
        return f"T{self.struct}"

    def __str__(self) -> str:
        st = f"{repr(self)}: {self.ctype or 'Undefined composition type'}"
        return st

    @property
    def working_string(self) -> str:
        """String representation of the tokens in elements mapped to the tree."""
        tokens = []
        comps = [e for c in self.root.compounds for e in c.content]
        for e in self.root.content + comps:
            tokens.append(e.head.content)
        return "".join([str(g) for g in tokens])

    def _populate(self, node: Node) -> None:
        """Realizes the defined structure by creating the appropriate number of nodes
        and setting the parent-child connections between them.
        """
        # Rank of the current node, number of its children and nodes on the next rank
        r = node.rank
        sibs = 2 ** self.struct[r]
        ranks = sum([2**s for s in self.struct[:r]]) + 1
        # Creating children of the current node
        for sibnum in range(0, sibs):
            ch = Node(r + 1)
            ch.key = node.key + f"{sibnum:b}".rjust(int(log(sibs, 2)), "0")
            ch.sibnum = sibnum
            ch.ranknum = int(ch.key, 2)
            ch.num = ch.ranknum + ranks
            ch.terminal = r + 1 == len(self.struct)
            ch._stance = Stance(
                pos=[int(k) for k in ch.key],
                rep=[0] * len(ch.key),
                depth=self._depth,
            )
            ch.parent = node
            node.children.append(ch)
            self.nodes.append(ch)
        # Performing the same operation for every child if it is not terminal
        if r + 1 < len(self.struct):
            for ch in node.children:
                self._populate(ch)
        return

    def _get_subtree(self, target: Optional[int] = None) -> Tree | Node:
        """Creates a structural copy of the tree and returns either that copy
        or its node of the given number.
        """
        subtree = Tree(self.struct)
        return subtree if target is None else subtree.nodes[target]

    @property
    def depth(self) -> int:
        """The depth of a tree is the number of complex embeddings that resulted
        in its assignment to the parent node.
        """
        return self._depth

    @depth.setter
    def depth(self, value: int) -> None:
        """Updating the depth of the tree also updates the depths of all its nodes
        and trees embedded in the nodes.
        """
        self._depth = value
        for node in self.all_nodes:
            node.stance.depth = value
            for tree in node.complexes:
                tree.depth = value
        return

    @property
    def all_nodes(self) -> List[Node]:
        """A list of all nodes present anywhere in the tree, including compounds
        and heads of complexes.
        """
        nodes: List[Node] = self.root.downstream
        return sorted(nodes, key=lambda node: node.num)

    def get_interpretable_nodes(self, complexes: bool = False) -> List[Node]:
        """Returns terminal nodes with content. If complexes is True,
        recursively includes lists of those for the embedded trees.
        """
        sorted_nodes = sorted(
            [n for n in self.all_nodes if n.content and n.terminal],
            key=lambda node: node.content[0].order,
        )

        out = [n for n in sorted_nodes]
        if complexes:
            ind = 0
            for node in sorted_nodes:
                for c in node.complexes:
                    embeds = c.get_interpretable_nodes(complexes=True)
                    out.insert(ind, embeds)
                    ind += 1
                ind += 1

        return out

    def draw(
        self,
        node: Optional[Node] = None,
        depth: int = 0,
        header: str = "└",
        top: bool = False,
        features: bool = False,
        all_nodes: bool = False,
    ) -> str:
        """Draws the structure of the tree to be printed."""
        if node is None:
            node = self.root
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

        st = prefix + arrow + "─" + repr(node)
        if features and node.feature and node.terminal:
            st += f" > {str(node.feature or '')}"
        st += "\n"

        for ch in node.children:
            if ch.content or all_nodes:
                st += self.draw(ch, depth + 1, "└", True, features, all_nodes)
        for cx in node.complexes:
            if cx.root.content or all_nodes:
                st += prefix + "⤷─" + str(cx) + "\n"
                st += cx.draw(cx.root, depth + 2, "└", True, features, all_nodes)
        for cd in node.compounds:
            if cd.content or all_nodes:
                st += self.draw(cd, depth, "⤷", True, features, all_nodes)

        return st

    def embed_compound(self, node: Node) -> None:
        """Adds a copy of the subtree originating from the node of the given number
        to the mapped compounds list of the node.
        """
        new_node: Node = self._get_subtree(node.num)
        d = len(new_node.key) - 1
        new_stance = Stance(
            new_node.stance.pos, new_node.stance.rep, new_node.stance.depth
        )
        new_stance.rep[d] = len(node.compounds) + 1
        new_node.stance = new_stance
        node.compounds.append(new_node)
        return

    def embed_complex(self, node: Node) -> None:
        """Adds a copy of the subtree originating from the root of the tree
        to the mapped complexes list of the node.
        """
        new_tree: Tree = self._get_subtree()
        new_tree.depth = self.depth + 1
        new_tree.stance = node.stance
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
        pos, comps = "".join(str(s) for s in stance.pos), stance.rep
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

    def set_element(self, e: Element, sep: str = "", set_all: bool = True) -> None:
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
                node.sep = sep
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
        self.rank: int = rank
        self.key: str = str()
        self.num: int = 0
        self.ranknum: int = 0
        self.sibnum: int = 0
        self.terminal: bool = False
        self._stance: Stance = Stance()
        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        self.compounds: List[Node] = []
        self.complexes: List[Tree] = []
        self.content: List[Element] = []
        self.feature: Optional[Feature] = None
        self.sep: str = str()
        return

    def __repr__(self) -> str:
        content = [str(e.preheader) for e in self.content] if self.content else ""
        return f"N({self.num}): {self.sep.join(content)}"

    def __str__(self) -> str:
        return repr(self)

    def map_element(self, e: Element) -> None:
        """Sets the given element as the content of the node."""
        if not self.content or not self.terminal:
            self.content.append(e)
        else:
            raise Exception(f"Tried to rewrite the content of terminal node {self}")

    @property
    def stance(self) -> Stance:
        """The stance of a node is the representation of its dichotomic positions,
        repetitions and depth.
        """
        return self._stance

    @property
    def downstream(self) -> List[Node]:
        """A list that includes the node itself, its compounds,
        the heads of its complexes, and the same objects for every child.
        """
        nodes: List[Node] = [self] + self.compounds
        children: List[Node] = []
        for node in nodes:
            for ch in node.children:
                children += ch.downstream
        return nodes + children

    @stance.setter
    def stance(self, value: Stance) -> None:
        """Updating the stance of the node also updates the stances of all nodes
        downstream of it, as well as those of its complexes and compounds.
        """
        for c in self.complexes:
            c.stance = value
        for c in self.compounds:
            c.stance.pos = value.pos
            c.stance.rep[:-1] = value.rep[:-1]
            c.stance.depth = value.depth
        for ch in self.children:
            ch.stance.pos[: len(value.pos)] = value.pos
            ch.stance.rep[: len(value.rep)] = value.rep
            ch.stance.depth = value.depth
        return
