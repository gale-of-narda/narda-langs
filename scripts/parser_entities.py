from math import log

from scripts.parser_dataclasses import Feature, Stance, Token
from scripts.util import ParsingFailure, binary, concat, digits


class Mapping:
    """A holder for the current stacks of elements and the parameter values
    involved in parsing.
    """

    def __init__(self, levels: range) -> None:
        self.cur_breaks = [[0] for _ in levels]
        self.cur_dpt = [0 for _ in levels]
        self.cur_bdr = [0 for _ in levels]
        self.elems = [[] for _ in levels]

    def get_interval(self, lvl: int) -> list:
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

    def get_stack(self, lvl: int = -1, interval: bool = False) -> list:
        """Returns the list of elements at the given level
        to which the next element should be appended.
        """
        stack = self.get_interval(lvl) if interval else self.elems[lvl]
        while len(stack) > 0 and isinstance(stack[-1], list):
            stack = stack[-1]
        return stack

    def enumerate_elems(
        self, num_key: list[int], elems: list[Element], d: int, preceding: bool = False
    ) -> dict[str, list[int]]:
        """Returns a dict mapping each rep-before-d key to the list of indices of
        the given elements that conform to the given key.

        If preceding is True, finds elements with key less than the given one.
        """
        matches = {}
        for i, e in enumerate(elems):
            if (not preceding and e.stance.pos[: len(num_key)] == num_key) or (
                preceding and e.stance.pos[: len(num_key)] < num_key
            ):
                slot = concat(e.stance.rep[:d])
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
        self.rev: bool = False
        self.ret: bool = False
        self.skip: bool = False
        self.split: bool = False
        self._pointer: int | None = None
        self.left: Mask
        self.right: Mask
        return

    def __repr__(self) -> str:
        try:
            return f"{repr(self.masks[0])}—{repr(self.masks[1])}"
        except AttributeError:
            return "(empty dichotomy)"

    @property
    def masks(self) -> list[Mask]:
        """Returns the masks in the appropriate order."""
        try:
            return [self.left, self.right] if not self.rev else [self.right, self.left]
        except AttributeError:
            return []

    @property
    def pointer(self) -> int | None:
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
    def key(self) -> str | None:
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
    def num_key(self) -> list[int]:
        """Represents the dichotomy key as a list of integers."""
        return digits(self.key or "")

    @property
    def depth(self) -> int:
        """The depth of the dichotomy is the greatest depth of its masks."""
        return max([m.depth for m in self.masks])


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
        self.literals, self.optionals, self.necessities = self._decode(premask)
        self.rank, self.num, self.depth = rank, num, depth
        self.tneuts: list | None = None
        self.rev: int = 0
        self.lemb: int = 0
        self.demb: int = 0
        self.pos: int | None = None
        self.rep: int = 0
        self.wild: bool = False
        self.freeze: bool = False
        return

    def __repr__(self) -> str:
        out = ""
        lits = ["".join(lit) for lit in self.literals]
        for n, lit in enumerate(lits):
            opt = "?" if self.optionals[n] else ""
            ncs = "!" if self.necessities[n] else ""
            brd = f"({lit})" if len(lit) > 1 else lit
            und = [f"{b}̲" if self.pos == n and b not in "()" else b for b in brd]
            out += opt + ncs + "".join(und)
        return f"'{out}'"

    @property
    def key(self) -> str:
        """The key of the mask is the binary representation of its number."""
        return binary(self.num, self.rank)

    @property
    def num_key(self) -> list[int]:
        """Represents the dichotomy key as a list of integers."""
        return digits(self.key)

    @property
    def active(self) -> bool:
        """A mask is active if its current position is not None."""
        return self.pos is not None

    def _decode(self, premask: str) -> tuple[list[list[str]], list[bool], list[bool]]:
        """Decodes the terminal permissions defined in the general rules
        to transform them into masks.
        """
        optional, necessary, bracketed = False, False, False
        literals, optionals, necessities, group = [], [], [], []

        for c in premask:
            match c:
                case "?":
                    optional = True
                case "!":
                    necessary = True
                case "(":
                    bracketed = True
                case ")":
                    optionals.append(optional)
                    necessities.append(necessary)
                    bracketed, optional, necessary = False, False, False
                    literals += [group]
                    group = []
                case _:
                    group += c
                    if not bracketed:
                        optionals.append(optional)
                        necessities.append(necessary)
                        optional, necessary = False, False
                        literals += [group]
                        group = []

        return literals, optionals, necessities

    def compare(
        self,
        e: Element,
        split: bool,
        force_mov: bool = False,
    ) -> tuple[int | None, int] | None:
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

    def move(self, step: int) -> tuple[int, int]:
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
        head = e.preheader.get_matching_head(e.stance.depth)
        # A non-molar matching head means the element has no molar host to match
        # on (e.g. a word made purely of an embedding): the parse is malformed.
        if not head.molar:
            raise ParsingFailure(f"Matching head {head!r} is not molar")
        aclass, lit = head.tok.base.aclass, head.tok.lit
        # If freeze is True, fitting to the mask is forbidden
        if self.freeze:
            return False
        if aclass == "wildcard":
            return self.wild
        if ignore_pos:
            by_class = any(aclass in m for lits in self.literals for m in lits)
            by_val = any(lit in m for lits in self.literals for m in lits)
            return by_class or by_val
        else:
            target_pos = self.move(pos)[0]
            by_class = any(aclass in m for m in self.literals[target_pos])
            by_val = any(lit in m for m in self.literals[target_pos])
            return by_class or by_val

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
        content: Token | list[Element],
        stance: Stance | None = None,
        level: int = 0,
    ) -> None:
        # A molar element wraps a single token; a composite holds child elements.
        if isinstance(content, Token):
            self.token: Token | None = content
            self.content: list[Element] = []
            self.molar: bool = True
        elif isinstance(content, list) and all(
            [isinstance(c, Element) for c in content]
        ):
            self.token = None
            self.content = content
            self.molar = False
        else:
            raise ValueError(f"Cannot create an element with content {content}")
        self.level: int = level
        self.heads: list[Element] = []
        self.stance: Stance = stance or Stance()
        return

    def __repr__(self) -> str:
        return self.represent()

    def represent(self, depth: int | None = None) -> str:
        """Represents the element as a string as seen from the given depth
        (depth defines which head is emphasized).
        """
        if depth is None:
            depth = self.stance.depth if self.stance else 0
        if self.molar:
            return str(self.tok)
        else:
            out = ""
            for c in self.content:
                sep = "·" if len(out) > 0 and c.level > 0 else ""
                if c is self.get_matching_head(depth):
                    out += f"{sep}{c.represent(depth)}"
                    if c.level == 0:
                        out += "̲"
                else:
                    out += f"{sep}{c.represent()}"
            return out

    @property
    def num(self) -> int:
        """Returns the number represented in the binary form by the stance."""
        return int(concat(self.stance.pos), 2)

    @property
    def order(self) -> int:
        """The order of the element is that of its head's content token."""
        return self.head.tok.order

    @property
    def head(self) -> Element:
        """The main head of the element, which is the first one or the element
        itself if no heads are set.
        """
        return self if not self.heads else self.heads[0]

    @property
    def tok(self) -> Token:
        """The token of a molar element. The head, header and matching-head
        leaves are always molar, so this is the token-carrying counterpart to
        ``content`` (which now only ever holds child elements).
        """
        if self.token is None:
            raise ValueError(f"Expected a molar element, got {self!r}")
        return self.token

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
        """Finds the content elements with the given binary number
        and sets them as heads.
        """
        # Simple elements always have themselves as heads
        if self.molar:
            return False
        fit = False

        # Try each permitted stance one by one until the first fitting stance
        # Compounds with the same stance are recorded together
        for num in nums if isinstance(nums, list) else [nums]:
            if not fit:
                for e in self.content:
                    if int(e.stance.key or "0", 2) == num:
                        self.heads.append(e)
                        fit = True

        # If fallback is permitted and no fitting elements are found,
        # set the first one as the head
        if fit:
            return True
        elif fallback:
            self.heads.append(self.content[0])
            return True
        else:
            return False

    def get_matching_head(self, depth: int | None = None) -> Element:
        """Retrieves the n-th head of the element counting from the right,
        where n is the difference between the element's current stance depth
        and the given depth. Used for matching elements to masks.

        If no depth is given, no stance exists, no heads are set,
        or the number n is greater than the number of heads,
        returns the main head.
        """
        head_num = self.stance.depth - (depth or 0)
        if any(
            [
                depth is None,
                self.stance is None,
                not self.heads,
                head_num >= len(self.heads),
            ]
        ):
            return self.head
        else:
            return self.heads[head_num]


class Tree:
    """A dichotomic tree defined by the given structure."""

    def __init__(self, struct: list[int], level: int = 0, depth: int = 0) -> None:
        self.struct: list[int] = struct
        self.level: int = level
        self._depth: int = depth
        self.root: Node = Node()
        self.nodes: list[Node] = [self.root]
        self.ctype: str | None = None
        self.ptype: str | None = None
        self.stance: Stance | None = None
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
            tokens.append(e.head.tok)
        return "".join([str(g) for g in tokens])

    def _populate(self, node: Node) -> None:
        """Realizes the defined structure by creating the appropriate number of nodes
        and setting the parent-child connections between them.
        """
        # Rank of the current node, number of its children and nodes on the next rank
        r = node.rank
        sibs = 2 ** self.struct[r] if self.struct[r] > 0 else 0
        ranks = sum([2**s for s in self.struct[:r]]) + 1
        # Creating children of the current node
        for sibnum in range(0, sibs):
            ch = Node(r + 1)
            ch.key = node.key + binary(sibnum, int(log(sibs, 2)))
            ch.sibnum = sibnum
            ch.ranknum = int(ch.key, 2)
            ch.num = ch.ranknum + ranks
            ch.terminal = r + 1 == len(self.struct)
            ch._stance = Stance(
                pos=digits(ch.key),
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

    def get_nodes(self, stance: Stance | None = None) -> list[Node]:
        """Returns the node addressed by the given stance as well as
        its ancestors.
        """
        if stance is None:
            stance = Stance()
        cursor, node, out = 0, self.root, [self.root]
        pos, comps = concat(stance.pos), stance.rep
        struct_sums = [sum(self.struct[: i + 1]) for i, s in enumerate(self.struct)]
        struct_to_loop = [s for s in struct_sums if s <= len(pos)]
        for s in struct_to_loop:
            if node.children:
                node = node.children[int(pos[cursor : cursor + s] or "0", 2)]
                comp = comps[cursor : cursor + s][-1]
                if comp > 0:
                    node = node.compounds[comp - 1]
                cursor += s
                out.append(node)
        return out

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
    def all_nodes(self) -> list[Node]:
        """A list of all nodes present anywhere in the tree, including compounds
        and heads of complexes.
        """
        nodes: list[Node] = self.root.downstream
        return sorted(nodes, key=lambda node: node.num)

    @property
    def complexes(self) -> list[Tree]:
        """The trees embedded as complexes on the nodes of this tree, in node
        order. Used to recurse the same operation into embedded trees.
        """
        return [c for n in self.all_nodes if n.complexes for c in n.complexes]

    def draw(
        self,
        node: Node | None = None,
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
        new_node = Tree(self.struct, self.level).nodes[node.num]
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
        new_tree = Tree(self.struct, self.level)
        new_tree.depth = self.depth + 1
        new_tree.stance = node.stance
        node.complexes.append(new_tree)
        return

    def get_interpretable_nodes(self, complexes: bool = False) -> list[Node]:
        """Returns terminal nodes with content. If complexes is True,
        recursively includes lists of those for the embedded trees.
        """
        sorted_nodes = sorted(
            [n for n in self.all_nodes if n.content and n.terminal],
            key=lambda node: node.content[0].order,
        )

        out = list(sorted_nodes)
        if complexes:
            ind = 0
            for node in sorted_nodes:
                for c in node.complexes:
                    embeds = c.get_interpretable_nodes(complexes=True)
                    out[ind:ind] = embeds
                    ind += 1
                ind += 1

        return out

    def _collect_population(self) -> list[dict[int, int]]:
        """For each rank, maps a node's position within the rank to the number of
        content-bearing nodes (including compounds) at that position. The data the
        composition types are matched against.
        """
        population: list[dict[int, int]] = []
        all_nodes = self.all_nodes
        ranks = max((n.rank for n in all_nodes), default=-1) + 1
        for r in range(ranks):
            nodes = [n for n in all_nodes if n.rank == r]
            min_num = min(n.num for n in nodes)
            counts: dict[int, int] = {}
            for n in nodes:
                if n.content:
                    offset = n.num - min_num
                    counts[offset] = counts.get(offset, 0) + 1
            population.append(counts)
        return population

    def _collect_swaps(self) -> set[int]:
        """The indexes of terminal nodes whose molar tokens bear a swapper. Only
        their presence is detected, to be matched against the permutation types.
        """
        return {
            n.ranknum
            for n in self.all_nodes
            if n.terminal and n.content and n.content[0].header.tok.swapper
        }

    def set_element(self, e: Element) -> None:
        """Maps the element to the node addressed by its stance as well as
        to parent nodes all the way up to the root.
        """
        cursor = 0
        struct = [0] + self.struct[: len(e.stance.pos)]
        for s in struct[: len(e.stance.pos) + 1]:
            node = self.get_nodes(e.stance.copy(cursor + s))[-1]
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
        self.rank: int = rank
        self.key: str = ""
        self.num: int = 0
        self.ranknum: int = 0
        self.sibnum: int = 0
        self.terminal: bool = False
        self._stance: Stance = Stance()
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.compounds: list[Node] = []
        self.complexes: list[Tree] = []
        self.content: list[Element] = []
        self.feature: Feature | None = None
        return

    def __repr__(self) -> str:
        cnt = [e.represent() for e in self.content] if self.content else ""
        sep = "" if not self.content or self.content[0].level == 0 else "·"
        return f"N({self.num}): {sep.join(cnt)}"

    def __str__(self) -> str:
        return repr(self)

    def map_element(self, e: Element) -> None:
        """Sets the given element as the content of the node."""
        if not self.content or not self.terminal:
            self.content.append(e)
        else:
            raise AttributeError(
                f"Tried to rewrite the content of terminal node {self}"
            )

    @property
    def stance(self) -> Stance:
        """The stance of a node is the representation of its dichotomic positions,
        repetitions and depth.
        """
        return self._stance

    @property
    def downstream(self) -> list[Node]:
        """A list that includes the node itself, its compounds,
        the heads of its complexes, and the same objects for every child.
        """
        nodes: list[Node] = [self] + self.compounds
        children: list[Node] = []
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
