import math

from typing import Tuple, List, Callable, Optional, Any
from collections import deque


class Mask:
    """
    A string to be used as the mask for the candidate character undergoing
    dichotomy resolution during parsing. Consists of positions to be moved
    through with a cursor, where each position can be optional and/or
    allow for several types of characters rather than one.
    """

    def __init__(self, premask: str, cyclic: bool = True) -> None:
        self.literals, self.optionals = self._decode(premask)
        self.cyclic = cyclic
        self.pos = None
        self.rep = 0
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

    def _decode(self, premask: str) -> Tuple[List[List[str]], List[bool]]:
        # Decodes the terminal permissions defined in the general rules
        # to transform them into masks
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
        """
        Changes the position of the cursor in the mask for the given number
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

    def match(self, rep: str, pos: int = 0, ignore_pos: bool = False) -> bool:
        if ignore_pos:
            return any([rep in m for pos in self.literals for m in pos])
        else:
            target_pos = self.move(pos)[0]
            return any([rep in m for m in self.literals[target_pos]])


class Element:
    """
    A wrapper for the given string that is understood as a language element
    of the given level.
    """

    def __init__(self, content: str, level: int = 0) -> None:
        self.content = content
        self.level = level
        self.rep: str = str()

    def __repr__(self) -> str:
        return self.content

    def __str__(self) -> str:
        return str(repr(self))


class Node:
    """
    A dichotomic tree node. Connected to its parent and children,
    defined by rank, number, and key. Elements can be mapped to nodes
    either directly as content or as a compound or complex elements
    (each type of mapped element is stored separately).
    """

    def __init__(self, rank: int = 0) -> None:
        self.parent: Node | None = None
        self.rank = rank
        self.num = 0
        self.ranknum = 0
        self.sibnum = 0
        self.children = []
        self.content = []
        self.masks = []
        self.compounds = []
        self.complexes = []
        self.key = str()
        self.perm = str()

    def __repr__(self) -> str:
        content = [s for s in self.content if isinstance(s, Element)]
        return f"N({self.num}): {''.join(str(s) for s in content)}"

    def __str__(self) -> str:
        return repr(self)

    def set_key(self, struct: List[int]):
        """
        Computes and saves the key of the node in a binary format.
        """
        layer = 1
        layers = [1] + [(layer := layer * 2**x) for x in struct]
        d = self.num - sum(layers[: self.rank])
        key = f"{d:b}".rjust(sum(struct[: self.rank]), "0")
        self.key = key if self.num > 0 else str()
        return

    def get_masks(self, d: int = 0, stances: Optional[List[int]] = None) -> List[Mask]:
        """
        Retrieves the masks applicable to the given stances and dimension.
        """
        if stances is None:
            stances = []
        if not self.children or d < len(stances):
            return self.masks
        else:
            m = self.masks[d]
            for s in stances:
                m = m[s]
            return m

    def map_element(self, e: Element) -> None:
        if isinstance(e, Element):
            self.content.append(e)
            return
        raise Exception(f"Failed to map {e} to node {self}: not an element")


class Tree:
    """
    A dichotomic tree defined by the provided structure. Keeps its nodes inside.
    """

    def __init__(self, struct: List[int]) -> None:
        self.struct = struct
        self.root = Node()
        self.nodes = [self.root]
        self.traverse(self.root, self._populate)
        self.perms = None

    def __repr__(self) -> str:
        return f"T{self.struct}"

    def __str__(self) -> str:
        st = repr(self) + "\n"
        st += self._draw(self.root)
        return st

    def _populate(self, node: Node, i: int):
        # Realizes the defined structure by creating the appropriate number of nodes
        # and setting the parent-child connections between them.
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
        # Draws the structure of the tree to be printed.
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
            st += self._draw(ch, depth + 1)
        for cd in node.compounds:
            st += self._draw(cd, depth, "⤷", top=True)
        for cx in node.complexes:
            st += prefix + "  " + "⤷─" + repr(cx) + "\n"
            st += cx._draw(cx.root, depth + 2, top=True)

        return st

    def _get_subtree(self, target: Optional[int]) -> Tree:
        # Creates a structural copy of the tree and returns either that copy
        # or its node of the given number
        struct = self.struct
        subtree = Tree(struct)
        subtree.set_permissions(self.perms)
        if target is None:
            return subtree
        else:
            return subtree.nodes[target]

    def traverse(self, subroot: Node, fun: Callable) -> Optional[Any]:
        """
        Applies the given function to the nodes of the subtree
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

    def set_permissions(self, perms: List[List[str]]) -> None:
        """
        Makes the list of partitions of the node's children's permissions.
        """

        # Element m holds the masks for m-th dichotomy at the node
        # If m > 1, the masks come in a binary tree list nagivated with stances
        def split(perms, d):
            mid = len(perms) // 2
            if d == 0:
                return Mask("".join(perms))
            else:
                return [split(perms[:mid], d - 1), split(perms[mid:], d - 1)]

        # Save and set the permissions first
        self.perms = perms
        for n in self.nodes:
            n.perm = perms[n.rank][n.ranknum]

        # Then compute the masks used in matching at non-terminal nodes
        for n in self.nodes:
            if n.children:
                length = int(math.log(len(n.children), 2))
                masks = []
                for d in range(length):
                    mask = split([ch.perm for ch in n.children], d + 1)
                    masks.append(mask)
                n.masks = masks

        return

    def embed_compound(self, tnum: int) -> None:
        """
        Adds a copy of the subtree originating from the node of the given number
        to the mapped compounds list of the node.
        """
        target = self.nodes[tnum]
        new_node = self._get_subtree(target.num)
        target.compounds.append(new_node)
        return

    def embed_complex(self, tnum: int) -> None:
        """
        Adds a copy of the subtree originating from the root of the tree
        to the mapped complexes list of the node.
        """
        target = self.nodes[tnum]
        new_tree = self._get_subtree()
        target.complexes.append(new_tree)
        return

    def get_item_by_key(self, keys: List[List[int]], ignore_comp: bool = False):
        """
        Returns the node defined by the keys. The keys consist of the target node's
        own key and the list of equivalent length that keeps track of compound
        numbers.
        """

        def fix_key(key: List) -> List:
            if not key:
                return key
            i, ks = 0, []
            marks = [[True] * s if s > 1 else False for s in self.struct]
            for m in marks:
                if i < len(key):
                    if isinstance(m, list):
                        ks.append(key[i : i + len(m)])
                        i += len(m)
                    else:
                        ks.append(key[i])
                        i += 1
            return ks

        if not keys or not isinstance(keys[0], List):
            keys = [keys, [None] * len(keys)]

        item = self.root
        keys[0] = fix_key(keys[0])

        for i in range(len(keys[0])):
            k = keys[0][i]
            if isinstance(k, list):
                num = int("".join(str(n) for n in k) or "0", 2)
                item = item.children[num]
            else:
                item = item.children[k]

            if not ignore_comp:
                # If no compound part is supplied, take the last compound that exists
                if keys[1][i] is None:
                    if item.compounds:
                        item = item.compounds[-1]
                # If the compound part is non-zero, take the corresponding compound;
                # else take the item itself
                elif keys[1][i] > 0:
                    item = item.compounds[keys[1][i] - 1]

        return item

    def set_item_by_key(self, keys: List[List[int]], cstr: str) -> None:
        """
        Maps the provided string as a content element to the node
        retrieved by the provided keys.
        """
        e = Element(cstr)
        node = self.get_item_by_key(keys)
        node.map_element(e)
        return
