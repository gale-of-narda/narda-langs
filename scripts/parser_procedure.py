import csv
import json
import logging
import logging.config

from typing import Tuple, List, Optional
from pathlib import Path

from scripts.parser_entities import Mapping, Dichotomy, Tree, Node, Mask, Element
from scripts.parser_dataclasses import Alphabet, GeneralRules, SpecialRules
from scripts.parser_dataclasses import Dialect, Feature, Stance, Grapheme

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.DEBUG,
)


class Parser:
    """The main class that orchestrates all language operations."""

    def __init__(
        self,
        start_level: int = 0,
        end_level: Optional[int] = None,
        path: str = "",
    ) -> None:
        self.slv: int = start_level
        self.elv: Optional[int] = end_level
        self.path: str = path
        self.mappings: List[Mapping] = []
        self.trees: List[Tree] = []
        self.cur_lvl = self.slv
        self.cur_dpt = 0

        self._load_params(path)
        if self.elv is None:
            self.elv = len(self.grules.struct) - 1
            
        return

    def _load_params(self, path: str = str()) -> None:
        """Creates the components and loads the alphabet,
        general and special rules from the given path.
        """
        loader = Loader(path)
        self.alphabet = loader.load_alphabet()
        self.grules = loader.load_grules()
        self.srules = loader.load_srules()
        self.dialect = loader.load_dialect()
        self.mapper = Mapper(self)
        self.interpreter = Interpreter(self)
        return

    def _itemize(self, st: str) -> List[Grapheme] | bool:
        """Transforms the given string into a list of graphemes."""
        if not st:
            st = self.input_string

        prep = self.alphabet.prepare(st)
        symb = self.alphabet.symbolize(prep, self.slv)
        graph = self.alphabet.graphemize(symb)

        if not graph:
            logger.error(f"Could not graphemize '{st}'")
            return False

        return graph

    def _flatten(self, mappings: List[Mapping]) -> List[Grapheme]:
        """Creates an element based on the provided items and sets a head for it."""

        graphemes = []
        for m in mappings:
            e = Element(m.elems, Stance(), self.cur_lvl)
            for num in self.grules.heads[self.cur_lvl - 1]:
                if e.set_head(num):
                    graphemes.append(e.head.content)
                    break
        return graphemes

    def process(self, items: str | List[str], verbose: bool = False) -> None:
        """Performs the parsing procedure for the input string, commits the mapping
        to the dichotomic tree and provides the interpretation.
        """
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.mappings: List[Mapping] = []
        self.trees: List[Tree] = []

        if not items:
            logger.error("Empty input")
        elif not isinstance(items, List):
            items = [items]

        for lvl in range(self.slv, self.elv + 1):
            self.cur_lvl = lvl
            for item in items:
                # Produce the mapping
                self.masker = Masker(self)
                if isinstance(item[0], Mapping):
                    graphemes = self._flatten(item)
                else:
                    graphemes = self._itemize(item)
                mapping = self.mapper.parse(graphemes)
                # Commit the mapping to the tree and interpret it
                if mapping:
                    logger.info(f"Successfully parsed {item}")
                    self.mappings.append(mapping)
                    tree = Tree(self.grules.struct[self.cur_lvl])
                    self.interpreter._apply(mapping.stack, tree)
                    self.interpreter._determine_ctype(tree)
                    self.interpreter._interpret(tree)
                    self.trees.append(tree)
                # Reload everything if parsing fails
                else:
                    self._load_params(self.path)
                    logger.info(f"Failed to parse {item}")
                    return
            items = [self.mappings]

        return

    def draw_tree(
        self,
        tree: Optional[Tree | int] = None,
        features: bool = False,
        all_nodes: bool = False,
    ) -> None:
        """Prints out the given dichotomic tree with mapped elements."""
        if tree is None:
            tree = self.trees[-1]
        elif isinstance(tree, int):
            tree = self.trees[tree]

        st = str(tree)

        if tree.stance:
            st += f" at {tree.stance}"
        st += "\n"
        st += tree.draw(features=features, all_nodes=all_nodes)

        print(st)

        return

    def gloss(
        self,
        tree: Optional[Tree | int] = None,
        to_gloss: Optional[str | List[Node]] = None,
    ) -> None:
        """Iterates the terminal nodes of the tree and replaces the representations
        of their contents with the gloss strings defined by their features.
        If to_gloss is a string, processes it first.
        If to_gloss is a list of nodes, glosses them.
        """
        if tree is None:
            tree = self.trees[-1]
        elif isinstance(tree, int):
            tree = self.trees[tree]

        if to_gloss is None:
            items = tree.get_interpretable_nodes(complexes=True)
        elif isinstance(to_gloss, str):
            self.process(to_gloss)
            self.gloss(self.parser.trees[-1])
        elif isinstance(to_gloss, List):
            items = to_gloss
        else:
            raise ValueError(f"Invalid input to gloss: {to_gloss}")

        glosses, current_glossless = [], ""
        for item in items:
            if isinstance(item, List):
                if current_glossless:
                    glosses.append(f"{current_glossless}-")
                    current_glossless = ""
                glosses.append(f"[{self.gloss(item)}]")
            elif item.feature.argument_gloss:
                if current_glossless:
                    glosses.append(f"{current_glossless}-")
                    current_glossless = ""
                glosses.append(f"{item.feature.argument_gloss}-")
            else:
                current_glossless += "".join(str(e) for e in item.content)

        gloss = "".join(glosses).strip("-")

        return gloss

    def get_stances(self, mapping: Optional[Mapping | int] = None) -> List[Stance]:
        if mapping is None:
            mapping = self.mappings[-1]
        elif isinstance(mapping, int):
            mapping = self.mappings[mapping]
        return [e.stance for e in mapping.elems]


class Loader:
    """Loads the alphabet, special and general rules to be transformed into parameters
    used by the parser.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        return

    def _load_json(self, path: str) -> str:
        path = Path(self.path + path)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.exception("Failed to load parameters from {path}")
            raise e
        return data

    def load_alphabet(self) -> Alphabet:
        """Loads the alphabet and extracts the four types of characters."""
        data = self._load_json("params/alphabet.json")
        alphabet = Alphabet(
            bases=data["Bases"],
            modifiers=data["Modifiers"],
            substitutions=data["Substitutions"],
        )
        alphabet._build_dicts()
        return alphabet

    def load_grules(self) -> GeneralRules:
        """Loads the general rules that define the syntax of the language."""
        data = self._load_json("params/rules_general.json")
        grules = GeneralRules(
            struct=data["Structure"],
            heads=data["Heads"],
            rets=data["Return restrictions"],
            skips=data["Skip restrictions"],
            splits=data["Split-set fits"],
            perms=data["Permissions"],
            revs=data["Reversals"],
            lembs=data["Compound lengths"],
            dembs=data["Complex depths"],
            wilds=data["Wildcard slots"],
        )
        return grules

    def load_srules(self) -> SpecialRules:
        """Loads the special rules that set the character permissions
        for each node of the trees.
        """
        data = self._load_json("params/rules_special.json")
        srules = SpecialRules(
            tperms=data["Terminal permissions"],
            tneuts=data["Terminal neutrals"],
        )
        return srules

    def load_dialect(self) -> Dialect:
        """Loads the typed and untyped feature files that contain the descriptions
        of functions and arguments.
        """
        # Loading the dialect parameters
        data = self._load_json("params/dialect.json")
        # Loading the features with functions and arguments
        tables = {"untyped": [], "typed": []}
        for t in tables:
            path = Path(self.path + f"params/features_{t}.csv")
            with path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=";")
                for line in reader:
                    new_feature = Feature(
                        ctype=line["ctype"] if t == "typed" else None,
                        pos=[int(x) for x in line["pos"]],
                        rep=[int(x) for x in line["rep"]],
                        cpos=[int(x) for x in line["cpos"]],
                        crep=[int(x) for x in line["crep"]],
                        content_class=line["content_class"],
                        priority=int(line["priority"]),
                        index=int(line["index"]),
                        function_name=line["function_name"],
                        argument_gloss=line["argument_gloss"],
                        argument_name=line["argument_name"],
                        argument_description=line["argument_description"],
                    )
                    tables[t].append(new_feature)

        dialect = Dialect(
            ctypes=data["Composition types"],
            untyped=tables["untyped"],
            typed=tables["typed"],
        )

        return dialect


class Masker:
    """Creates, holds, and manipulates the dichotomies and masks as needed
    by the parsing procedure.
    """

    def __init__(self, parser: Parser) -> None:
        self.parser: Parser = parser
        # Level > Depth > Rank > Dichotomy
        self.masks: List[List[List[List[Dichotomy]]]] = [[]]
        self._construct(d=0)
        return

    def _construct(self, d: int) -> None:
        """Creates the hierarchy of dichotomies loaded with mask pairs."""
        lv = self.parser.cur_lvl
        struct = self.parser.grules.struct
        perms = self.parser.grules.perms
        revs = self.parser.grules.revs
        dembs = self.parser.grules.dembs
        wilds = self.parser.grules.wilds
        rets = self.parser.grules.rets
        skips = self.parser.grules.skips
        splits = self.parser.grules.splits
        lembs = self.parser.grules.lembs
        tneuts = self.parser.srules.tneuts

        if len(self.masks) <= lv:
            self.masks.append([])

        # Define which dichotomies are last on rank & which are non-binary
        dich_num = sum(struct[lv])
        lds = [r == len(range(s)) - 1 for s in struct[lv] for r in range(s)]
        nbs = [r != 0 for s in struct[lv] for r in range(s)]

        # Create the non-existing depths of masks
        ranks = []
        for d in range(dich_num):
            dichs = []
            for p in range(0, len(perms[lv][d]), 2):
                left_perms = perms[lv][d][p : p + 2][0]
                right_perms = perms[lv][d][p : p + 2][1]
                left_mask = Mask(left_perms, dich_num - d, p)
                right_mask = Mask(right_perms, dich_num - d, p + 1)

                left_mask.rev = revs[lv][d][p : p + 2][0]
                right_mask.rev = revs[lv][d][p : p + 2][1]
                left_mask.demb = dembs[lv][d][p : p + 2][0]
                right_mask.demb = dembs[lv][d][p : p + 2][1]
                left_mask.wild = wilds[lv][d][p : p + 2][0]
                right_mask.wild = wilds[lv][d][p : p + 2][1]

                # Compound embedding only for the last dichs on rank
                rlembs, rlds = lembs[lv][::-1][d], lds[::-1][d]
                left_mask.lemb = rlembs[p : p + 2][0] if rlds else 0
                right_mask.lemb = rlembs[p : p + 2][1] if rlds else 0
                left_mask.depth = d
                right_mask.depth = d

                # Neutral elements only for the lowest terminal masks
                if d == 0 and lv == 0:
                    left_mask.tneuts = tneuts[p : p + 2][0]
                    right_mask.tneuts = tneuts[p : p + 2][1]

                dich = Dichotomy(d=dich_num - d - 1, nb=nbs[d])
                dich.terminal = d == 0
                dich.left, dich.right = [left_mask, right_mask]
                dich.rev = min(revs[lv][d][p : p + 2]) if d < 1 else 0
                dich.ret = rets[lv][d]
                dich.skip = skips[lv][d]
                dich.split = splits[lv][d]
                dichs.append(dich)

            ranks.append(dichs)

        self.masks[lv].append(ranks[::-1])

        return

    def _find_dichs(self, num_key: List[int], depth: int) -> List[Dichotomy]:
        """Returns dichotomies whose keys start with the given one."""
        dichs = [mp for r in self.masks[self.parser.cur_lvl][depth] for mp in r]
        out = []
        for dich in dichs:
            if num_key == dich.num_key[: len(num_key)]:
                out.append(dich)
        return out

    def get_mask(self, stance: Stance, depth: int) -> Mask:
        """Returns the mask with the key defined by the stance."""
        dichs = self._find_dichs(stance.pos[:-1], depth)
        out = dichs[0].masks[stance.pos[-1]]

        if out:
            return out
        else:
            raise Exception(f"Could not find dichotomy by stance {stance}")

    def get_dichs(
        self, stance: Stance, depth: int, downstream: bool = False
    ) -> Dichotomy | List[Dichotomy]:
        """Returns the dichotomy with the key defined by the stance.
        If downstream is True, also returns every dichotomy downstream of it.
        """
        dichs = self._find_dichs(stance.pos, depth)
        out = dichs if downstream else dichs[0]

        if out:
            return out
        else:
            raise Exception(f"Could not find dichotomy by stance {stance}")

    def set_dichotomy(self, dich: Dichotomy, comp: Tuple[int, int], depth: int) -> None:
        """Records the given tuple of pos and rep to the pointed mask.
        If non-terminal, resets dichotomies downstream of the other mask.
        If rep is increased, also resets those downstream the pointed mask.
        """
        target_mask = dich.masks[dich.pointer]
        other_mask = dich.masks[1 - dich.pointer]
        if not dich.terminal:
            self.reset_dichotomies(depth, other_mask.num_key)
            if target_mask.rep < comp[1]:
                self.reset_dichotomies(depth, target_mask.num_key)
        target_mask.pos, target_mask.rep = comp
        return

    def reset_dichotomies(
        self,
        depth: int,
        num_key: Optional[List[int]] = None,
        total: bool = False,
    ) -> None:
        """Sets the pointers of dichotomies with and downstream of the given key
        to None. Used to reset the masks of one branch when the pointer is set
        to the other, as well as to prepare for parsing the next element.
        """
        dichs = self.get_dichs(Stance(pos=num_key or []), depth, downstream=True)
        for dich in dichs:
            dich.pointer = None
            for mask in dich.masks:
                mask.rep = 0
                if total:
                    mask.freeze = False
        return


class Mapper:
    """Performs the parsing procedure. Splits the input string into language elements
    and produces a mapping of the elements to the dichotomic masks.

    Successful termination of the parsing procedure is the criterion of grammaticality.
    """

    def __init__(self, parser: Parser) -> None:
        self.parser: Parser = parser
        self.graphemes: List[Grapheme] = []
        return

    @property
    def working_string(self) -> str:
        """String representation of the grapheme list loaded in the mapper."""
        return "".join([str(g) for g in self.graphemes])

    def _produce_mapping(self, graphemes: Optional[List[Grapheme]] = None) -> bool:
        """Creates the mapping of elements to dichotomic masks."""

        def record_wildcard(self, g: Grapheme) -> None:
            masks = self.parser.masker.masks[self.parser.cur_lvl][
                self.mapping.cur_depth
            ]
            for dich in masks[-1]:
                for mask in dich.masks:
                    if mask.wild:
                        stance = Stance(
                            mask.num_key,
                            [0] * len(mask.num_key),
                            self.mapping.cur_depth,
                        )
                        e = Element(g, stance, self.mapping.cur_depth)
                        if self._fit_element(e, force_mov=True):
                            self.mapping.record_element(e)
                            logger.debug(f"-> Fit a wildcard to {stance}")

        def check_breaker(self, g: Grapheme, quality: int) -> None:
            for mod in g.modifiers:
                if (
                    mod.asubcat == "Breaker"
                    and mod.quality == quality
                    and mod.level == self.parser.cur_lvl
                ):
                    self.mapping.breaks[self.mapping.cur_depth] += mod.index + 1
                    logger.debug("-> Breaker accounted")
            return

        if graphemes is None:
            graphemes = self.graphemes

        # Iterating the input string with the separator appropriate for the level
        for n, g in enumerate(graphemes):
            logger.debug(f"Working with '{g}'")
            # Dealing with wildcards
            if g.aclass == "Wildcard":
                record_wildcard(self, g)
                continue
            # Dealing with complex embedding
            elif g.is_popper(self.parser.cur_lvl) and self.mapping.cur_depth > 0:
                if self._close_clause():
                    depth = self.mapping.cur_depth
                    self.parser.masker.reset_dichotomies(depth, total=True)
                    self.mapping.breaks[self.mapping.cur_depth] = 0
                    self.mapping.pop()
                    e = self.mapping.stack[-1]
                    logger.debug(f"=> Depth decreased to {self.mapping.cur_depth}")
                else:
                    return False
            elif g.is_pusher(self.parser.cur_lvl):
                self.mapping.push()
                depth = self.mapping.cur_depth
                # Create the mask level if it doesn't exist, reset if it does
                if len(self.parser.masker.masks) - 1 < self.mapping.cur_depth:
                    self.parser.masker._construct(depth)
                else:
                    self.parser.masker.reset_dichotomies(depth, total=True)
                logger.debug(f"=> Depth increased to {self.mapping.cur_depth}")
                continue
            else:
                e = Element(g, Stance(), self.parser.cur_lvl)

            # Update the breaker count if an early breaker is encountered
            check_breaker(self, e.head.content, 0)

            # Determine the stance of the element
            if e.stance == Stance():
                e.stance = self._determine_stance(e)

            # Update the breaker count if a late breaker is encountered
            check_breaker(self, e.head.content, 1)

            # Record the result if viable
            if e.stance is False:
                return False
            elif e.stance is True:
                continue
            elif not g.is_popper(self.parser.cur_lvl):
                self.mapping.record_element(e)
            logger.debug(f"=> Assigned the stance {e.stance}")

            # If the end of the string is reached but depth is still positive,
            # add provisional popper until depth zero is reached
            if n == len(graphemes) - 1 and self.mapping.cur_depth > 0:
                p = self.parser.alphabet.embedders[self.parser.cur_lvl][1]
                g = self.parser.alphabet.get_grapheme(p)
                graphemes.append(g)

        return True

    def _shift_nonbinary_mappings(
        self,
        dich: Dichotomy,
        invert: bool = False,
        force: bool = False,
    ) -> bool:
        """Shifts the mappings from the first to the second mask of the dichotomy
        (or vice versa if invert is True) continuously slot by slot as long as
        the shift produces a valid mapping.
        """
        # Shift is only possible if the target mask has an empty slot.
        # Equivalent mappings are shifted together or not at all,
        # compounds from closest to farthest to the target mask
        # and only if they fit (given lembs and perms).

        # Get keys for both masks
        mask_from, mask_to = dich.masks if not invert else dich.masks[::-1]
        elems = self.mapping.stack
        depth = self.mapping.cur_depth
        matches = self.mapping.enumerate_elems(mask_from.num_key, dich.d)

        # Skip the shift to a non-empty mask unless forced
        if not force and (mask_to.pos is not None or mask_to.rep > 0):
            return True

        # Setting the pointer to activate the target mask
        dich.pointer = 1 - int(invert)

        # Perform the shift slot by slot
        slots_shifted, elems_shifted = 0, 0
        num = min(mask_to.lemb + 1, len(matches))

        # Reverse order?
        for n, slot in enumerate(reversed(matches) if not dich.rev else matches):
            slot_in_process = False
            if num > 0:
                for i in matches[slot]:
                    old_stance = str(elems[i].stance)
                    new_stance = elems[i].stance.copy()
                    new_stance.pos[dich.d] = mask_to.num_key[-1]
                    new_stance.rep[dich.d] = mask_to.rep + n
                    fit = self._fit_element(elems[i], new_stance, dich.d)
                    if fit:
                        logger.debug(
                            f"-> Shifted {elems[i]} from {old_stance} to {new_stance}"
                        )
                        slot_in_process = True
                        elems_shifted += 1
                    else:
                        # Terminate successfully if the first elem from slot fails
                        # unsuccessfully if any other elem fails
                        res = not slot_in_process
                        if not res:
                            logger.warning(
                                f"=> No place to shift {elems[i]} along {dich}"
                            )
                        return res
                num -= 1
            slots_shifted += 1

        # Reset the dichotomies downstream of the mask &
        # subtract the shifted elements from the mask
        self.parser.masker.reset_dichotomies(depth, mask_from.num_key)
        mask_from.subtract(elems_shifted, slots_shifted)

        return True

    def _fill_empty_terminals(self, dich: Dichotomy) -> bool:
        """Adds neutral elements to mirror those with no siblings at terminal nodes.
        Only applicable to the zeroth level.
        """
        if self.parser.cur_lvl != 0:
            return True

        elems = self.mapping.stack
        depth = self.mapping.cur_depth
        left_nk, right_nk = dich.masks[0].num_key, dich.masks[1].num_key
        left_matches = self.mapping.enumerate_elems(left_nk, dich.d)
        right_matches = self.mapping.enumerate_elems(right_nk, dich.d)

        to_fill = [
            (left_matches[lm], dich.masks[1], elems[left_matches[lm][0]])
            for lm in left_matches
            if not right_matches
        ] + [
            (right_matches[rm], dich.masks[0], elems[right_matches[rm][0]])
            for rm in right_matches
            if not left_matches
        ]

        for filling in to_fill:
            indices, neut_mask, e = filling
            if not neut_mask.tneuts[depth]:
                continue
            op_stance = e.stance.copy()
            op_stance.pos[-1] = 1 - op_stance.pos[-1]
            g = self.parser.alphabet.get_grapheme(neut_mask.tneuts[depth])
            neut = Element(g, op_stance, self.parser.cur_lvl)
            fit = self._fit_element(neut, op_stance, term_only=True)
            if not fit:
                logger.warning(
                    f"-> Could not fit neutral element {neut} with stance {op_stance}"
                )
                return False
            else:
                # Insert the neutral to the right or to the left of the original
                # depending on rev and whether it is the right or left sibling
                logger.debug(f"-> Inserting {neut} with stance {op_stance}")
                slot = op_stance.pos[-1] if not neut_mask.rev else 1 - op_stance.pos[-1]
                insert_index = min(indices) if slot == 0 else max(indices)
                neut.head.content.base.order = (
                    elems[insert_index].head.content.base.order + slot - 1
                )
                elems.insert(insert_index + slot, neut)

        return True

    def _validate_mapping(self, elems: Optional[List[Element]] = None) -> bool:
        """Checks that every mapping complies with terminal permissions.
        Only applicable to the zeroth level.
        """
        if self.parser.cur_lvl != 0:
            return True
        if elems is None:
            elems = self.mapping.stack

        cnt = 0
        addr = None
        for i, e in enumerate(elems):
            cnt = cnt + 1 if e.stance.pos + e.stance.rep[:-1] == addr else 0
            addr = e.stance.pos + e.stance.rep[:-1]

            rev = bool(self.parser.grules.revs[self.parser.cur_lvl][0][e.num])
            addrs = [e for e in elems if e.stance.pos + e.stance.rep[:-1] == addr]

            perms = self.parser.srules.tperms[e.num]
            priority = cnt if not rev else len(addrs) - 1 - cnt
            perm = perms[min(self.mapping.cur_depth, len(perms) - 1)][priority]

            aclass = e.head.content.aclass
            base = str(e.head.content.base)
            if not any([base in perm or aclass in perm, aclass == "Wildcard"]):
                logger.error(
                    f"-> No permission for '{e.head}' [{aclass}] at {e.stance}"
                )
                return False

        return True

    def _determine_stance(self, e: Element) -> Stance | bool:
        """Produces the stance for the given element by deciding the dichotomies."""
        # Cycle through the ranks and determine the positions of the string for each
        depth = self.mapping.cur_depth
        e.stance = Stance(depth=depth)
        ds = self.parser.masker.masks[self.parser.cur_lvl][depth]
        for d, _ in enumerate(ds):
            dich = self.parser.masker.get_dichs(e.stance, depth)
            decision = self._decide_dichotomy(e, dich)
            if isinstance(decision, bool):
                return decision
            else:
                e.stance.pos.append(decision[0])
                e.stance.rep.append(decision[1])

        return e.stance

    def _decide_dichotomy(
        self, e: Element, dich: Dichotomy, forbid_shift: bool = False
    ) -> Tuple[int, int] | bool:
        """Produces the decision for the given element and dichotomy, linking
        the former to either first or second mask of the latter.
        """
        fit = None
        depth = self.mapping.cur_depth

        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any((not dich.pointer == 1, not dich.ret)),
            any((dich.pointer == 0, not dich.skip)),
        ]
        # Results of fit for the masks: tuple(pos, rep)
        comps = [
            dich.masks[0].compare(e, dich.split),
            dich.masks[1].compare(e, dich.split),
        ]

        # Skip to the second mask if the breaker count is positive
        if self.mapping.breaks[depth] > 0:
            self.mapping.breaks[depth] -= 1
            if not comps[1]:
                return False
            # Breaking is permanent, so fitting to the first mask is now forbidden
            dich.masks[0].freeze = True
            fit = 1

        # Determine the fit the normal way
        if not fit:
            # If both masks are fitting, choose the first one unless
            # it is the only one that increases rep
            if conds[0] and comps[0] and conds[1] and comps[1]:
                if comps[1][1] == dich.masks[1].rep and comps[0][1] > dich.masks[0].rep:
                    fit = 1
                else:
                    fit = 0
            # First mask fitting (the second one wasn't fit OR ret is not forbidden)
            elif conds[0] and comps[0]:
                fit = 0
            # Second mask fitting (the first one wasn't fit OR skip is not forbidden)
            elif conds[1] and comps[1]:
                fit = 1
            # Otherwise and if the dich is non-binary, perform a shift and try again
            elif dich.nb and not forbid_shift:
                self._shift_nonbinary_mappings(dich, invert=True, force=True)
                return self._decide_dichotomy(e, dich, forbid_shift=True)
            # If all fails
            else:
                logger.warning(f"=> Could not decide {dich} for {e}")
                return False

        # Attempt closure if the obtained fit flips the pointer to 1
        if not dich.terminal and (dich.pointer or 0) != fit:
            closure = self._close_dichotomies(dich)
            if not closure:
                return False

        dich.pointer = fit
        old_mask = f"{dich.masks[fit]}"

        # Prepare the decision to output
        self.parser.masker.set_dichotomy(dich, comps[fit], depth)
        pos = fit if not dich.rev else 1 - fit
        rev = dich.masks[fit].rep

        new_mask = f"{dich.masks[fit]}"
        num_strings = ["1st", "2nd"]
        content = f"-> Fitting {e.head.content} to the {num_strings[fit]}"
        if dich.split:
            logger.debug(f"{content} mask {new_mask}")
        else:
            logger.debug(f"{content} mask {old_mask} → {new_mask}")

        return (pos, rev)

    def _close_clause(self) -> bool:
        """Closes the last fitted branch of the topmost dichotomy
        and applies special rules to validate the final mapping.
        """
        depth = self.mapping.cur_depth
        dich = self.parser.masker.masks[self.parser.cur_lvl][depth][0][0]

        if not self._close_dichotomies(dich):
            logger.error("Failed to close the dichotomies")
            return False

        if not self._validate_mapping():
            logger.error("Failed to validate the mapping")
            return False

        return True

    def _close_dichotomies(self, dich: Dichotomy) -> bool:
        """For dichotomies downstream of the given dichotomy, performs
        the finalizing operations: shift the mappings for the non-binary ones,
        add neutral elements as needed for the terminal ones.
        """
        res = True
        stance = Stance(pos=dich.masks[dich.pointer or 0].num_key)
        depth = self.mapping.cur_depth
        dichs = self.parser.masker.get_dichs(stance, depth, downstream=True)
        invert = bool(dich.pointer or 0)
        for dich in dichs:
            if dich.nb:
                res = res and self._shift_nonbinary_mappings(dich, invert=invert)
            if dich.terminal:
                res = res and self._fill_empty_terminals(dich)
        return bool(res)

    def _fit_element(
        self,
        e: Element,
        stance: Optional[Stance] = None,
        d: Optional[int] = None,
        term_only: bool = False,
        force_mov: bool = False,
    ) -> bool:
        """Records the element if it can be fit with the given stance."""
        depth = self.mapping.cur_depth
        if stance is None:
            stance = e.stance
        for p, pos in enumerate(stance.pos):
            if (term_only and p != len(stance.pos) - 1) or (d is not None and p < d):
                continue
            part_stance = stance.copy(p)
            dich = self.parser.masker.get_dichs(part_stance, depth)
            cur_mask = pos if not dich.rev else 1 - pos

            comp = dich.masks[cur_mask].compare(e, dich.split, force_mov)
            if comp:
                e.stance = stance
                dich.pointer = cur_mask
                self.parser.masker.set_dichotomy(dich, comp, depth)
            else:
                return False
        return True

    def parse(self, graphemes: List[Grapheme]) -> Mapping | bool:
        """Parses the input string, producing the mapping of elements to stances
        and applying it to the dichotomic tree.
        """
        self.graphemes = graphemes
        self.mapping = Mapping(self.parser.cur_lvl)
        self.mapping.heads = self.parser.grules.heads[self.parser.cur_lvl]

        # Apply general rules to produce the mapping
        res = self._produce_mapping()
        if not res:
            logger.error("Failed to produce or finalize the mapping")
            return False

        # Do the backward-looking corrections and apply the special rules
        if self._close_clause():
            self.graphemes = []
            return self.mapping
        else:
            return False


class Interpreter:
    """Commits the given mapping to the dichotomic tree and provides dialectic
    interpretations to the functions of its nodes with the received arguments.
    """

    def __init__(self, parser: Parser) -> None:
        self.parser = parser
        return

    def _apply(self, elems: List[Element], tree: Tree) -> None:
        """Applies the given elements to the given tree using their stances.
        Embeds complexes and compounds in the tree as needed.
        """
        st = self.parser.grules.struct[self.parser.cur_lvl]
        finals = [r + 1 == s for s in st for r in range(s)]
        # Iterating the elements and setting each
        for i, e in enumerate(elems):
            # Accounting for reps
            for j, st in enumerate(e.stance.pos):
                if finals[j]:
                    base_stance = e.stance.copy(j + 1)
                    base_stance.rep[-1] = 0
                    base_node = tree.get_nodes(base_stance)
                    if e.stance.rep[j] > len(base_node.compounds):
                        for c in range(e.stance.rep[j] - len(base_node.compounds)):
                            tree.embed_compound(base_node)

            # Accounting for depth
            if e.complex:
                base_node = tree.get_nodes(e.stance)
                tree.embed_complex(base_node)
                self._apply(e.content, base_node.complexes[-1])

            tree.set_element(e, set_all=True)

        return

    def _determine_ctype(self, tree: Tree) -> None:
        """Determines the composition type of the element recorded in the tree."""
        # Try different types one by one
        # Types specific for the depth level go first, general types last
        ctypes = self.parser.dialect.ctypes[self.parser.cur_lvl]
        if not ctypes:
            ctypes = []
        elif str(tree.depth) in ctypes:
            ctypes = ctypes[str(tree.depth)] | ctypes[""]
        else:
            ctypes = ctypes[""]
        fits = []
        for ctype in ctypes:
            fit = True
            # If the type relates to trees embedded in nodes with the defined stance,
            # first exclude it if the given tree stance does not match it
            if "cpos" in ctypes[ctype]:
                if tree.stance is None or tree.stance.pos != ctypes[ctype]["cpos"]:
                    fit = False
            if "crep" in ctypes[ctype]:
                if tree.stance is None or tree.stance.rep != ctypes[ctype]["crep"]:
                    fit = False
            # Every type has conditions for nodes of different ranks
            for r, rank in enumerate(ctypes[ctype]["Ranks"]):
                nodes = [n for n in tree.all_nodes if n.rank == r]
                min_num = min([node.num for node in nodes])
                # Conditions place limits on the number of nodes in the rank
                # that have any content
                for i, perm in enumerate(rank):
                    hits = [n for n in nodes if n.num - min_num == i and n.content]
                    conds = [
                        perm not in ("*", "+", "-") and len(hits) > int(perm),
                        perm not in ("*", "+", "-") and len(hits) == 0,
                        perm == "+" and len(hits) == 0,
                        perm == "-" and len(hits) != 0,
                    ]
                    if any(conds):
                        fit = False
            if fit:
                fits.append(ctype)

        # Choose the first type that fits
        if fits:
            tree.ctype = fits[0]
        else:
            logger.warning("Could not determine the composition type")

        # Do the same for all embedded trees
        for node in [n for n in tree.all_nodes if n.complexes]:
            for c in node.complexes:
                self._determine_ctype(c)

        return

    def _interpret(self, tree: Tree) -> None:
        """Inscribes interpretations defined in the dialect to the tree nodes
        depending on their content.
        """
        # Find features for the tree nodes and their compounds
        nodes = [n for n in tree.all_nodes if n.terminal and n.content]
        for node in nodes:
            if not node.content:
                continue
            e = node.content[0]
            feature = self.parser.dialect.get_feature(
                e.head.content.index,
                e.head.content.aclass,
                node.stance,
                tree.stance,
                tree.ctype,
            )
            if not feature:
                continue
            node.feature = feature
        # Interpret the embedded trees
        for node in [n for n in tree.all_nodes if n.complexes]:
            for c in node.complexes:
                self._interpret(c)
        return

    def describe(
        self,
        tree: Optional[Tree | int] = None,
        verbose: bool = False,
        prefix: str = "·",
    ) -> None:
        """Prints out a summary of interpreted features currently loaded
        into the tree.
        """
        if tree is None:
            tree = self.parser.trees[0]
        elif isinstance(tree, int):
            tree = self.parser.trees[tree]

        logger.info(f"{prefix} {tree.ctype} '{tree.working_string}'")
        nodes = tree.get_interpretable_nodes()
        # Describe the elements mapped to the nodes themselves and their compounds
        featureless = []
        for node in nodes:
            if node.feature:
                msg = "%s> '%s' — %s: %s"
                args = [
                    prefix,
                    node.content[0],
                    node.feature.function_name,
                    node.feature.argument_name,
                ]
                if verbose:
                    msg += " — %s"
                    args.append(node.feature.argument_description)
                logger.info(msg, *args)
            else:
                featureless.append(node)
        # Note nodes with content but no discerned features
        featureless_content = ", ".join([str(n) for n in featureless])
        if featureless_content:
            logger.info(
                f"{prefix}>> Features lacking interpretation: {featureless_content}"
            )
        # Describe the embedded complexes
        for node in [n for n in tree.all_nodes if n.complexes]:
            for c in node.complexes:
                self.describe(c, prefix=prefix + "·")
        return
