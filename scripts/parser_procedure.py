import csv
import json

from typing import Tuple, List, Optional
from pathlib import Path

from scripts.parser_entities import Mapping, Dichotomy, Tree, Mask, Element
from scripts.parser_dataclasses import Alphabet, GeneralRules, SpecialRules
from scripts.parser_dataclasses import Dialect, Feature, Stance


class Parser:
    """The main class that orchestrates all language operations."""

    def __init__(self, level: int, path: str = "") -> None:
        self.level = level
        self.path = path
        self._load_params(path)
        return

    def _load_params(self, path: str = "") -> None:
        """Creates the components and and loads the alphabet,
        general and special rules from the given path.
        """
        self.loader = Loader(self)
        self.alphabet = self.loader.load_alphabet(self.level)
        self.grules = self.loader.load_grules(self.level)
        self.srules = self.loader.load_srules(self.level)
        self.dialect = self.loader.load_dialect()
        self.masker = Masker(self)
        self.mapper = Mapper(self)
        self.interpreter = Interpreter(self)
        return

    def process(self, input_string: str) -> None:
        """Performs the parsing procedure for the input string, commits the mapping
        to the dichotomic tree and provides the dialectic interpretation.
        """
        # Create the mapping
        self.masker = Masker(self)
        self.mapper = Mapper(self)
        mapping = self.mapper.parse(input_string)
        if not mapping:
            self._load_params(self.path)
            return
        self.mapping = mapping

        # Commit the mapping to the tree
        self.interpreter = Interpreter(self)
        self.interpreter._apply(self.mapping.elems)
        self.interpreter._determine_ctype()
        self.interpreter._interpret()

        return


class Loader:
    """Loads the alphabet, special and general rules to be transformed into parameters
    used by the parser.
    """

    def __init__(self, parser: Parser) -> None:
        # Directory from which the JSONs are loaded
        self.parser = parser
        self.path = parser.path
        return

    def load_alphabet(self, level: int) -> Alphabet:
        """Loads the alphabet and extracts the four types of characters."""
        path = Path(self.path + "params/alphabet.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = Alphabet(
            # Intrinsically meaningful strings
            # Content strings come in classes and are represented by their class
            # The first string for each class is designated as the neutral string
            content=data["Content"],
            # Chars that replace any other of their level
            wildcards=data["Wildcards"],
            # Mappings between special strings and ordinary alphabetic strings
            equivalents=data["Equivalencies"],
            # Chars that delineate elements of the same level
            separators=data["Separators"][level],
            # Chars that force reversal of the order of matching dichotomic halves
            breakers=data["Breakers"][level],
            # Chars that indicate the borders of embedded elements
            embedders=data["Embedders"][level],
            # Content char classes that breakers can follow
            bclasses=data["Breaking classes"],
        )
        return params

    def load_grules(self, level: int) -> GeneralRules:
        """Loads the general rules that define the syntax of the language."""
        path = Path(self.path + "params/rules_general.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = GeneralRules(
            struct=data["Structure"][level],
            heads=data["Heads"][level],
            rets=data["Return restrictions"][level],
            skips=data["Skip restrictions"][level],
            splits=data["Split-set fits"][level],
            perms=data["Permissions"][level],
            revs=data["Reversals"][level],
            lembs=data["Compound lengths"][level],
            dembs=data["Complex depths"][level],
        )
        return params

    def load_srules(self, level: int) -> SpecialRules:
        """Loads the special rules that set the character permissions
        for each node of the trees.
        """
        path = Path(self.path + "params/rules_special.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = SpecialRules(
            tperms=data["Terminal permissions"],
            tneuts=data["Terminal neutrals"],
        )
        return params

    def load_dialect(self) -> Dialect:
        """Loads the typed and untyped feature files that contain the descriptions
        of functions and arguments.
        """
        # Loading the dialect parameters
        path = Path(self.path + "params/dialect.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

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
                        argument_name=line["argument_name"],
                        argument_description=line["argument_description"],
                    )
                    tables[t].append(new_feature)

        params = Dialect(
            ctypes=data["Composition types"],
            untyped=tables["untyped"],
            typed=tables["typed"],
        )

        return params


class Masker:
    """Creates, holds, and manipulates the dichotomies and masks as needed
    by the parsing procedure.
    """

    def __init__(self, parser: Parser) -> None:
        self.parser = parser
        self.perms, self.revs, self.dembs = parser.grules._unravel_term_params()
        self.struct = parser.grules.struct
        self.rets = parser.grules.rets
        self.skips = parser.grules.skips
        self.splits = parser.grules.splits
        self.lembs = parser.grules.lembs
        self.tneuts = parser.srules.tneuts
        self.masks = []
        self.construct()
        return

    def construct(self, depth: int = 0) -> None:
        """Creates the hierarchy of dichotomies loaded with mask pairs."""
        # Define which dichotomies are last on rank & which are non-binary
        dich_num = sum(self.struct)
        lds = [r == len(range(s)) - 1 for s in self.struct for r in range(s)]
        nbs = [r != 0 for s in self.struct for r in range(s)]

        # Create the non-existing depths of masks
        for cur_depth in range(depth - len(self.masks) + 1):
            ranks = []
            for d in range(dich_num):
                dichs = []
                for p in range(0, len(self.perms[d]), 2):
                    left_mask = Mask(self.perms[d][p : p + 2][0], dich_num - d, p)
                    right_mask = Mask(self.perms[d][p : p + 2][1], dich_num - d, p + 1)
                    left_mask.rev = self.revs[d][p : p + 2][0]
                    right_mask.rev = self.revs[d][p : p + 2][1]
                    left_mask.demb = self.dembs[d][p : p + 2][0]
                    right_mask.demb = self.dembs[d][p : p + 2][1]
                    # Compound embedding only for the last dichs on rank
                    rlembs, rlds = self.lembs[::-1][d], lds[::-1][d]
                    left_mask.lemb = rlembs[p : p + 2][0] if rlds else 0
                    right_mask.lemb = rlembs[p : p + 2][1] if rlds else 0
                    left_mask.depth = depth - cur_depth
                    right_mask.depth = depth - cur_depth
                    # Neutral elements only for the terminal masks
                    if d == 0:
                        left_mask.tneuts = self.tneuts[p : p + 2][0][0]
                        right_mask.tneuts = self.tneuts[p : p + 2][1][0]

                    dich = Dichotomy(d=dich_num - d - 1, nb=nbs[d])
                    dich.terminal = d == 0
                    dich.left, dich.right = [left_mask, right_mask]
                    dich.rev = min(self.revs[d][p : p + 2]) if d < 1 else 0
                    dich.ret = self.rets[d]
                    dich.skip = self.skips[d]
                    dich.split = self.splits[d]
                    dichs.append(dich)

                ranks.append(dichs)

            self.masks.append(ranks[::-1])

        return

    def get_mask(self, stance: Stance, depth: int) -> Mask:
        """Returns the mask with the key defined by the stance."""
        dichs = self.find_dichs(stance.pos[:-1], depth)
        out = dichs[0].masks[stance.pos[-1]]

        if out:
            return out
        else:
            raise Exception(f"Couldn't find dichotomy by stance {stance}")

    def get_dichs(
        self, stance: Stance, depth: int, downstream: bool = False
    ) -> Dichotomy | List[Dichotomy]:
        """Returns the dichotomy with the key defined by the stance.
        If downstream is True, also returns every dichotomy downstream of it.
        """
        dichs = self.find_dichs(stance.pos, depth)
        out = dichs if downstream else dichs[0]

        if out:
            return out
        else:
            raise Exception(f"Couldn't find dichotomy by stance {stance}")

    def find_dichs(self, num_key: List[int], depth: int) -> List[Dichotomy]:
        """Returns dichotomies whose keys start with the given one."""
        dichs = [mp for r in self.masks[depth] for mp in r]
        out = []
        for dich in dichs:
            if num_key == dich.num_key[: len(num_key)]:
                out.append(dich)
        return out

    def set_dichotomy(self, dich: Dichotomy, comp: Tuple[int, int], depth: int) -> None:
        """Records the given tuple of pos and rep to the pointed mask.
        If non-terminal, resets dichotomies downstream of the other mask.
        If rep is increased, also resets those downstream the pointed mask.
        """
        target_mask = dich.masks[dich.pointer]
        other_mask = dich.masks[1 - dich.pointer]
        if not dich.terminal:
            self.reset_dichotomies(other_mask.num_key, depth)
            if target_mask.rep < comp[1]:
                self.reset_dichotomies(target_mask.num_key, depth)
        target_mask.pos, target_mask.rep = comp
        return

    def reset_dichotomies(self, num_key: List[int], depth: int) -> None:
        """Sets the pointers of dichotomies with and downstream of the given key
        to None. Used to reset the masks of one branch when the pointer is set
        to the other.
        """
        dichs = self.get_dichs(Stance(pos=num_key), depth, downstream=True)
        for dich in dichs:
            dich.pointer = None
            for mask in dich.masks:
                mask.rep = 0
        return


class Mapper:
    """Performs the parsing procedure. Splits the input string into language elements
    and produces a mapping of the elements to the dichotomic masks.

    Successful termination of the parsing procedure is the criterion of grammaticality.
    """

    def __init__(self, parser: Parser) -> None:
        self.parser = parser
        self.level = parser.level
        self.alphabet = parser.alphabet
        self.grules = parser.grules
        self.srules = parser.srules
        return

    def _produce_mapping(self, prep_string: str) -> bool:
        """Creates the mapping of elements to dichotomic masks."""
        sep = self.alphabet.separators
        pusher, popper = self.alphabet.embedders
        mapping = self.mapping

        # Iterating the input string with the separator appropriate for the level
        string_iterator = prep_string.split(sep) if sep else [s for s in prep_string]
        for n, string in enumerate(string_iterator):
            print(f"Working with '{string}'")

            # Dealing with complex embedding depth controllers
            if string == popper and mapping.cur_depth > 0:
                if self._close_clause():
                    mapping.pop()
                    elem = mapping.stack[-1]
                    print(f"=> Depth decreased to {mapping.cur_depth}")
                else:
                    return False
            elif string == pusher:
                mapping.push()
                self.parser.masker.construct(mapping.cur_depth)
                print(f"=> Depth increased to {mapping.cur_depth}")
                continue

            else:
                elem = Element(string, Stance(), self.level)

            # Determining the stance of the element
            elem.stance = self._determine_stance(elem)

            # Recording the result if viable
            if elem.stance is False:
                return False
            elif elem.stance is True:
                continue
            elif string != popper:
                mapping.record_element(elem)
                print(f"=> Assigned the stance {elem.stance}")

            # If the end of the string is reached but depth is still positive,
            # add provisional popper until depth zero is reached
            if n == len(string_iterator) - 1 and mapping.cur_depth > 0:
                string_iterator.append(popper)

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
                        print(
                            f"-> Shifted {elems[i]} from {old_stance} to {new_stance}"
                        )
                        slot_in_process = True
                        elems_shifted += 1
                    else:
                        # Terminate successfully if the first elem from slot fails
                        # unsuccessfully if any other elem fails
                        res = not slot_in_process
                        if not res:
                            print(f"=> No place to shift {elems[i]} along {dich}")
                        return res
                num += -1
            slots_shifted += 1

        # Reset the dichotomies downstream of the mask &
        # subtract the shifted elements from the mask
        self.parser.masker.reset_dichotomies(mask_from.num_key, depth)
        mask_from.subtract(elems_shifted, slots_shifted)

        return True

    def _fill_empty_terminals(self, dich: Dichotomy) -> bool:
        """Adds neutral elements to mirror those with no siblings at terminal nodes.
        Only applicable to the zeroth level.
        """
        if self.level != 0:
            return True

        elems = self.mapping.stack
        depth = self.mapping.cur_depth
        left_nk, right_nk = dich.masks[0].num_key, dich.masks[1].num_key
        left_matches = self.mapping.enumerate_elems(left_nk, dich.d)
        right_matches = self.mapping.enumerate_elems(right_nk, dich.d)

        to_fill = [
            (left_matches[lm], dich.masks[1], elems[left_matches[lm][0]].stance)
            for lm in left_matches
            if not right_matches
        ] + [
            (right_matches[rm], dich.masks[0], elems[right_matches[rm][0]].stance)
            for rm in right_matches
            if not left_matches
        ]

        for filling in to_fill:
            indices, neut_mask, neut_stance = filling
            op_stance = neut_stance.copy()
            op_stance.pos[-1] = 1 - op_stance.pos[-1]
            neut = Element(neut_mask.tneuts[depth], op_stance, self.level)
            fit = self._fit_element(neut, op_stance, term_only=True)
            if not fit:
                print(f"-> Couldn't fit neutral element {neut} with stance {op_stance}")
                return False
            else:
                # Insert the neutral to the right or to the left of the original
                # depending on rev and whether it is the right or left sibling
                print(f"-> Inserting {neut} with stance {neut_stance}")
                slot = op_stance.pos[-1] if not neut_mask.rev else 1 - op_stance.pos[-1]
                insert_index = min(indices) if slot == 0 else max(indices)
                elems.insert(insert_index + slot, neut)

        return True

    def _validate_mapping(self, elems: List[Element]) -> bool:
        """Checks that every mapping complies with terminal permissions.
        Only applicable to the zeroth level.
        """
        if self.level != 0:
            return True

        cnt = 0
        addr = None
        for i, e in enumerate(elems):
            cnt = cnt + 1 if e.stance.pos + e.stance.rep[:-1] == addr else 0
            addr = e.stance.pos + e.stance.rep[:-1]

            rev = bool(self.grules.revs[e.num])
            addrs = [e for e in elems if e.stance.pos + e.stance.rep[:-1] == addr]

            rep = self.alphabet.represent(e.head.content, self.level)
            perms = self.srules.tperms[e.num]
            priority = cnt if not rev else len(addrs) - 1 - cnt
            perm = perms[min(self.mapping.cur_depth, len(perms) - 1)][priority]

            if e.head.content not in perm and rep not in perm:
                print(f"-> No permission for '{e.head}' of class '{rep}' at {e.stance}")
                return False

        return True

    def _determine_stance(self, e: Element) -> Stance | bool:
        """Produces the stance for the given element by deciding the dichotomies."""
        # Cycle through the ranks and determine the positions of the string for each
        e.stance = Stance()
        depth = self.mapping.cur_depth
        for d in range(len(self.parser.masker.masks[depth])):
            dich = self.parser.masker.get_dichs(e.stance, depth)
            decision = self._decide_dichotomy(e, dich)
            if type(decision) is bool:
                return decision
            else:
                e.stance.pos.append(decision[0])
                e.stance.rep.append(decision[1])

        e.stance.depth = depth

        return e.stance

    def _decide_dichotomy(
        self, e: Element, dich: Dichotomy, forbid_shift: bool = False
    ) -> Tuple(int, int) | bool:
        """Produces the decision for the given element and dichotomy, linking
        the former to either first or second mask of the latter.
        """
        fit = None
        depth = self.mapping.cur_depth

        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any([not dich.pointer == 1, not dich.ret]),
            any([dich.pointer == 0, not dich.skip]),
        ]
        # Results of fit for the masks: tuple(pos, rep)
        comps = [
            self._compare_with_mask(e, dich.masks[0], dich.split),
            self._compare_with_mask(e, dich.masks[1], dich.split),
        ]

        # Skip to the second mask if the breaker count is positive
        if self.mapping.breaks > 0:
            self.mapping.breaks += -1
            print(f"-> Skipping {dich.masks[0]}")
            if not comps[1]:
                return False
            # Breaking is permanent, so fitting to the first mask is now forbidden
            dich.masks[0].freeze = True
            fit = 1

        # Update the breaker count if a breaker is encountered
        for i, br in enumerate(self.alphabet.breakers):
            if e.head.content in br:
                self.mapping.breaks += i + 1
                print("=> Breaker recorded")
                return True

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
                print(f"=> Couldn't decide {dich} for {e}")
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
            print(f"{content} mask {new_mask}")
        else:
            print(f"{content} mask {old_mask} → {new_mask}")

        return (pos, rev)

    def _close_clause(self) -> bool:
        """Closes the last fitted branch of the topmost dichotomy
        and applies special rules to validate the final mapping.
        """
        elems = self.mapping.stack
        depth = self.mapping.cur_depth
        dich = self.parser.masker.masks[depth][0][0]

        if not self._close_dichotomies(dich):
            print("Failed to close the dichotomies")
            return False

        if not self._validate_mapping(elems):
            print("Failed to validate the mapping")
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
                res *= self._shift_nonbinary_mappings(dich, invert=invert)
            if dich.terminal:
                res *= self._fill_empty_terminals(dich)
        return bool(res)

    def _compare_with_mask(
        self, e: Element, mask: Mask, split: bool
    ) -> Tuple(int, int) | None:
        """Produces the movement for the mask required to fit the element
        with its current stance. If no fit is possible, return None.
        """
        # Check if the representation of the candidate string fits the given mask
        # First check if a complex element can be fit
        if e.complex and mask.demb is not None and mask.demb != -1:
            if e.stance.depth > mask.demb - 1:
                return None

        rep_str = self.alphabet.represent(e.head.content, self.level)

        # Bypass positional matching if split-set fitting is applied
        if split:
            if mask.match(rep_str, ignore_pos=True):
                return (mask.pos, mask.rep)
            else:
                return None

        # Otherwise start going throgh the literals one-by-one
        singular = len(mask.literals) == 1
        incr = 1 if singular and mask.active else 0
        while any(
            [
                # Current string, unless the mask is singular and was already fit
                incr == 0 and not (singular and mask.active),
                # Next string, if the mask is singular or was already fit
                incr == 1 and (singular or mask.active),
                # Next string(s), unless a non-optional string is getting skipped
                incr > 0 and not singular and mask.optionals[mask.move(incr - 1)[0]],
            ]
        ):
            if mask.match(rep_str, incr):
                mov = mask.move(incr)
                # Check that we aren't adding compounds beyond the restriction
                if mov[1] <= (mask.lemb or 0) or mask.lemb == -1:
                    # print(mask, mask.lemb, mask.rep, mov)
                    return mov
            incr += 1

        return None

    def _fit_element(
        self,
        e: Element,
        stance: Stance,
        d: Optional[int] = None,
        term_only: bool = False,
    ) -> bool:
        """Records the element if it can be fit with the current stance."""
        depth = self.mapping.cur_depth
        for p, pos in enumerate(stance.pos):
            if (term_only and p != len(stance.pos) - 1) or (d is not None and p < d):
                continue
            part_stance = stance.copy(p)
            dich = self.parser.masker.get_dichs(part_stance, depth)
            cur_mask = pos if not dich.rev else 1 - pos
            comp = self._compare_with_mask(e, dich.masks[cur_mask], dich.split)
            if comp:
                e.stance = stance
                dich.pointer = cur_mask
                self.parser.masker.set_dichotomy(dich, comp, depth)
            else:
                return False
        return True

    def parse(self, input_string: str) -> Mapping:
        """Parses the input string, producing the mapping of elements to stances
        and applying it to the dichotomic tree.
        """
        self.mapping = Mapping(self.level)
        self.mapping.heads = self.grules.heads

        prepared_string = self.alphabet.prepare(input_string)
        self.prepared_string = prepared_string
        print(f"Parsing '{input_string}' as '{prepared_string}'")

        # Apply general rules to produce the mapping
        res = self._produce_mapping(prepared_string)
        if not res:
            print("Failed to produce the mapping")
            return False

        # Do the backward-looking corrections and _apply the special rules
        if self._close_clause():
            print(f"Successfully parsed '{prepared_string}'")
            return self.mapping
        else:
            return False


class Interpreter:
    """Commits the given mapping to the dichotomic tree and provides dialectic
    interpretations to the functions of its nodes with the received arguments.
    """

    def __init__(self, parser: Parser) -> None:
        self.parser = parser
        self.struct = parser.grules.struct
        self.dialect = parser.dialect
        self.tree = Tree(self.parser.grules.struct)
        return

    def _apply(self, elems: List[Element], tree: Optional[Tree] = None) -> None:
        """Applies the given elements to the given tree using their stances.
        Embeds complexes and compounds in the tree as needed.
        """
        if tree is None:
            tree = self.tree

        finals = [r + 1 == s for s in self.struct for r in range(s)]
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

    def _determine_ctype(self) -> None:
        """Determines the composition type of the element recorded in the tree."""
        fits = []
        ctypes = self.dialect.ctypes
        # Try different types one by one
        for ctype in ctypes:
            fit = True
            # Every type has conditions for nodes of different ranks
            for rank in ctypes[ctype]:
                nodes = [n for n in self.tree.all_nodes if n.rank == int(rank)]
                min_num = min([node.num for node in nodes])
                # Conditions place limits on the number of nodes in the rank
                # that have any content
                for i, perm in enumerate(ctypes[ctype][rank]):
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
            self.tree.ctype = fits[0]
        else:
            print("Could not determine the composition type")
        return

    def _interpret(self, tree: Optional[Tree] = None) -> None:
        """Inscribes interpretations defined in the dialect to the tree nodes
        depending on their content.
        """
        if tree is None:
            tree = self.tree
        # Finding features for the tree nodes and their compounds
        nodes = [n for n in tree.all_nodes if n.terminal and n.content]
        for node in nodes:
            content = node.content[0].head.content[0]
            cc_index = self.parser.alphabet.get_index(content)
            if not cc_index:
                continue
            cc, index = cc_index
            feature = self.dialect.get_feature(
                index, cc, node.stance, tree.stance, tree.ctype
            )
            if not feature:
                continue
            node.feature = feature
        # Intepreting the embedded trees
        for node in [n for n in tree.all_nodes if n.complexes]:
            for c in node.complexes:
                self._interpret(c)
        return

    def describe(
        self, tree: Optional[Tree] = None, verbose: bool = False, prefix: str = str()
    ) -> None:
        """Prints out a summary of interpreted features currently loaded
        into the tree.
        """
        if tree is None:
            tree = self.tree
        string = prefix
        string += f"{tree.ctype} '{self.parser.mapper.prepared_string}'"
        nodes = [n for n in tree.all_nodes if n.feature]
        # Describe the elements mapped to the nodes themselves and their compounds
        for node in nodes:
            string += f"\n{prefix}"
            string += f"-> {node.feature.function_name} '{node.content[0]}': "
            string += f"{node.feature.argument_name} "
            if verbose:
                string += f"— {node.feature.argument_description} "
        print(string)
        # Describe the embedded complexes
        for node in [n for n in tree.all_nodes if n.complexes]:
            for c in node.complexes:
                self.describe(c, prefix=prefix + "· ")
        return
