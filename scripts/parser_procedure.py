import json

from typing import Tuple, List, Optional
from pathlib import Path
from math import log

from scripts.parser_entities import Tree, Mask, Mapping, Element
from scripts.parser_dataclasses import Alphabet, GeneralRules, SpecialRules
from scripts.parser_dataclasses import Buffer, Stance, Dichotomy


class Loader:
    """
    Loads the alphabet, special and general rules to be transformed into parameters
    used by the parser.
    """

    def __init__(self, path: str = ""):
        # Directory from which the JSONs are loaded
        self.path = path
        return

    def load_alphabet(self, level: int):
        """
        Loads the alphabet and extracts the four types of characters.
        """
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
        )
        return params

    def load_grules(self, level: int) -> GeneralRules:
        """
        Loads the general rules that define the syntax of the language.
        """
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
            lemb=data["Compound lengths"][level],
            demb=data["Complex depths"][level],
        )
        return params

    def load_srules(self, level: int):
        """
        Loads the special rules that set the character permissions
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


class Masker:
    def __init__(self, grules: GeneralRules, srules: SpecialRules):
        self.tneuts = srules.tneuts
        self.struct = grules.struct
        self.rets = grules.rets
        self.skips = grules.skips
        self.splits = grules.splits
        self.lembs = grules.lemb
        self.dembs = [grules.demb]
        self.perms = [grules.perms]
        self.revs = [grules.revs]
        self.masks = []
        self._unravel_term_params()
        self.construct_masks()
        return

    def _unravel_term_params(self) -> None:
        # Creates perms, revs, dembs for each mask based on per-terminal grules
        for r in range(int(log(len(self.perms[0])))):
            split_perms, split_revs, split_dembs = [], [], []
            for i in range(0, len(self.perms[-1]), 2):
                rev = min(self.revs[-1][i : i + 2]) if r < 1 else 0
                demb = max(self.dembs[-1][i : i + 2])
                perm = self.perms[-1][i : i + 2]
                split_perms.append("".join(perm[::-1] if rev else perm))
                split_revs.append(rev)
                split_dembs.append(demb)
            self.perms.append(split_perms)
            self.revs.append(split_revs)
            self.dembs.append(split_dembs)
        return

    def construct_masks(self, depth: int = 0) -> None:
        # Creates pairs of masks for each dichotomy
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
                    # Neutral elements for the terminal masks
                    if d == 0:
                        left_mask.tneuts = self.tneuts[p : p + 2][0][0]
                        right_mask.tneuts = self.tneuts[p : p + 2][1][0]

                    dich = Dichotomy(d=dich_num - d - 1, nb=nbs[d])
                    dich.terminal = d == 0
                    dich.set_masks([left_mask, right_mask])
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
        if len(self.masks) < depth:
            self.construct_masks(depth)

        key = "".join(str(s) for s in stance.pos[:-1])
        dichs = self.find_dichs(key, depth)
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
        if len(self.masks) < depth:
            self.construct_masks(depth)

        key = "".join(str(s) for s in stance.pos)
        dichs = self.find_dichs(key, depth)
        if downstream:
            out = dichs
        else:
            out = dichs[0]

        if out:
            return out
        else:
            raise Exception(f"Couldn't find dichotomy by stance {stance}")

    def find_dichs(self, key: str, depth: int) -> List[Dichotomy]:
        # Return dichotomies whose keys start with the given one.
        dichs = [mp for r in self.masks[depth] for mp in r]
        out = []
        for dich in dichs:
            if key == dich.key[: len(key)]:
                out.append(dich)
        return out

    def reset_masks(
        self,
        target: Mask,
        depth: int,
        rep: Optional[int] = None,
        full_reset: bool = False,
    ) -> None:
        dichs = [dich for rank in self.masks[depth] for dich in rank]
        for dich in dichs:
            for mask in dich.masks:
                if mask.key[: len(target.key)] == target.key:
                    if mask.active or full_reset:
                        mask.pos = len(mask.literals) - 1 if not full_reset else 0
                        mask.rep = 0 if rep is None else rep
                        mask.active = False
        return


class Parser:
    def __init__(self, level: int, path: str = "") -> None:
        self.level = level
        self._load_params(path)
        self.buffer = Buffer()
        return

    def _load_params(self, path: str = "") -> None:
        self.loader = Loader(path)
        self.alphabet = self.loader.load_alphabet(self.level)
        self.grules = self.loader.load_grules(self.level)
        self.srules = self.loader.load_srules(self.level)
        return

    def prepare(self, st: str) -> str:
        # Removes non-alphabetic characters and makes the replacements
        # Replace the special strings as defined in the alphabet
        reps = self.alphabet.equivalents
        replaced_string = [reps[ch] if ch in reps else ch for ch in st]
        replaced_string = "".join(replaced_string)

        # Erase the non-alphabetic strings from the string
        content = "".join(self.alphabet.content.values())
        separators = "".join(self.alphabet.separators)
        breakers = "".join(self.alphabet.breakers)
        embedders = "".join(self.alphabet.embedders)
        full_mask = content + separators + breakers + embedders
        to_strip = separators  # + embedders
        masked = [ch for ch in replaced_string if ch in full_mask]

        # Strip separators and embedders strings from both ends
        stripped_string = "".join(masked).strip("".join(to_strip))

        return stripped_string.lower()

    def represent(self, ch: str) -> str:
        # Repalces the input character with its alphabetic representation
        if self.level != 0:
            return ch.upper()
        else:
            content = self.alphabet.content.items()
            breakers = list(self.alphabet.breakers)
            embedders = list(self.alphabet.embedders)
            if any(ch in nc for nc in breakers + embedders):
                return ch
            for key, val in content:
                if ch in val:
                    return key
            raise ValueError(f"No representation for '{ch}' on level {self.level}")

    def parse(self, input_string: str) -> bool:
        # Parse the input string
        self.masker = Masker(self.grules, self.srules)
        self.buffer.tree = Tree(self.grules.struct)
        self.buffer.mapping = Mapping(self.level)
        self.buffer.mapping.heads = self.grules.heads

        prepared_string = self.prepare(input_string)
        self.buffer.parsed_string = prepared_string
        print(f"Parsing '{input_string}' as '{prepared_string}'")

        # Apply general rules to produce the mapping
        res = self.produce_mapping(prepared_string)
        if not res:
            print("Failed to produce the mapping")
            return False
        # Do the backward-looking corrections and apply the special rules
        elif self.close_clause(0):
            # Commit the mapping to the tree
            self.apply(self.buffer.mapping.elems, self.buffer.tree)
            print(self.buffer.tree)
            return True

        return False

    def produce_mapping(self, prep_string: str) -> bool:
        # Produces the dichotomic stances for the split items of prep_string
        sep = self.alphabet.separators
        pusher, popper = self.alphabet.embedders
        mapping = self.buffer.mapping

        string_iterator = prep_string.split(sep) if sep else [s for s in prep_string]
        for n, string in enumerate(string_iterator):
            print(f"Working with '{string}'")

            if string == popper and self.buffer.mapping.cur_depth > 0:
                if self.close_clause(mapping.cur_depth):
                    mapping.pop()
                    elem = mapping.stack[-1]
                    print(f"=> Depth decreased to {mapping.cur_depth}")
                else:
                    return False
            elif string == pusher:
                mapping.push()
                self.masker.construct_masks(self.buffer.mapping.cur_depth)
                print(f"=> Depth increased to {mapping.cur_depth}")
                continue

            else:
                elem = Element(string, Stance(), self.level)

            elem.stance = self.determine_stance(elem, self.buffer.mapping.cur_depth)

            if elem.stance is False:
                print(f"=> Failed to match '{string}'")
                return False
            elif elem.stance is True:
                continue
            elif string != popper:
                mapping.record_element(elem)
                print(f"=> Assigned the stance {elem.stance}")

            if n == len(string_iterator) - 1 and self.buffer.mapping.cur_depth > 0:
                string_iterator.append(popper)

        return True

    def close_clause(self, depth: int = 0) -> bool:
        # Closes the last fitted branch of the upper dichotomy
        # and applies special rules to validate the final mapping
        elems = self.buffer.mapping.stack
        dichs = self.masker.get_dichs(Stance(), depth, downstream=True)
        for dich in dichs:
            if not dich.terminal:
                pointer = elems[-1].stance.pos[0]
                closure = self.attempt_closure(dich, depth, pointer, invert=bool(pointer))

        validation = self.validate_mapping(elems, depth)

        return all([closure, validation])

    def shift_nonbinary_mappings(
        self,
        dich: Dichotomy,
        depth: int,
        invert: bool = False,
        force: bool = False,
    ) -> bool:
        # For the given mask pair, shift the mappings from first to second mask
        # (taking invert into account) if the latter has an empty slot
        # Equivalent mappings are shifted together or not at all,
        # compounds from closest to farthest to the target mask
        # and only if they fit (given lembs and perms)

        # Get keys for both masks
        mask_from, mask_to = dich.masks if not invert else dich.masks[::-1]
        stance_from = [int(p) for p in mask_from.key]
        stance_to = [int(p) for p in mask_to.key]
        elems = self.buffer.mapping.stack

        # Skip the shift to a non-empty mask unless forced
        if not force and (mask_to.pos is not None or mask_to.rep > 0):
            return True

        # Resetting the position to activate the target mask
        cur_comp = tuple([mask_to.pos, mask_to.rep])
        self.set_position(mask_to, mask_from, cur_comp, depth)

        # Find indexes of stances to shift
        matches = {}
        for i, e in enumerate(elems):
            if e.stance.pos[: len(stance_from)] == stance_from:
                slot = e.stance.rep[dich.d]
                if slot not in matches:
                    matches[slot] = [i]
                else:
                    matches[slot].append(i)

        # Perform the shift slot by slot
        slots_shifted = -1
        num = min(mask_to.lemb + 1, len(matches))
        for n, slot in enumerate(reversed(matches) if not dich.rev else matches):
            slot_in_process = False
            if num > 0:
                for i in matches[slot]:
                    old_stance = str(elems[i].stance)
                    new_rep = n if not invert else mask_to.lemb - n
                    elems[i].stance.pos[dich.d] = stance_to[-1]
                    elems[i].stance.rep[dich.d] = new_rep
                    new_stance = str(elems[i].stance)
                    print(f"-> Shifting {elems[i]} from {old_stance} to {new_stance}")
                    fit = self.fit_element(elems[i], depth, dich.d)
                    if fit:
                        slot_in_process = True
                    else:
                        # Terminate successfully if the first elem from slot fails
                        # unsuccessfully if any other elem fails
                        return not slot_in_process
                num += -1
            slots_shifted += 1

        new_rep = max(mask_from.rep - slots_shifted, 0)
        self.masker.reset_masks(mask_from, depth, new_rep, full_reset=True)

        return True

    def fill_empty_terminals(self, dich: Dichotomy, depth: int) -> bool:
        # Adds neutral strings to mirror those with no siblings at terminal nodes
        if self.level != 0:
            return True

        elems = self.buffer.mapping.stack
        left_stance = [int(p) for p in dich.masks[0].key]
        right_stance = [int(p) for p in dich.masks[1].key]

        left_matches = {}
        for i, e in enumerate(elems):
            if e.stance.pos[: len(left_stance)] == left_stance:
                slot = e.stance.rep[dich.d]
                if slot not in left_matches:
                    left_matches[slot] = [i]
                else:
                    left_matches[slot].append(i)

        right_matches = {}
        for i, e in enumerate(elems):
            if e.stance.pos[: len(right_stance)] == right_stance:
                slot = e.stance.rep[dich.d]
                if slot not in right_matches:
                    right_matches[slot] = [i]
                else:
                    right_matches[slot].append(i)

        to_fill = [
            (left_matches[lm], dich.masks[1], elems[left_matches[lm][-1]].stance)
            for lm in left_matches
            if not right_matches
        ] + [
            (right_matches[rm], dich.masks[0], elems[right_matches[rm][-1]].stance)
            for rm in right_matches
            if not left_matches
        ]

        for filling in to_fill:
            indices, neut_mask, neut_stance = filling
            op_stance = neut_stance.copy()
            op_stance.pos[-1] = 1 - op_stance.pos[-1]
            neut = Element(neut_mask.tneuts[depth], op_stance, self.level)
            fit = self.fit_element(neut, depth, term_only=True)
            if not fit:
                return False
            else:
                # Insert it to the right or to the left of the original
                # depending on rev and whether it is the right or left sibling
                print(f"Inserting {neut} with stance {neut_stance}")
                slot = op_stance.pos[-1] if not neut_mask.rev else 1 - op_stance.pos[-1]
                elems.insert(max(indices) + slot, neut)

        return True

    def validate_mapping(self, elems: List[Element], depth: int) -> bool:
        # Check that every mapping complies with terminal permissions
        # return True
        if self.level != 0:
            return True

        for i, e in enumerate(elems):
            rep = self.represent(e.head.content)
            key = "".join([str(s) for s in e.stance.pos])
            rev = bool(self.grules.revs[int(key, 2)])
            perms = self.srules.tperms[int(key, 2)]
            perm_depth = perms[min(depth, len(perms) - 1)]
            perm = (
                perm_depth[e.stance.rep[-1]]
                if not rev
                else perm_depth[::-1][e.stance.rep[-1]]
            )
            if e.head.content not in perm and rep not in perm:
                print(f"No permission for '{e.head}' of class '{rep}' at {e.stance}")
                return False

        return True

    def apply(self, elems: List[Element], tree: Tree) -> None:
        # Maps the strings to the tree with the obtained stances
        # Embeds compounds as it meets their addresses in the stances
        finals = [r + 1 == s for s in self.grules.struct for r in range(s)]
        # Iterating the elements and setting each
        for i, e in enumerate(elems):
            # Accounting for reps
            for j, st in enumerate(e.stance.pos):
                if finals[j]:
                    base_stance = e.stance.copy(j + 1)
                    base_stance.rep[-1] = 0
                    base_node = tree.get_node(base_stance)
                    if e.stance.rep[j] > len(base_node.compounds):
                        for c in range(e.stance.rep[j] - len(base_node.compounds)):
                            tree.embed_compound(base_node)
                            print(f"Embedded a compound at node {base_node.num}")

            # Accounting for depth
            if e.complex:
                base_node = tree.get_node(e.stance)
                tree.embed_complex(base_node)
                print(f"Embedded a complex at node {base_node.num}")
                self.apply(e.content, base_node.complexes[-1])

            tree.set_element(e, set_all=True)

        return

    def determine_stance(self, e: Element, d: int) -> Stance | bool:
        # Cycle through the ranks and determine the positions of the string for each
        e.stance = Stance()
        for dich in range(len(self.masker.masks[d])):
            decision = self.decide_dichotomy(e, d)
            if type(decision) is bool:
                return decision
            else:
                e.stance.pos.append(decision[0])
                e.stance.rep.append(decision[1])

        e.stance.depth = d

        return e.stance

    def decide_dichotomy(
        self, e: Element, depth: int, forbid_shift: bool = False
    ) -> Tuple(int, int) | bool:
        # Determine the position of the candidate with respect to the dichotomy
        # found at the given stance

        dich = self.masker.get_dichs(e.stance, depth)

        # Updating the breaker count if a breaker is encountered
        for i, br in enumerate(self.alphabet.breakers):
            if e.head.content in br:
                print("=> Breaker recorded")
                if dich.masks[1 - dich.rev].active:
                    prev_fit = 1
                elif dich.masks[dich.rev].active:
                    prev_fit = 0
                else:
                    prev_fit = None
                if prev_fit == 0 and dich.d < sum(self.grules.struct) - 1:
                    closure = self.attempt_closure(dich, depth, 0)
                    if not closure:
                        return False
                self.masker.reset_masks(dich.masks[0], depth)
                self.buffer.mapping.breaks += i + 1
                return True

        # Skipping to the second mask if the breaker count is positive
        if self.buffer.mapping.breaks > 0:
            self.buffer.mapping.breaks += -1
            dich.masks[1].active = True
            print(f"-> Skipping {dich.masks[0]}")

        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any([not dich.masks[1].active, not dich.ret]),
            any([dich.masks[0].active, not dich.skip]),
        ]
        # Results of fit for the masks: tuple(pos, rep)
        comps = [
            self.compare_with_mask(e, dich.masks[0], dich.split),
            self.compare_with_mask(e, dich.masks[1], dich.split),
        ]

        num_strings = ["first", "second"]
        revs = [int(dich.rev), int(not dich.rev)]

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
            self.shift_nonbinary_mappings(dich, depth, invert=True, force=True)
            return self.decide_dichotomy(e, depth, forbid_shift=True)
        # If all fails
        else:
            return False

        if dich.masks[1 - dich.rev].active:
            prev_fit = 1
        elif dich.masks[dich.rev].active:
            prev_fit = 0
        else:
            prev_fit = None
        if (
            prev_fit is not None
            and prev_fit != fit
            and dich.d < sum(self.grules.struct) - 1
        ):
            closure = self.attempt_closure(dich, depth, prev_fit or 0)
            if not closure:
                return False

        init_str = f"-> Fitting {e.head.content} to the {num_strings[fit]} mask {dich.masks[fit]}"
        self.set_position(dich.masks[fit], dich.masks[1 - fit], comps[fit], depth)
        fin_str = f"{init_str} → {dich.masks[fit]}" if not dich.split else init_str
        print(fin_str)

        return (revs[fit], dich.masks[fit].rep)

    def compare_with_mask(self, e: Element, mask: Mask, split: bool):
        # Check if the representation of the candidate string fits the given mask
        # First check if a complex element can be fit
        if e.complex and mask.demb is not None and mask.demb != -1:
            if e.stance.depth > mask.demb - 1:
                return None

        rep = self.represent(e.head.content)

        # Bypass positional matching if split-set fitting is applied
        if split:
            if mask.match(rep, ignore_pos=True):
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
            if mask.match(rep, incr):
                mov = mask.move(incr)
                # Check that we aren't adding compounds beyond the restriction
                if mov[1] <= (mask.lemb or 0) or mask.lemb == -1:
                    return mov
            incr += 1

        return None

    def attempt_closure(
        self, dich: Dichotomy, depth: int, cursor: int, invert: bool = False
    ) -> bool:
        # Check if fit matches the previous one
        # If not, perform the given operation for the dichotomy of the previous fit
        res = True
        stance = Stance()
        stance.pos = [int(p) for p in dich.masks[cursor].key]
        dichs = self.masker.get_dichs(stance, depth, downstream=True)
        for dich in dichs:
            if dich.nb:
                res *= self.shift_nonbinary_mappings(dich, depth, invert=invert)
            if dich.terminal:
                res *= self.fill_empty_terminals(dich, depth)
        return bool(res)

    def set_position(
        self, target_mask: Mask, other_mask: Mask, comp: Tuple[int], d: int
    ):
        # Record the pos and rep params obtained by comparison in the mask pair
        # If another compound is embedded,
        # annul the parameters of the target mask and the downward masks
        self.masker.reset_masks(other_mask, d)
        if target_mask.rep < comp[1]:
            self.masker.reset_masks(target_mask, d)
        target_mask.pos, target_mask.rep = comp
        target_mask.active = True
        return

    def fit_element(
        self,
        e: Element,
        depth: int,
        d: Optional[int] = None,
        term_only: bool = False,
    ) -> bool:
        # Attempts to record the element if it is valid
        for p, pos in enumerate(e.stance.pos):
            if (term_only and p != len(e.stance.pos) - 1) or (d is not None and p < d):
                continue
            part_stance = e.stance.copy(p)
            dich = self.masker.get_dichs(part_stance, depth)
            mask_cur = pos if not dich.rev else 1 - pos
            comp = self.compare_with_mask(e, dich.masks[mask_cur], dich.split)
            if comp:
                target, other = dich.masks[mask_cur], dich.masks[1 - mask_cur]
                self.set_position(target, other, comp, depth)
            else:
                print(f"Failed to fit {e} to {dich.masks}")
                return False
        return True
