import json

from typing import Dict, Tuple, List, Optional
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
    def __init__(self, grules: GeneralRules):
        self.struct = grules.struct
        self.term_perms = grules.perms
        self.term_revs = grules.revs
        self.term_dembs = grules.demb
        self.lemb = grules.lemb
        self.perms = []
        self.revs = []
        self.demb = []
        self.masks = []
        self._unravel_term_params()
        self.construct_masks()
        return

    def _unravel_term_params(self) -> None:
        # Creates perms and revs per each node based on terminal node parameters
        perms = self.term_perms
        revs = self.term_revs
        dembs = self.term_dembs
        self.perms, self.revs, self.demb = [perms], [revs], [dembs]

        for r in range(int(log(len(perms)))):
            perms, revs, dembs = self._split(perms, revs, dembs, r)
            self.perms.append(perms)
            self.revs.append(revs)
            self.demb.append(dembs)

        return

    def _split(
        self, perms: List[str], revs: List[int], dembs: List[int], r: int
    ) -> List[List[str]]:
        # Splits the perms into n chunks, reversing the order according to revs
        split_perms, split_revs, split_dembs = [], [], []
        for i in range(0, len(perms), 2):
            rev = min(revs[i : i + 2]) if r < 1 else 0
            demb = max(dembs[i : i + 2])
            perm = perms[i : i + 2]
            split_perms.append("".join(perm[::-1] if rev else perm))
            split_revs.append(rev)
            split_dembs.append(demb)

        return split_perms, split_revs, split_dembs

    def construct_masks(self, depth: int = 0) -> None:
        # Creates pairs of masks for each dichotomy
        i = 0
        dichs = [r == len(range(s)) - 1 for s in self.struct for r in range(s)]
        nbs = [r != 0 for s in self.struct for r in range(s)]
        ranks = [(i := i + 1) if d else None for d in dichs]

        masks = []
        for r, rank in enumerate(ranks[::-1]):
            dichs = []
            perm_rank = self.perms[r]

            for p in range(0, len(perm_rank), 2):
                left_mask = Mask(perm_rank[p : p + 2][0], len(ranks) - r, p)
                right_mask = Mask(perm_rank[p : p + 2][1], len(ranks) - r, p + 1)
                left_mask.rev = self.revs[r][p : p + 2][0]
                right_mask.rev = self.revs[r][p : p + 2][1]
                left_mask.demb = self.demb[r][p : p + 2][0]
                right_mask.demb = self.demb[r][p : p + 2][1]

                if rank:
                    left_mask.lemb = self.lemb[rank][p : p + 2][0]
                    right_mask.lemb = self.lemb[rank][p : p + 2][1]

                dich = Dichotomy(nb=nbs[r])
                dich.set_masks([left_mask, right_mask])
                dichs.append(dich)

            masks.append(dichs)

        # Create non-existing depths of masks
        for d in range(depth - len(self.masks) + 1):
            self.masks.append(masks[::-1])
            for rank in self.masks[depth - d]:
                for dich in rank:
                    for mask in dich.masks:
                        mask.depth = depth - d

        return

    def get_masks(
        self, stance: Stance, depth: int, get_dich: bool = False
    ) -> Mask | Tuple[Mask]:
        """
        Returns the mask at the address defined by the stance.
        If get_pair, returns the pair of masks into which the addressed mask splits.
        """
        if len(self.masks) < depth:
            self.construct_masks(depth)

        masks_at_depth = self.masks[depth]

        if len(stance.pos) == 0:
            return masks_at_depth[0][0] if get_dich else None
        else:
            key = "".join(str(s) for s in stance.pos)
            dichs = [mp for r in masks_at_depth for mp in r]
            for dich in dichs:
                keys = dich.masks[0].key[:-1], dich.masks[1].key[:-1]
                if get_dich:
                    if all([key == k for k in keys]):
                        return dich
                else:
                    if all([key[:-1] == k for k in keys]):
                        return dich.masks[stance.pos[-1]]
            raise Exception(f"Couldn't find masks with stance {stance}")

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
        self.masker = Masker(self.grules)
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
        elif self.secondary_pipe(self.buffer.mapping.elems, 0):
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
                if self.secondary_pipe(mapping.cursor, mapping.cur_depth):
                    mapping.pop()
                    elem = mapping.cursor[-1]
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

    def secondary_pipe(
        self, elems: Optional[List[Element]] = None, depth: int = 0
    ) -> bool:
        # Applies backward-looking corrections and validates the final mapping
        # using special rules
        pipe = [
            # Shift the non-binary mappings to the right to fill empty slots
            # (self.shift_nonbinary_mappings, "Failed to make a shift"),
            # Fill empty terminal siblings of existing mappings with neutral strings
            (self.fill_empty_terminals, "Failed to fill an empty terminal"),
            # Apply special rules to validate the final mapping
            (self.validate_mapping, "The mapping is invalid"),
        ]

        for fun, err in pipe:
            if not fun(elems, depth):
                print(err)
                return False

        return True

    def shift_nonbinary_mappings(
        self, elems: List[Element], masks: List[Mask], dich: int, depth: int
    ) -> bool:
        # For the given mask pair, shift the mappings from first to second mask
        # (taking rev into account) if the latter has an empty slot
        # Equivalent mappings are shifted together or not at all,
        # compounds from closest to farthest to the target mask
        # and only if they fit (given lembs and perms)

        # Get keys for both masks
        rev = min(m.rev for m in masks)
        mask_from, mask_to = masks if not rev else masks[::-1]
        stance_from = [int(p) for p in mask_from.key]
        stance_to = [int(p) for p in mask_to.key]

        # Resetting the position to activate the target mask
        cur_comp = tuple([mask_to.pos, mask_to.rep])
        self.set_position(mask_to, mask_from, cur_comp, depth)

        # Find indexes of stances to shift
        matches = {}
        for i, e in enumerate(elems):
            if e.stance.pos[: len(stance_from)] == stance_from:
                slot = e.stance.rep[dich]
                if slot not in matches:
                    matches[slot] = [i]
                else:
                    matches[slot].append(i)

        # Perform the shift slot by slot
        slots_shifted = -1
        num = min(mask_to.lemb + 1, len(matches))
        for n, slot in enumerate(reversed(matches) if not rev else matches):
            slot_in_process = False
            if num > 0:
                for i in matches[slot]:
                    old_stance = str(elems[i].stance)
                    elems[i].stance.pos[dich] = stance_to[-1]
                    elems[i].stance.rep[dich] = max(mask_to.lemb - n, 0)
                    new_stance = str(elems[i].stance)
                    print(f"-> Shifting {old_stance} to {new_stance}")
                    fit = self.fit_element(elems[i], depth, dich)
                    if fit:
                        slot_in_process = True
                    else:
                        return not slot_in_process
                num += -1
            slots_shifted += 1

        new_rep = max(mask_from.rep - slots_shifted, 0)
        self.masker.reset_masks(mask_from, depth, new_rep, full_reset=True)

        return True

    def fill_empty_terminals(self, elems: List[Element], depth: int) -> bool:
        # Adds neutral strings to mirror those with no siblings at terminal nodes
        return True
        if self.level != 0:
            return True

        for i, e in enumerate(elems):
            op_stance = e.stance.copy()
            op_stance.pos[-1], op_stance.rep[-1] = 1 - op_stance.pos[-1], 0

            cnt = len(
                [
                    e
                    for e in elems
                    if e.stance.pos == op_stance.pos
                    and e.stance.rep[:-1] == op_stance.rep[:-1]
                ]
            )

            if cnt == 0:
                mask_stance = Stance()
                mask_stance.pos = op_stance.pos[:-1]
                mask = self.masker.get_masks(mask_stance, depth)

                key = "".join([str(s) for s in op_stance.pos])
                rev = bool(self.grules.revs[int(key, 2)])
                neuts = self.srules.tneuts[int(key, 2)]
                neut_depth = neuts[min(depth, len(neuts) - 1)]
                neut = (
                    neut_depth[e.stance.rep[-1]]
                    if not rev
                    else neut_depth[::-1][e.stance.rep[-1]]
                )
                slot = op_stance.pos[-1] if not mask.rev else 1 - op_stance.pos[-1]
                neut_elem = Element(neut, op_stance, self.level)

                fit = self.fit_element(neut_elem, depth, term_only=True)

                if not fit:
                    return False
                else:
                    # Insert it to the right or to the left of the original
                    # depending on rev and whether it is the right or left sibling
                    print(f"Inserting {neut_elem} with stance {op_stance}")
                    elems.insert(i + slot, neut_elem)

        return True

    def validate_mapping(self, elems: List[Element], depth: int) -> bool:
        # Check that every mapping complies with terminal permissions
        return True
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
        self, e: Element, d: int, forbid_shift: bool = False
    ) -> Tuple(int, int) | bool:
        # Determine the position of the candidate with respect to the dichotomy
        # found at the given stance
        old_dich = len(e.stance.pos)
        ret = bool(self.grules.rets[old_dich])
        skip = bool(self.grules.skips[old_dich])
        split = bool(self.grules.splits[old_dich])
        nb = [r != 0 for s in self.grules.struct for r in range(s)][old_dich]

        # Which mask is the first/second depends on rev & binarity of the dichotomy
        dich = self.masker.get_masks(e.stance, d, get_dich=True)
        base_mask = self.masker.get_masks(e.stance, d)
        # rev = min(m.rev for m in mask_pair)
        # rev = rev if not nb else not rev
        rev = bool(base_mask.rev) if base_mask else False

        # Updating the breaker count if a breaker is encountered
        for i, br in enumerate(self.alphabet.breakers):
            if e.head.content in br:
                print("=> Breaker recorded")
                self.buffer.mapping.breaks += i + 1
                return True

        # Skipping to the second mask if the breaker count is positive
        if self.buffer.mapping.breaks > 0:
            self.masker.reset_masks(dich.masks[0], d)
            dich.masks[1].active = True
            self.buffer.mapping.breaks += -1
            print(f"-> Skipping {dich.masks[0]}")

        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any([not dich.masks[1].active, not ret]),
            any([dich.masks[0].active, not skip]),
        ]
        # Results of fit for the masks: tuple(pos, rep)
        comps = [
            self.compare_with_mask(e, dich.masks[0], split),
            self.compare_with_mask(e, dich.masks[1], split),
        ]

        num_strings = ["first", "second"]
        revs = [int(rev), int(not rev)]

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
        elif nb and not forbid_shift:
            self.shift_nonbinary_mappings(self.buffer.mapping.cursor, dich, old_dich, d)
            return self.decide_dichotomy(e, d, forbid_shift=True)
        # If all fails
        else:
            return False

        init_str = f"-> Fitting {e.head.content} to the {num_strings[fit]} mask {dich.masks[fit]}"
        self.set_position(dich.masks[fit], dich.masks[1 - fit], comps[fit], d)
        fin_str = f"{init_str} → {dich.masks[fit]}" if not split else init_str
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
                # Following strings, unless a non-optional string is getting skipped
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
        dich: Optional[int] = None,
        term_only: bool = False,
    ) -> bool:
        # Attempts to record the element if it is valid
        for n, pos in enumerate(e.stance.pos):
            if (term_only and n != len(e.stance.pos) - 1) or (
                dich is not None and n < dich
            ):
                continue
            part_stance = e.stance.copy(n)
            split = self.grules.splits[n]
            masks = self.masker.get_masks(part_stance, depth, get_pair=True)
            comp = self.compare_with_mask(e, masks[pos], split)
            if comp:
                target, other = masks[pos], masks[1 - pos]
                self.set_position(target, other, comp, depth)
            else:
                print(f"Failed to fit {e} to {masks}")
                return False
        return True
