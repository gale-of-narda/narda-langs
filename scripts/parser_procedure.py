import json

from typing import Dict, Tuple, List
from pathlib import Path
from math import log

from scripts.parser_entities import Tree, Mask
from scripts.parser_dataclasses import Alphabet, GeneralRules, SpecialRules
from scripts.parser_dataclasses import Buffer, Mapping
from scripts.parser_dataclasses import Stance


class Loader:
    """
    Loads the alphabet, special and general rules to be transformed into parameters
    used by the parser.
    """

    def __init__(self, path: str = ""):
        # Directory from which the JSONs are loaded
        self.path = path
        return

    def load_alphabet(self, level: int = 0):
        """
        Loads the alphabet and extracts the four types of characters.
        """
        path = Path(self.path + "params/alphabet.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = Alphabet(
            # Intrinsically meaningful chars
            # Content chars come in classes and are represented by their class
            # The first char for each class is designated as the neutral char
            content=data["Content"],
            # Mappings between special chars and ordinary alphabetic chars
            equivalents=data["Equivalencies"],
            # Chars that delineate elements of the same level
            separators=data["Separators"][level],
            # Chars that force reversal of the order of matching dichotomic halves
            breakers=data["Breakers"][level],
            # Chars that indicate the borders of embedded elements
            embedders=data["Embedders"][level],
        )
        return params

    def load_grules(self, level: int = 0) -> GeneralRules:
        """
        Loads the general rules that define the syntax of the language.
        """
        path = Path(self.path + "params/rules_general.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = GeneralRules(
            struct=data["Structure"][level],
            rets=data["Return restrictions"][level],
            skips=data["Skip restrictions"][level],
            splits=data["Split-set fits"][level],
            perms=data["Permissions"][level],
            revs=data["Reversals"][level],
            lemb=data["Compound lengths"][level],
            demb=data["Complex depths"][level],
        )
        return params

    def load_srules(self, level: int = 0):
        """
        Loads the special rules that set the character permissions
        for each node of the trees.
        """
        path = Path(self.path + "params/rules_special.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = SpecialRules(
            tperms=data["Terminal permissions"], tneuts=data["Terminal neutrals"]
        )
        return params


class Masker:
    def __init__(self, grules: GeneralRules):
        self.struct = grules.struct
        self.term_perms = grules.perms
        self.term_revs = grules.revs
        self.lemb = grules.lemb
        self.demb = grules.demb
        self.perms = []
        self.revs = []
        self.masks = []
        self._unravel_term_params()
        self._construct_masks()
        return

    def _unravel_term_params(self) -> None:
        # Creates perms and revs per each node based on terminal node parameters
        perms = self.term_perms
        revs = self.term_revs
        self.perms, self.revs = [perms], [revs]

        for r in range(int(log(len(perms)))):
            perms, revs = self._split(perms, revs, r)
            self.perms.append(perms)
            self.revs.append(revs)

        return

    def _construct_masks(self) -> None:
        # Creates pairs of masks for each dichotomy
        i = 0
        dichs = [r == len(range(s)) - 1 for s in self.struct for r in range(s)]
        ranks = [(i := i + 1) if d else None for d in dichs]

        for r, rank in enumerate(ranks[::-1]):
            mask_pairs = []
            perm_rank = self.perms[r]

            for p in range(0, len(perm_rank), 2):
                left_mask = Mask(perm_rank[p : p + 2][0], len(ranks) - r, p)
                right_mask = Mask(perm_rank[p : p + 2][1], len(ranks) - r, p + 1)
                left_mask.rev = self.revs[r][p : p + 2][0]
                right_mask.rev = self.revs[r][p : p + 2][1]
                if rank:
                    left_mask.lemb = self.lemb[rank][p : p + 2][0]
                    left_mask.demb = self.demb[rank][p : p + 2][0]
                    right_mask.lemb = self.lemb[rank][p : p + 2][1]
                    right_mask.demb = self.demb[rank][p : p + 2][1]
                mask_pairs.append([left_mask, right_mask])

            self.masks.append(mask_pairs)

        self.masks = self.masks[::-1]
        self.perms = self.perms[::-1]
        self.revs = self.revs[::-1]

        return

    def _split(self, perms: List[str], revs: List[int], r: int) -> List[List[str]]:
        # Splits the perms into n chunks, reversing the order according to revs
        split_perms, split_revs = [], []
        for i in range(0, len(perms), 2):
            rev = min(revs[i : i + 2]) if r < 1 else 0
            perm = perms[i : i + 2]
            split_perms.append("".join(perm[::-1] if rev else perm))
            split_revs.append(rev)

        return split_perms, split_revs

    def get_masks(self, stance: Stance, get_pair: bool = False) -> Mask | Tuple[Mask]:
        """
        Returns the mask at the address defined by the stance.
        If get_pair, returns the pair of masks into which the addressed mask splits.
        """
        if len(stance[0]) == 0:
            return self.masks[0][0] if get_pair else None
        else:
            key = "".join(str(s) for s in stance[0])
            pairs = [mp for r in self.masks for mp in r]
            for pair in pairs:
                if get_pair:
                    if pair[0].key[:-1] == key and pair[1].key[:-1] == key:
                        return pair
                else:
                    if pair[0].key[:-1] == key[:-1] and pair[1].key[:-1] == key[:-1]:
                        return pair[stance[0][-1]]
            raise Exception(f"Couldn't find masks with stance {stance[0]}")

    def reset_masks(self, target: Mask) -> None:
        masks = [m for r in self.masks for mp in r for m in mp]
        for mask in masks:
            if mask.key[: len(target.key)] == target.key:
                mask.pos, mask.rep = None, 0
        return


class Parser:
    def __init__(self, level: int = 0, path: str = "") -> None:
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

        # Replace the special chars as defined in the alphabet
        reps = self.alphabet.equivalents
        replaced_string = [reps[ch] if ch in reps else ch for ch in st]
        replaced_string = "".join(replaced_string)

        # Erase the non-alphabetic chars from the string
        content = "".join(self.alphabet.content.values())
        separators = "".join(self.alphabet.separators)
        breakers = "".join(self.alphabet.breakers)
        embedders = "".join(self.alphabet.embedders)
        full_mask = content + separators + breakers + embedders
        to_strip = separators + embedders
        masked = [ch for ch in replaced_string if ch in full_mask]

        # Strip separators and embedders chars from both ends
        stripped_string = "".join(masked).strip("".join(to_strip))

        return stripped_string.lower()

    def represent(self, ch: str) -> str:
        # Repalces the input character with its alphabetic representation
        if self.level == 0:
            return ch.upper()
        else:
            content = self.alphabet.content.items()
            breakers = list(self.alphabet.breakers)
            embedders = list(self.alphabet.embedders)
            if any(ch == nc for nc in breakers + embedders):
                return ch
            for key, val in content:
                if ch in val:
                    return key
            raise ValueError(f"No representation for '{ch}' on level {self.level}")

    def parse(self, input_string: str) -> bool:
        # Parses the input string
        self.buffer.tree = Tree(self.grules.struct)
        self.masker = Masker(self.grules)

        parsed_string = self.prepare(input_string)
        self.buffer.parsed_string = parsed_string
        print(f"Parsing '{input_string}' as '{parsed_string}'")

        # Apply general rules to produce the mapping
        self.buffer.mapping = self.produce_mapping(parsed_string)
        if self.buffer.mapping:
            if self.shift_nonbinary_mappings():
                self.fill_empty_terminals()
                # Apply special rules to validate the mapping
                if self.validate_mapping():
                    # Commit the mapping to the tree
                    self.apply()
                    print(self.buffer.tree)
                    return True
                else:
                    print("Couldn't validate the mapping")
                    return False
            else:
                print("Failed to make a shift")
                return False
        else:
            print("Couldn't produce the mapping")
            return False

    def produce_mapping(self, prep_string: str) -> Mapping | bool:
        # Produces the dichotomic stances for each character in the string
        stances, chars = [], []
        for char in prep_string:
            print(f"Working with '{char}'")
            stance = self.determine_char_stance(char)
            if stance is False:
                print(f"=> Failed to match '{char}'")
                return False
            elif stance is True:
                continue
            else:
                stances.append(stance)
                chars.append(char)
                print(f"=> Assigned the stance {stance}")

        mapping = Mapping(chars, stances)

        return mapping

    def shift_nonbinary_mappings(self) -> None:
        # For non-binary dichotomies, shift the mappings from the first to the second
        # mask (taking rev into account) if the second mask has no mappings
        # Equivalent mappings are shifted together or not at all,
        # compounds in LIFO order, and only if lembs allow
        # queue = []
        nbs = [r != 0 for s in self.grules.struct for r in range(s)]
        for d, nb in enumerate(nbs):
            if not nb:
                continue
            for mask_pair in self.masker.masks[d]:
                # Get keys for both masks
                keys = [[int(p) for p in mask.key] for mask in mask_pair]
                base_mask = self.masker.get_masks(tuple([keys[0][:-1], []]))
                base_rev = bool(base_mask.rev) if base_mask else 0
                min_rev = min([mask.rev for mask in mask_pair])
                fkey, skey = keys if not base_rev else keys[::-1]
                lemb = mask_pair[1 if not base_rev else 0].lemb
                # Find indexes of orresponding mappings
                fmatches = self.get_matching_stances(fkey, d)
                smatches = self.get_matching_stances(skey, d)
                # Discard mappings present in both masks
                matches = {m: fmatches[m] for m in fmatches if m not in smatches}
                # Shift the remaining mappings, but only the last lemb+1 compounds
                for prev_reps in matches:
                    num = min(lemb + 1, len(matches[prev_reps]))
                    for n, drep in enumerate(reversed(matches[prev_reps])):
                        if num > (0 if not min_rev else 1):
                            for i in matches[prev_reps][drep]:
                                print(f"-> Shifting {self.buffer.mapping.stances[i]}")
                                stance = self.buffer.mapping.stances[i]
                                char = self.buffer.mapping.chars[i]
                                stance[0][d] = skey[-1]
                                stance[1][d] = lemb - n - 1
                                fit = self.fit_stance(stance, char)
                                if not fit:
                                    return False
                            num += -1
        return True

    def fit_stance(self, stance: Stance, cand: str) -> bool:
        # Sets the given stance to the masks if it is valid
        for n in range(len(stance[0])):
            part = tuple([stance[0][: n], stance[1][: n]])
            masks = self.masker.get_masks(part, get_pair=True)
            split = self.grules.splits[n]
            comp = self.compare_with_mask(cand, masks[stance[0][n]], split)
            if comp:
                self.set_position(masks[stance[0][n]], masks[1 - stance[0][n]], comp)
            else:
                return False
        return True

    def fill_empty_terminals(self) -> None:
        # Adds neutral chars to mirror those with no siblings at terminal nodes
        stances = self.buffer.mapping.stances
        chars = self.buffer.mapping.chars
        for i, stance in enumerate(stances):
            op_stance, cnt, _ = self.count_slots(stance)
            key = "".join([str(s) for s in op_stance[0]])
            mask = self.masker.get_masks(tuple([op_stance[0][:-1], []]))
            if cnt == 0:
                key = "".join([str(s) for s in op_stance[0]])
                neut_char = self.srules.tneuts[int(key, 2)][0]
                slot = op_stance[0][-1] if not mask.rev else 1 - op_stance[0][-1]
                # Insert it to the right or to the left of the original
                # depending on rev and whether it is the right or left sibling
                stances.insert(i + slot, op_stance)
                chars.insert(i + slot, neut_char)
        return

    def get_matching_stances(self, stance: Stance, d: int) -> Dict[Dict[int]]:
        # Collects stances equal to the given one up to d-th rep
        # Returns a dict of dicts (reps before d -> d-th rep) of indices
        matches = {}
        for i, st in enumerate(self.buffer.mapping.stances):
            keymask = st[0][: len(stance)]
            prev_reps = "".join([str(s) for s in st[1][:d]])
            drep = st[1][d]
            if keymask == stance:
                if prev_reps not in matches:
                    matches[prev_reps] = {drep: [i]}
                elif drep not in matches[prev_reps]:
                    matches[prev_reps][drep] = [i]
                else:
                    matches[prev_reps][drep].append(i)

        return matches

    def count_slots(self, stance: Stance, d: int = -1):
        # Checks how many vacant terminal slots exist opposite to the given one
        # along the given dichotomy
        cnt = 0
        op_stance = tuple([[s for s in pt] for pt in stance])
        op_stance[0][d], op_stance[1][d] = 1 - op_stance[0][d], 0
        mask = self.masker.get_masks(op_stance)

        for st in self.buffer.mapping.stances:
            if st[0] == op_stance[0] and st[1][:-1] == op_stance[1][:-1]:
                cnt += 1

        if mask.lemb == -1:
            slots = 1
        else:
            slots = 1 + mask.lemb - cnt

        return op_stance, cnt, slots

    def validate_mapping(self) -> bool:
        # Check that every mapping complies with terminal permissions
        if self.level == 0:
            return True

        for i, stance in enumerate(self.buffer.mapping.stances):
            char = self.buffer.mapping.chars[i]
            rep = self.represent(char)
            key = "".join([str(s) for s in stance[0]])
            perm = self.srules.tperms[int(key, 2)][0]
            if char not in perm and rep not in perm:
                print(f"No permission for {char}/{rep} at {stance[0]}")
                return False

        return True

    def apply(self) -> None:
        # Maps the chars to the tree with the obtained stances
        # Embeds compounds as it meets their addresses in the stances
        maps, tree = self.buffer.mapping, self.buffer.tree
        finals = [r + 1 == s for s in self.grules.struct for r in range(s)]
        for i, stance in enumerate(maps.stances):
            char = maps.chars[i]
            for j, st in enumerate(stance[0]):
                if finals[j]:
                    base_stance = tuple([st[: j + 1] for st in stance])
                    base_stance[1][-1] = 0
                    base_node = tree.get_node(base_stance)
                    if stance[1][j] > len(base_node.compounds):
                        tree.embed_compound(base_node)
                        print(f"Embedded a compound at node {base_node.num}")
            tree.set_element(stance, char, set_all=True)
        return

    def determine_char_stance(self, char: str) -> Stance | bool:
        # Cycle through the ranks and determine the positions of the char for each
        stance = ([], [])
        for d in range(len(self.masker.masks)):
            decision = self.decide_dichotomy(char, stance)
            if type(decision) is bool:
                return decision
            else:
                stance[0].append(decision[0])
                stance[1].append(decision[1])

        return stance

    def decide_dichotomy(self, cand: str, stance: Stance) -> Tuple(int, int) | bool:
        # Determine the position of the candidate with respect to the dichotomy
        # found at the given stance
        breakers = self.alphabet.breakers

        rank = len(stance[0])
        ret = bool(self.grules.rets[rank])
        skip = bool(self.grules.skips[rank])
        split = bool(self.grules.splits[rank])

        mask_pair = self.masker.get_masks(stance, get_pair=True)
        base_mask = self.masker.get_masks(stance)
        rev = bool(base_mask.rev) if base_mask else 0

        # Which mask is the first/second depends on rev
        masks = mask_pair[::-1] if rev else mask_pair
        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any([masks[1].pos is None, not ret]),
            any([masks[0].pos is not None, masks[1].pos is not None, not skip]),
        ]
        # Results of fit for the masks: tuple(pos, rep)
        comps = [
            self.compare_with_mask(cand, masks[0], split),
            self.compare_with_mask(cand, masks[1], split),
        ]

        num_strings = ["first", "second"]
        revs = [int(rev), int(not rev)]

        # Skipping the breaker
        if split and cand in breakers:
            print("Ignoring the breaker")
            masks[0].pos, masks[1].pos = None, 0
            return True
        # If both masks are fitting, choose the one with lesser rep
        # elif (conds[0] and comps[0]) and (conds[1] and comps[1]):
        #    fit = 0 if comps[0][1] <= comps[1][1] else 1
        # First mask fitting (the second one wasn't fit OR ret is not forbidden)
        elif conds[0] and comps[0]:
            fit = 0
        # Second mask fitting (either mask wasn't fit OR skip is not forbidden)
        elif conds[1] and comps[1]:
            fit = 1
        # No fit
        else:
            return False

        init_str = f"-> Fitting {cand} to the {num_strings[fit]} mask {masks[fit]}"
        self.set_position(masks[fit], masks[1 - fit], comps[fit])
        fin_str = f"{init_str} → {masks[fit]}" if not split else init_str
        print(fin_str)

        return (revs[fit], masks[fit].rep)

    def compare_with_mask(self, cand: str, mask: Mask, split: bool):
        # Check if the representation of the candidate char fits the given mask
        rep = self.represent(cand)
        singular = len(mask.literals) == 1
        incr = 1 if singular and mask.pos is not None else 0

        # Bypass positional matching if split-set fitting is applied
        if split:
            if mask.match(rep, ignore_pos=True):
                return (mask.pos, mask.rep)
            else:
                return None

        # Otherwise start going throgh the literals one-by-one
        while any(
            [
                # Current char, unless the mask is singular and was already fit
                incr == 0 and not (singular and mask.pos is not None),
                # Next char, if the mask is singular or was already fit
                incr == 1 and (singular or mask.pos is not None),
                # Following chars, unless a non-optional char is getting skipped
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

    def set_position(self, target_mask: Mask, other_mask: Mask, comp: Tuple[int]):
        # Record the pos and rep params obtained by comparison in the mask pair
        # If another compound is embedded,
        # annul the parameters of the target mask and the downward masks
        if target_mask.rep < comp[1]:
            self.masker.reset_masks(target_mask)
        target_mask.pos, target_mask.rep = comp
        self.masker.reset_masks(other_mask)
        return
