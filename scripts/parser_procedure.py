import json

from typing import Tuple, List
from pathlib import Path

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
        perms = self._mult(self.term_perms)
        revs = self._mult(self.term_revs)
        self.perms, self.revs = [perms], [revs]

        while len(perms) > 1:
            perms, revs = self._split(perms, revs)
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

    def _mult(self, cperms: List[str]) -> List[str]:
        # Multiplies the perms by chunks according to struct
        chunks = cperms
        for i in range(len(self.struct)):
            size = 2 ** (len(self.struct) - i)
            chunks = [
                chunks[j : j + size] * self.struct[::-1][i]
                for j in range(0, len(chunks), size)
            ]
            chunks = [b for a in chunks for b in a]

        return chunks

    def _split(self, perms: List[str], revs: List[int]) -> List[List[str]]:
        # Splits the perms into n chunks, reversing the order according to revs
        split_perms, split_revs = [], []
        for i in range(0, len(perms), 2):
            rev = min(revs[i : i + 2])
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
            pair_stance = "".join(str(s) for s in stance[0])
            pairs = [mp for r in self.masks for mp in r]
            for pair in pairs:
                if pair[0].key[:-1] == pair_stance and pair[1].key[:-1] == pair_stance:
                    return pair if get_pair else pair[stance[0][-1]]
            raise Exception(f"Couldn't find masks with stance {stance}")

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
        self.masker = Masker(self.grules)
        return

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

    def parse(self, input_string: str) -> bool:
        # Parses the input string
        self.buffer.tree = Tree(self.grules.struct)
        
        parsed_string = self.prepare(input_string)
        self.buffer.parsed_string = parsed_string
        print(f"Parsing '{input_string}' as '{parsed_string}'")

        mapping = self.produce_mapping(parsed_string)
        if mapping:
            self.shift_nonbinary_mappings(mapping)
            self.buffer.mapping = mapping
            self.apply()
            print(self.buffer.tree)
            return True
        else:
            return False

    def shift_nonbinary_mappings(self, mapping: Mapping) -> None:
        # For non-binary dichotomies, shift the mappings towards the right
        # to fill the vacant place

        def check_antagonist(stance: Stance, lemb: int) -> Stance | None:
            new_stance = None
            stances_to_check = [s[0] for s in mapping.stances]
            shifted_stance = [p for p in stance[0]]
            shifted_stance[d] = 1 - shifted_stance[d]

            matches = sum([1 for st in stances_to_check if st == shifted_stance])
            slots_left = 1 if lemb == -1 else 1 + lemb - matches
            shifted_emb = [p for p in stance[1]]
            shifted_emb[d] = matches
            if slots_left > 0:
                new_stance = (shifted_stance, shifted_emb)

            return new_stance

        nbs = [r != 0 for s in self.grules.struct for r in range(s)]
        for d, nb in enumerate(nbs):
            if not nb:
                continue
            for mask_pair in self.masker.masks[d]:
                rev = bool(min(mask_pair[0].rev, mask_pair[1].rev))
                side = 0 if not rev else 1
                key = [int(p) for p in mask_pair[side].key]
                for i, stance in enumerate(mapping.stances):
                    if stance[0][: len(key)] == key:
                        new_stance = check_antagonist(stance, mask_pair[1 - side].lemb)
                        if new_stance:
                            print(f"-> Shifting {stance} to {new_stance}")
                            mapping.stances[i] = new_stance
        return

    def apply(self) -> None:
        # Maps the chars to the tree with the obtained stances
        for i, stance in enumerate(self.buffer.mapping.stances):
            char = self.buffer.mapping.chars[i]
            self.buffer.tree.set_element(stance, char, set_all=True)
        return

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
        if rank < 2:
            rev = 0

        first_mask, second_mask = mask_pair[::-1] if rev else mask_pair

        # Parsing the breaker for a split-set dichotomy and moving to the second mask
        if split and cand in breakers:
            print("Ignoring the breaker")
            first_mask.pos, second_mask.pos = None, 0
            return True

        # Comparing with the first mask
        # Ensure that the second mask wasn't yet fit OR the return is not restricted
        if second_mask.pos is None or not ret:
            print(f"-> Comparing {cand} with first mask {first_mask}")
            comparison = self.compare_with_mask(cand, first_mask, split)
            if comparison:
                self.set_position(first_mask, second_mask, comparison)
                return (int(rev), first_mask.rep)

        # Comparing with the second mask
        # Ensure that either mask was already fit OR the skip is not restricted
        if first_mask.pos is not None or second_mask.pos is not None or not skip:
            print(f"-> Comparing {cand} with second mask {second_mask}")
            comparison = self.compare_with_mask(cand, second_mask, split)
            if comparison:
                self.set_position(second_mask, first_mask, comparison)
                return (int(not rev), second_mask.rep)

        # If neither mask accepts the candidate
        return False

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
                mask.optionals[mask.move(incr)[0]],
            ]
        ):
            if mask.match(rep, incr):
                mov = mask.move(incr)
                # Check that we aren't adding compounds beyond the restriction for the mask
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
        other_mask.pos = None
        return
