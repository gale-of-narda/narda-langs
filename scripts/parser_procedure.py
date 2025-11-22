import json

from typing import Tuple, List, Dict, Iterable
from pathlib import Path

from scripts.parser_entities import Tree, Node, Mask
from scripts.parser_dataclasses import Alphabet, GeneralRules, SpecialRules, Buffer


class Loader:
    def __init__(self, path: str = ""):
        self.path = path
        return

    def load_alphabet(self):
        path = Path(self.path + "params/alphabet.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = Alphabet(
            # Intrinsically meaningful chars
            # Content chars come in classes and are represented by their class
            # The first char for each class is designated as the neutral char
            content=data["Content"],
            # Chars that delineate elements of the same level
            separators=data["Separators"],
            # Chars that force reversal of the order of matching dichotomic halves
            breakers=data["Breakers"],
            # Chars that indicate the borders of embedded elements
            embedders=data["Embedders"],
        )
        return params

    def load_grules(self, level: int = 0) -> GeneralRules:
        path = Path(self.path + "params/rules_general.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = GeneralRules(
            struct=data["Structure"][level],
            rets=data["Return restrictions"][level],
            skips=data["Skip restrictions"][level],
            perms=data["Permissions"][level],
            revs=data["Reversals"][level],
        )
        return params

    def load_srules(self, level: int = 0):
        path = Path(self.path + "params/rules_special.json")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = SpecialRules(
            tperms=data["Terminal permissions"],
            tneuts=data["Terminal neutrals"],
            lemb=data["Compound lengths"][level],
            demb=data["Complex depths"][level],
        )
        return params

    def compute_permissions(self, level: int = 0) -> List:
        grules = self.load_grules(level)
        struct, perms, revs = grules.struct, grules.perms, grules.revs

        def flatten(seq: Iterable, depth: int = 0):
            for item in seq:
                if isinstance(item, list) and depth > 0:
                    if len(item) > 1:
                        it = item[0][::-1] if item[1] and depth > 1 else item[0]
                    else:
                        it = item
                    yield from flatten(it, depth - 1)
                else:
                    yield item

        def build_struct(building_struct: List) -> List:
            rooted = [1] + struct
            for rank, mult in enumerate(rooted):
                rsize = 2 ** (len(rooted) - rank - 1)
                gsize = len(building_struct) // rsize
                layer = []
                for binnum in range(rsize):
                    group = building_struct[binnum * gsize : (binnum + 1) * gsize]
                    group_rev = [group, revs[::-1][rank][binnum]]
                    multed = [group_rev] * mult
                    layer += multed
                building_struct = layer
            return building_struct

        def aggregate_struct(built_struct: List) -> List:
            rooted = [1] + struct
            agg_struct = []
            for rank in range(len(rooted)):
                layer = []
                nodes = list(flatten(built_struct, rank))
                for node in nodes:
                    parts: list = list(flatten([node], len(rooted) - rank))
                    joined = "".join(parts)
                    layer += [joined]
                agg_struct += [layer]
            return agg_struct

        built_struct = build_struct(perms)
        agg_struct = aggregate_struct(built_struct)

        return agg_struct


class Parser:
    def __init__(self):
        self.buff = Buffer()
        return

    def load_params(self, path: str = "", level: int = 0) -> None:
        self.loader = Loader(path)
        self.level = level
        self.alphabet = self.loader.load_alphabet()
        self.grules = self.loader.load_grules(level)
        self.srules = self.loader.load_srules(level)
        return

    def _prepare(self, s: str) -> str:
        # Erase the non-alphabetic chars from the string
        content = "".join(self.alphabet.content.values())
        separators = "".join(self.alphabet.separators[self.level])
        breakers = "".join(self.alphabet.breakers[self.level])
        embedders = "".join(self.alphabet.embedders[self.level])
        mask = content + separators + breakers + embedders
        masked = [ch for ch in s if ch in mask]
        # Strip the non-content chars from both ends
        non_content = [ch for ch in masked if ch not in content]
        stripped = "".join(masked).strip("".join(non_content))
        return stripped.lower()

    def _get_bins(self, stances: List[int], stop: int) -> Tuple[int, int]:
        struct = self.buff.tree.struct
        lst = [r == 0 for s in struct for r in range(s)]
        fstances = [p for i, p in enumerate(stances) if lst[i]][:stop]
        binrank = len(fstances)
        binnum = int("".join(str(s) for s in fstances) or "0", 2)
        return binrank, binnum

    def _represent(self, s: str) -> str:
        if self.level == 0:
            return s.upper()
        elif self.level == 1:
            # Non-content chars represent themselves
            if s in [c for lvl in self.alphabet.breakers[self.level] for c in lvl]:
                return s
            if s in [c for lvl in self.alphabet.embedders[self.level] for c in lvl]:
                return s
            # Content chars are represented by their class
            for key, val in self.alphabet.content.items():
                if s in val:
                    return key
            raise ValueError(f"No representation for '{s}' on level {self.level}")
        else:
            return s

    def _compare(self, cand: str, mask: Mask) -> Tuple[int, int]:
        # Candidate string is compared to the mask via its representation
        rep = self._represent(cand)
        ln = len(mask.literals)
        # True if rep already matches the current (positive) position
        if mask.pos >= 0 and mask.match(rep):
            return mask.move(0)
        # Else try to move the position one step; true if rep matches
        if mask.match(rep, 1):
            return mask.move(1)
        # Else check if position is optional; try the next one if this one fails
        # When a non-optional position is reached, make the final decision
        incr = 1
        while ln > 1 and mask.optionals[mask.move(incr)[0] or 0]:
            incr += 1
            if mask.match(rep, incr):
                return mask.move(incr)
        # Return none if all else failed
        return None

    def _match_char(self, cand: str, d: int, node: Node, stances: List[int]):
        r = node.rank
        binrank, binnum = self._get_bins(stances[0], 2**r - 1)

        rev = bool(self.grules.revs[binrank][binnum])
        ret = bool(self.grules.rets[r][d])
        skip = bool(self.grules.skips[r][d])

        pstances = stances[0][-d:] if d > 0 else []
        masks = node.get_masks(d, pstances)
        fm, sm = masks if not rev else masks[::-1]
        fm.cyclic, sm.cyclic = not ret, not ret

        # Breaker moves the positions on both sides
        # The second side will be matched next
        if cand == self.alphabet.breakers[self.level]:
            print("Ignoring the breaker")
            fm.move(1, inplace=True)
            sm.move(1, inplace=True)
            return (-1, -1)

        # Try to match the candidate to the first half
        # Ensure the second half wasn't yet matched OR return is not restricted
        if sm.pos < 0 or not ret:
            print(f"-> Comparing {cand} with first mask {fm}")
            comp_res = self._compare(cand, fm)
            if comp_res:
                fm.pos = comp_res[0]
                fm.rep = comp_res[1]
                node.reset_masks(d, pstances, int(not rev))
                return (int(rev), comp_res[1])

        # If unsuccessful, proceed to the second half
        # Ensure the first half was already matched OR skipping is not restricted
        if fm.pos >= 0 or not skip:
            print(f"-> Comparing {cand} with second mask {sm}")
            comp_res = self._compare(cand, sm)
            if comp_res:
                sm.pos = comp_res[0]
                sm.rep = comp_res[1]
                node.reset_masks(d, pstances, int(rev))
                return (int(not rev), comp_res[1])

        # If both unsuccessful, the char's stance cannot be determined
        return None

    def _process_char(self, s: str) -> List[int]:
        tree = self.buff.tree
        stances = [[], []]
        # Iterating over the ranks of the tree
        for r in range(len(tree.struct)):
            node = tree.get_item_by_key(stances[0])
            # Determining the dichotomic stance of the char
            for d in range(tree.struct[r]):
                matches = self._match_char(s, d, node, stances)
                # No stance means failure, -1 means char to be skipped
                if matches is None:
                    return [[], []]
                elif matches == (-1, -1):
                    return [[-1], [-1]]
                else:
                    stances[0].append(matches[0])
                    stances[1].append(matches[1])
                    if matches[1] > 0:
                        ch = tree.get_item_by_key(stances[0], ignore_comp=True)
                        lemb = self.srules.lemb[ch.num]
                        if len(ch.compounds) < matches[1]:
                            if len(ch.compounds) >= lemb and lemb != -1:
                                print(
                                    f"Failed to map '{s}' to {ch}: too many compounds"
                                )
                                return [[], []]
                            else:
                                print(f"-> Embedding a compound at N({ch.num})")
                                tree.embed_compound(ch.num)
        return stances

    def _neutralize(self) -> bool:
        tree = self.buff.tree
        # Populate trivial terminal splits with neutral characters
        for n in tree.nodes:
            term = self.level == 1 and n.rank == len(tree.struct)
            dkey = [int(k) for k in n.key]
            neut = self.srules.tneuts[int(n.key, 2)][0] if term else ""
            perm = self.srules.tperms[int(n.key, 2)][0] if term else ""
            if term and n.parent and not n.content:
                if sum([len(ch.content) for ch in n.parent.children]):
                    nc = self.alphabet.content[neut][0]
                    if not any((self._represent(nc) in perm, nc in perm)):
                        print(f"Failed to map '{nc}' to {n}: not permitted")
                        return False
                    else:
                        tree.set_item_by_key(dkey, nc)
        return True

    def _map_char(self, m: List[int], c: str) -> bool:
        tree = self.buff.tree
        rep = self._represent(c)

        for n in tree.nodes:
            term = self.level == 1 and n.rank == len(tree.struct)
            dkey = [int(k) for k in n.key]
            lemb = self.srules.lemb[n.num]
            perm = self.srules.tperms[int(n.key, 2)][0] if term else ""
            if m[0][: len(dkey)] == dkey:
                if len(n.content) > lemb and lemb != -1:
                    print(f"Failed to map '{c}' to {n}: too many compounds")
                    return False
                if term and not any((rep in perm, c in perm)):
                    print(f"Failed to map '{c}' to {n}: not permitted")
                    return False
                else:
                    tree.set_item_by_key([dkey, m[1]], c)
        return True

    def _nonbinary_shift(self, maps: List[List[int]], i: int) -> None:
        m = maps[i][0]
        struct = self.buff.tree.struct
        lst = [r == 0 for s in struct for r in range(s)]
        for p, st in enumerate(lst):
            binrank, binnum = self._get_bins(m, p + 1)
            rev = bool(self.grules.revs[binrank][binnum])
            if not st:
                if m[p] == rev:
                    mc = maps[i + 1 :] if not rev else maps[:i]
                    if not any(mp[0][: p + 1] == m[:p] + [1 - m[p]] for mp in mc):
                        print(f"-> Caret moved at {m}")
                        m[1] = 1 - m[1]
        return

    def _apply(self, mapping: Dict) -> bool:
        # Go through the mappings and set them to the tree
        for i, _ in enumerate(mapping["Characters"]):
            c = mapping["Characters"][i]
            m = mapping["Keys"][i]
            # Move the mapping along the non-binary ranks to fill empty nodes
            self._nonbinary_shift(mapping["Keys"], i)
            # Find maps pointing to each node and apply them
            if self._map_char(m, c):
                continue
            else:
                return False

        return self._neutralize()

    def _produce(self, pst: str) -> bool:
        mapping = {"Characters": [], "Keys": []}
        # Processing the input string in one pass
        # Each char is processed separately
        for c in pst:
            print(f"Working with '{c}'")
            # Determine the stance of the char w.r.t all dichotomies
            stances = self._process_char(c)
            # If matching fails, halt the procedure
            if stances == [[], []]:
                print(f"Failed to match '{c}'")
                return False
            # If the char shouldn't be mapped, ignore it
            elif stances == [[-1], [-1]]:
                continue
            # Otherwise, map the char with the stances determined
            else:
                print(f"=> Assigned stances {stances}")
                mapping["Characters"].append(c)
                mapping["Keys"].append(stances)

        # Save the produced mappings of the chars
        self.buff.mapping = mapping

        return True

    def parse(self, input_string: str) -> bool:
        # Create and save the parsing tree
        tree = Tree(self.grules.struct)
        tree.set_permissions(self.loader.compute_permissions(self.level))

        # Excluding non-alphabetic chars
        pst = self._prepare(input_string)
        self.buff.tree = tree
        self.buff.pst = pst

        print(f"Parsing string '{input_string}' as '{pst}'")

        # Using general rules to produce the mappings
        general = self._produce(pst)
        if general:
            # Applying the mappings and the special rules
            if self.buff.mapping:
                special = self._apply(self.buff.mapping)
                if special:
                    return True
            else:
                print("The mapping is empty")
        else:
            print("Could not produce a mapping")

        return False
