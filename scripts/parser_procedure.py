import csv
import json
import logging
import logging.config

from typing import Tuple, Optional, Generator
from pathlib import Path

from scripts.parser_entities import Mapping, Dichotomy, Tree, Node, Mask, Element
from scripts.parser_dataclasses import Alphabet, GeneralRules, SpecialRules
from scripts.parser_dataclasses import Dialect, Feature, Stance, Token, Symbol

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(format="[%(levelname)s] %(message)s")


class ParsingFailure(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"{message}")
        return


class Processor:
    """Orchestrates all language operations."""

    def __init__(
        self,
        max_level: int | None = None,
        path: str = "",
    ) -> None:
        self.max_level: int | None = max_level
        self.path: str = path
        self._load_params()
        return

    def _load_params(self) -> None:
        """Creates the components and loads the alphabet and the rules."""
        # Loading parameters
        loader = Loader(self.path)
        self.alphabet = loader.load_alphabet()
        self.grules = loader.load_grules()
        self.srules = loader.load_srules()
        self.dialect = loader.load_dialect()
        self.levels = range(len(self.grules.struct))
        # Creating components
        self.mapper = Mapper(self)
        self.masker = Masker(self)
        self.interpreter = Interpreter(self)
        self.streamer = Streamer(self)

    def process(self, instr: str, verbose: bool = False) -> None:
        """Parses the input string and applies the parsing to the trees."""
        logger.level = logging.DEBUG if verbose else logging.INFO
        logger.info(f"Parsing {instr}")
        # Resetting the parameters
        self.mapping = Mapping(self.levels)
        self.trees = [[] for lvl in self.levels]
        self.streamer.instr = instr
        self.masker.construct()
        # Parsing the string
        try:
            self.streamer.feed()
        except ParsingFailure:
            logger.info(f"Failed to parse {instr}")
            return False
        logger.info(f"Successfully parsed {instr} as {self.mapping.elems[-1]}")
        # Applying the obtained mapping to the trees
        for lvl in self.levels:
            if lvl < len(self.levels) - 1:
                elems = [e.preheader.content for e in self.mapping.elems[lvl + 1]]
            else:
                elems = [self.mapping.elems[lvl]]
            for es in elems:
                tree = Tree(self.grules.struct[lvl], lvl)
                self.trees[lvl].append(tree)
                self.interpreter.apply(es, tree)
                if self.max_level is None or self.max_level >= lvl:
                    self.interpreter.determine_ctype(tree)
                    self.interpreter.interpret(tree)
        return True

    def get_stances(self, lvl: int = -1) -> list[Stance]:
        """Produces the list of stances of the elements of the given level."""
        return [e.stance for e in self.mapping.elems[lvl]]


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


class Streamer:
    """Takes an input string and creates a stream of tokens wrapped into
    elements to be parsed, depth and level structure accounted.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc = prc
        self.e: Element | None = None
        self.t: Token | None = None
        return

    def _tokenize(self) -> Generator[Token, None, None]:
        """Performs the splitting, alphabetic filtering, symbolization,
        and tokenization of the input string.
        """
        t: Token | None = None
        linstr = [ch for ch in self.instr]
        for i, ch in enumerate(linstr):
            # Wrapping the char into a symbol if it is alphabetic
            subs = self.prc.alphabet.substitutions
            if ch in subs:
                sub = subs[ch]
                # If the replacement consists of more than one symbol,
                # add the additional symbols to the string
                for j, s in enumerate(sub[1:]):
                    linstr.insert(i + j + 1, s)
                ch = sub[0]
            if ch not in self.prc.alphabet.lookup:
                continue
            params = self.prc.alphabet.lookup[ch]
            s = Symbol(ch, *params.values(), i)
            # Wrapping the base symbol into a token (if base)
            # or modifying the previous one (if modifier)
            # The token is yielded as the next base symbol is read
            # or at the end of the string
            if isinstance(t, Token):
                if s.acat == "Base":
                    yield t
                elif s.acat == "Modifier" and s.aclass == t.base.aclass:
                    t.modifiers.append(s)
            if s.acat == "Base":
                t = Token(base=s)
        if isinstance(t, Token) and s.aclass != "Separator":
            yield t

    def feed(self) -> bool:
        """Processes the stream of tokens according to their alphabetic parameters.
        Content tokens are parsed as language elements, others set the parameters
        of the processor.
        """
        stream = self._tokenize()
        # Exhausting the stream
        while True:
            t = next(stream, None)
            # When the last token is reached, separation and popping is enforced
            if t is None:
                self.pop(max(self.prc.levels) - 1, deep=True)
                self.separate(max(self.prc.levels), deep=True)
                return True
            self.t = t
            lvl = t.base.level
            if t.base.asubcat == "Guiding":
                # Separating intervals of elements
                if t.base.aclass == "Separator":
                    self.separate(lvl)
                # Decreasing depth of complex embedding
                elif t.is_popper(lvl) and self.prc.mapping.cur_dpt[t.base.level] > 0:
                    self.pop(lvl)
                # Increasing depth of complex embedding
                elif t.is_pusher(lvl):
                    self.push(lvl)
            # Parsing content tokens while accounting for early & late breakers
            if t.base.asubcat == "Content" or t.base.aclass == "Wildcard":
                self.account_breaker(late=False)
                self.e = Element(t, level=0)
                self.add()
                self.account_breaker(late=True)

    def add(self) -> None:
        """Adds the current element to stack, parses and closes it."""
        lvl = self.e.level
        self.e.stance = Stance(depth=self.prc.mapping.cur_dpt[lvl])
        self.prc.mapper.close(self.e)
        self.parse()
        stack = self.prc.mapping.get_stack(lvl)
        stack.append(self.e)
        return

    def separate(self, lvl: int, deep: bool = False) -> None:
        """Wraps the interval of elements limited by the current border position
        into an element of the higher level and adds it to corresponding stack.
        """
        for level in self.prc.levels:
            if level == lvl or deep:
                # If the current depth of embedding is above zero on the same level,
                # pop until it reaches zero
                interval = self.prc.mapping.get_interval(level)
                if interval and level < len(self.prc.levels) - 1:
                    self.e = Element(interval, level=level + 1)
                    self.add()
                    self.prc.masker.construct(level)
                    self.prc.mapping.update_interval(level)
                if level > 0:
                    self.pop(lvl, deep=True)
        return

    def pop(self, lvl: int, deep: bool = False) -> None:
        """Decreases the current depth of complex embedding, wraps the current
        stack interval into an element of the same level and parses it.
        """
        if lvl > 0:
            self.separate(lvl - 1, deep=True)
        while self.prc.mapping.cur_dpt[lvl] > 0:
            # Popping only operates on the latest item in the element buffer,
            # which must also be a list of elements
            content = self.prc.mapping.elems[lvl][-1]
            if not isinstance(content, list):
                return
            elif len(content) == 0:
                del self.prc.mapping.elems[lvl][-1]
                return
            e = Element(content, level=lvl)
            self.e = e
            self.prc.mapper.close(e)
            self.prc.masker.construct(lvl, self.prc.mapping.cur_dpt[lvl])
            self.prc.mapping.cur_breaks[lvl][self.prc.mapping.cur_dpt[lvl]] = 0
            self.prc.mapping.elems[lvl][-1] = e
            self.prc.mapping.cur_dpt[lvl] -= 1
            self.parse()
            dpt = self.prc.mapping.cur_dpt[lvl]
            logger.debug(f"-> Depth at level {lvl} decreased to {dpt}")
            if not deep:
                break
        return

    def push(self, lvl: int) -> None:
        """Decreases the current depth of complex embedding.
        Performs separation on the previous level if necessary.
        """
        if lvl > 0:
            self.separate(lvl - 1)
        self.prc.mapping.elems[lvl].append([])
        self.prc.mapping.cur_dpt[lvl] += 1
        if self.prc.mapping.cur_dpt[lvl] >= len(self.prc.mapping.cur_breaks[lvl]):
            self.prc.mapping.cur_breaks[lvl].append(0)
        dpt = self.prc.mapping.cur_dpt[lvl]
        logger.debug(f"-> Depth at level {lvl} increased to {dpt}")
        return

    def account_breaker(self, late: bool) -> None:
        """Scans the modifiers of the current token for late or early breakers
        and increases the current breaker values as needed.
        """
        lvl = self.t.base.level
        dpt = self.prc.mapping.cur_dpt[lvl]
        for mod in self.t.modifiers:
            if mod.asubcat == "Breaker" and mod.quality == int(late):
                self.prc.mapping.cur_breaks[lvl][dpt] += mod.index + 1
                brk = self.prc.mapping.cur_breaks[lvl][dpt]
                logger.debug(
                    f"-> Breaker count at level {lvl}, depth {dpt} increased to {brk}"
                )
        return

    def parse(self) -> bool:
        """Passes the current element to the mapper method that determines
        its dichotomic stance.
        """
        lvl = self.e.level
        # Skipping levels beyond the set maximum
        if self.prc.max_level is not None and self.prc.max_level < lvl:
            return True
        # Elements with lists of elements as content must have heads
        if not self.e.molar:
            self.e.set_head(self.prc.grules.heads[lvl - 1], fallback=True)
        self.prc.mapper.determine_stance(self.e)
        logger.debug(f"=> Assigned the stance {self.e.stance} to {self.e}")
        return True


class Masker:
    """Creates, holds, and manipulates the dichotomies and masks as needed
    by the parsing procedure.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc: Processor = prc
        # Level > Depth > Rank > Dichotomy
        self.masks: list[list[list[list[Dichotomy]]]] = [
            [] for lvl in range(len(self.prc.grules.struct))
        ]
        self.construct()
        return

    def construct(self, level: int | None = None, depth: int | None = None) -> None:
        """Creates the hierarchy of dichotomies loaded with mask pairs.
        If level and depth are given, create or recreate the corresponding
        particular set of dichotomies.
        """
        struct = self.prc.grules.struct
        perms = self.prc.grules.perms
        revs = self.prc.grules.revs
        dembs = self.prc.grules.dembs
        wilds = self.prc.grules.wilds
        rets = self.prc.grules.rets
        skips = self.prc.grules.skips
        splits = self.prc.grules.splits
        lembs = self.prc.grules.lembs
        tneuts = self.prc.srules.tneuts

        for lvl in range(len(struct)):
            # Skip the irrelevant and reset the masks on relevant levels
            if level is not None and level != lvl:
                continue
            if depth is None:
                self.masks[lvl] = [[]]
            else:
                while len(self.masks[lvl]) < (depth or 0) + 1:
                    self.masks[lvl].append([])
                self.masks[lvl][depth or 0] = []

            # Define which dichotomies are last on rank & which are non-binary
            dich_num = sum(struct[lvl])
            lds = [r == len(range(s)) - 1 for s in struct[lvl] for r in range(s)]
            nbs = [r != 0 for s in struct[lvl] for r in range(s)]

            ranks = []
            for d in range(dich_num):
                dichs = []
                for p in range(0, len(perms[lvl][d]), 2):
                    left_perms = perms[lvl][d][p : p + 2][0]
                    right_perms = perms[lvl][d][p : p + 2][1]
                    left_mask = Mask(left_perms, dich_num - d, p)
                    right_mask = Mask(right_perms, dich_num - d, p + 1)

                    left_mask.rev = revs[lvl][d][p : p + 2][0]
                    right_mask.rev = revs[lvl][d][p : p + 2][1]
                    left_mask.demb = dembs[lvl][d][p : p + 2][0]
                    right_mask.demb = dembs[lvl][d][p : p + 2][1]
                    left_mask.wild = wilds[lvl][d][p : p + 2][0]
                    right_mask.wild = wilds[lvl][d][p : p + 2][1]

                    # Compound embedding only for the last dichs on rank
                    rlembs, rlds = lembs[lvl][::-1][d], lds[::-1][d]
                    left_mask.lemb = rlembs[p : p + 2][0] if rlds else 0
                    right_mask.lemb = rlembs[p : p + 2][1] if rlds else 0
                    left_mask.depth = depth or 0
                    right_mask.depth = depth or 0
                    left_mask.logger = logger
                    right_mask.logger = logger

                    # Neutral elements only for the lowest terminal masks
                    if d == 0 and lvl == 0:
                        left_mask.tneuts = tneuts[p : p + 2][0]
                        right_mask.tneuts = tneuts[p : p + 2][1]

                    dich = Dichotomy(level=lvl, d=dich_num - d - 1, nb=nbs[d])
                    dich.terminal = d == 0
                    dich.left, dich.right = [left_mask, right_mask]
                    dich.rev = min(revs[lvl][d][p : p + 2]) if d < 1 else 0
                    dich.ret = rets[lvl][d]
                    dich.skip = skips[lvl][d]
                    dich.split = splits[lvl][d]
                    dichs.append(dich)

                ranks.append(dichs)

            self.masks[lvl][depth or 0] = ranks[::-1]

        return

    def _find_dichs(self, num_key: list[int], depth: int, lvl: int) -> list[Dichotomy]:
        """Returns dichotomies whose keys start with the given one."""
        if len(self.masks[lvl]) <= depth:
            self.construct(lvl, depth)
        dichs = [mp for r in self.masks[lvl][depth] for mp in r]
        out = []
        for dich in dichs:
            if num_key == dich.num_key[: len(num_key)]:
                out.append(dich)
        return out

    def get_dichs(
        self, stance: Stance, lvl: int, downstream: bool = False
    ) -> Dichotomy | list[Dichotomy]:
        """Returns the dichotomy with the key defined by the stance.
        If downstream is True, also returns every dichotomy downstream of it.
        """
        dichs = self._find_dichs(stance.pos, stance.depth, lvl)
        out = dichs if downstream else dichs[0]

        if out:
            return out
        else:
            raise ValueError(f"Could not find dichotomy by stance {stance}")

    def set_dichotomy(self, dich: Dichotomy, comp: Tuple[int, int], level: int) -> None:
        """Records the given tuple of pos and rep to the pointed mask.
        If non-terminal, resets dichotomies downstream of the other mask.
        If rep is increased, also resets those downstream the pointed mask.
        """
        target_mask = dich.masks[dich.pointer]
        other_mask = dich.masks[1 - dich.pointer]
        if not dich.terminal:
            self.reset_dichotomies(level, other_mask.depth, other_mask.num_key)
            if target_mask.rep < comp[1]:
                self.reset_dichotomies(level, target_mask.depth, target_mask.num_key)
        target_mask.pos, target_mask.rep = comp
        return

    def reset_dichotomies(
        self,
        level: int,
        depth: int,
        num_key: Optional[list[int]] = None,
        total: bool = False,
    ) -> None:
        """Sets the pointers of dichotomies with and downstream of the given key
        to None. Used to reset the masks of one branch when the pointer is set
        to the other, as well as to prepare for parsing the next element.
        """
        stance = Stance(pos=num_key or [], rep=[], depth=depth)
        dichs = self.get_dichs(stance, level, downstream=True)
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

    def __init__(self, prc: Processor) -> None:
        self.prc: Processor = prc
        self.e: Element | None = None
        return

    def _decide_dichotomy(self, dich: Dichotomy, forbid_shift: bool = False) -> bool:
        """Produces the decision for the current element and dichotomy, linking
        the former to either first or second mask of the latter.
        """
        lvl, dpt = self.e.level, self.e.stance.depth
        # Conditions of fit for the 1st and 2nd masks
        conds = [
            any((not dich.pointer == 1, not dich.ret)),
            any((dich.pointer == 0, not dich.skip)),
        ]
        # Results of fit for the masks
        comps = [
            dich.masks[0].compare(self.e, dich.split),
            dich.masks[1].compare(self.e, dich.split),
        ]
        fit = None
        # Skip to the second mask if the breaker count is positive
        if self.prc.mapping.cur_breaks[lvl][dpt] > 0:
            self.prc.mapping.cur_breaks[lvl][dpt] -= 1
            if not comps[1]:
                return False
            # Breaking is permanent, so fitting to the first mask is now forbidden
            dich.masks[0].freeze = True
            fit = 1

        # Determine the fit the normal way
        if not fit:
            # If both masks are fitting, choose the first one unless
            # it is the only one that increases rep
            if all([conds[0], comps[0], conds[1], comps[1]]):
                if comps[1][1] == dich.masks[1].rep and comps[0][1] > dich.masks[0].rep:
                    fit = 1
                else:
                    fit = 0
            # First mask fitting (the second one wasn't fit OR ret is not forbidden)
            elif all([conds[0], comps[0]]):
                fit = 0
            # Second mask fitting (the first one wasn't fit OR skip is not forbidden)
            elif all([conds[1], comps[1]]):
                fit = 1
            # Otherwise and if the dich is non-binary, perform a shift and try again
            elif dich.nb and not forbid_shift:
                self._shift_nonbinary_mappings(dich, invert=True, force=True)
                return self._decide_dichotomy(dich, forbid_shift=True)
            # If all fails
            else:
                logger.warning(f"=> Could not decide {dich} for {self.e}")
                return False

        # Attempt closure if the obtained fit flips the pointer to 1
        if not dich.terminal and (dich.pointer or 0) != fit:
            closure = self._close_dichotomies(dich)
            if not closure:
                logger.warning(f"=> Could not close {dich}")
                return False

        dich.pointer = fit
        old_mask = f"{dich.masks[fit]}"

        # Prepare the decision
        self.prc.masker.set_dichotomy(dich, comps[fit], lvl)
        pos = fit if not dich.rev else 1 - fit
        rep = dich.masks[fit].rep

        new_mask = f"{dich.masks[fit]}"
        num_strings = ["1st", "2nd"]
        content = f"-> Fitting {repr(self.e.head)} to the {num_strings[fit]}"
        if dich.split:
            logger.debug(f"{content} mask {new_mask}")
        else:
            logger.debug(f"{content} mask {old_mask} → {new_mask}")

        self.e.stance.pos.append(pos)
        self.e.stance.rep.append(rep)

        return True

    def _shift_nonbinary_mappings(
        self, dich: Dichotomy, invert: bool = False, force: bool = False
    ) -> bool:
        """Shifts the mappings of the elements in the given list (or in the
        current stack if none are given) from the first to the second mask
        of the dichotomy (or vice versa if invert is True) continuously slot by slot
        as long as the shift produces a valid mapping.
        """
        # Shift is only possible if the target mask has an empty slot.
        # Equivalent mappings are shifted together or not at all,
        # compounds from closest to farthest to the target mask
        # and only if they fit (given lembs and perms).
        # Get keys for both masks
        level = dich.level
        mapping = self.prc.mapping
        elems = mapping.get_stack(level, interval=True)

        mask_from, mask_to = dich.masks if not invert else dich.masks[::-1]
        matches_from = mapping.enumerate_elems(mask_from.num_key, elems, dich.d)
        matches_to = mapping.enumerate_elems(mask_to.num_key, elems, dich.d)

        # Skip the shift to a non-empty mask unless forced
        num_occupied = len(matches_to)
        if not force and num_occupied > 0:
            return True

        # Setting the pointer to activate the target mask
        dich.pointer = 1 - int(invert)

        # Perform the shift slot by slot
        slots_shifted, elems_shifted = 0, 0
        num = mask_to.lemb + 1 - num_occupied
        items = reversed(matches_from) if not dich.rev else matches_from
        for n, slot in enumerate(items):
            slot_in_process = False
            if num > 0:
                for i in matches_from[slot]:
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
        if elems_shifted > 0:
            self.prc.masker.reset_dichotomies(level, mask_from.depth, mask_from.num_key)
            mask_from.subtract(elems_shifted, slots_shifted)

        return True

    def _fill_empty_terminals(self, dich: Dichotomy) -> bool:
        """Adds neutral elements to mirror those with no siblings at terminal node
        within the given list (or in the current stack if none are given).

        Only applicable to the zeroth level.
        """
        level, depth = dich.level, dich.depth
        if level != 0:
            return True
        elems = self.prc.mapping.get_stack(level, interval=True)

        left_nk, right_nk = dich.masks[0].num_key, dich.masks[1].num_key
        left_matches = self.prc.mapping.enumerate_elems(left_nk, elems, dich.d)
        right_matches = self.prc.mapping.enumerate_elems(right_nk, elems, dich.d)

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
            g = self.prc.alphabet.get_token(neut_mask.tneuts[depth])
            neut = Element(g, op_stance, level)
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
                if isinstance(self.e.content, list):
                    if self.e.content[0].level != self.e.level:
                        self.e.content.insert(insert_index + slot, neut)
                bdr = self.prc.mapping.cur_bdr[level]
                stack = self.prc.mapping.get_stack(level)
                stack.insert(bdr + insert_index + slot, neut)

        return True

    def _validate_mapping(self) -> bool:
        """Checks that every stance in the current element's content
        complies with terminal permissions.
        """
        level = 0
        elems = self.e.content

        cnt = 0
        addr = None
        for i, e in enumerate(elems):
            if e.level != 0:
                continue
            cnt = cnt + 1 if e.stance.pos + e.stance.rep[:-1] == addr else 0
            addr = e.stance.pos + e.stance.rep[:-1]

            rev = bool(self.prc.grules.revs[level][0][e.num])
            addrs = [e for e in elems if e.stance.pos + e.stance.rep[:-1] == addr]

            perms = self.prc.srules.tperms[e.num]
            priority = cnt if not rev else len(addrs) - 1 - cnt
            perm = perms[min(e.stance.depth, len(perms) - 1)][priority]

            aclass = e.head.content.base.aclass
            base = str(e.head.content.base)
            if not any([base in perm or aclass in perm, aclass == "Wildcard"]):
                logger.warning(
                    f"-> No permission for '{e.head}' [{aclass}] at {e.stance}"
                )
                return False

        return True

    def _close_dichotomies(self, dich: Dichotomy | None = None) -> bool:
        """For dichotomies downstream of the given dichotomy, performs
        the finalizing operations: shift the mappings for the non-binary ones,
        add neutral elements as needed for the terminal ones.

        If no dichotomy is given, starts from the last fitted branch
        of the topmost dichotomy for the current element.
        """

        if dich is None:
            elems = self.e.content
            level, depth = elems[0].level, elems[0].stance.depth
            dich = self.prc.masker.get_dichs(Stance(depth=depth), level)
        else:
            elems = None
            level, depth = dich.level, dich.depth

        res = True
        stance = Stance(pos=dich.masks[dich.pointer or 0].num_key, depth=depth)
        dichs = self.prc.masker.get_dichs(stance, level, downstream=True)
        invert = bool(dich.pointer or 0)

        for cdich in dichs:
            if cdich.nb:
                res = res and self._shift_nonbinary_mappings(cdich, invert=invert)
            if cdich.terminal:
                res = res and self._fill_empty_terminals(cdich)

        return res

    def _fit_element(
        self,
        e: Element,
        stance: Optional[Stance] = None,
        d: Optional[int] = None,
        term_only: bool = False,
        force_mov: bool = False,
    ) -> bool:
        """Records the element if it can be fit with the given stance."""
        if stance is None:
            stance = e.stance
        for p, pos in enumerate(stance.pos):
            if (term_only and p != len(stance.pos) - 1) or (d is not None and p < d):
                continue
            part_stance = stance.copy(p)
            dich = self.prc.masker.get_dichs(part_stance, e.level)
            cur_mask = pos if not dich.rev else 1 - pos

            comp = dich.masks[cur_mask].compare(e, dich.split, force_mov)
            if comp:
                e.stance = stance
                dich.pointer = cur_mask
                self.prc.masker.set_dichotomy(dich, comp, e.level)
            else:
                return False
        return True

    def determine_stance(self, e: Element) -> bool:
        """Sets the given element as current and  produces the stance for it
        by deciding the dichotomies.
        """
        self.e = e
        if e.stance is None:
            e.stance = Stance()
        for d in range(0, sum(self.prc.grules.struct[e.level])):
            dich = self.prc.masker.get_dichs(e.stance, e.level)
            if not self._decide_dichotomy(dich):
                raise ParsingFailure(f"Failed to parse {e}")
        return True

    def close(self, e: Element) -> bool:
        """Sets the given element as current, closes the corresponding dichotomy
        and applies special rules to validate its content.
        """
        self.e = e
        if e.molar:
            return True

        if not self._close_dichotomies():
            raise ParsingFailure("Failed to close the dichotomies")

        if not self._validate_mapping():
            raise ParsingFailure("Failed to validate the mapping")

        return True


class Interpreter:
    """Commits the given mapping to the dichotomic tree and provides dialectic
    interpretations to the functions of its nodes with the received arguments.
    """

    def __init__(self, prc: Processor) -> None:
        self.prc = prc
        return

    def apply(self, elems: list[Element], tree: Tree) -> None:
        """Applies the given elements to the given tree using their stances.
        Embeds complexes and compounds in the tree as needed.
        """
        finals = [r + 1 == s for s in tree.struct for r in range(s)]
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
            if not e.molar and e.content[0].level == e.level:
                base_node = tree.get_nodes(e.stance)
                tree.embed_complex(base_node)
                logger.debug(f"Embedded a complex at {base_node}")
                self.apply(e.content, base_node.complexes[-1])

            if e.level > 0:
                sep = self.prc.alphabet.separators[e.level - 1]
            else:
                sep = ""
            tree.set_element(e, sep, set_all=True)

        return

    def determine_ctype(self, tree: Tree) -> None:
        """Determines the composition type of the element recorded in the tree."""
        # Try different types one by one
        # Types specific for the depth level go first, general types last
        ctypes = self.prc.dialect.ctypes[tree.level]
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
            logger.warning(f"Illegal composition type at level {tree.level}")

        # Do the same for all embedded trees
        for node in [n for n in tree.all_nodes if n.complexes]:
            for c in node.complexes:
                self.determine_ctype(c)

        return

    def interpret(self, tree: Tree) -> None:
        """Inscribes interpretations defined in the dialect to the tree nodes
        depending on their content.
        """
        # Find features for the tree nodes and their compounds
        nodes = [n for n in tree.all_nodes if n.terminal and n.content]
        for node in nodes:
            if not node.content:
                continue
            e = node.content[0]
            feature = self.prc.dialect.get_feature(
                e.header.content.base.index,
                e.header.content.base.aclass,
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
                self.interpret(c)
        return

    def describe(
        self,
        tree: Tree | int | None = None,
        verbose: bool = False,
        prefix: str = "·",
    ) -> None:
        """Prints out a summary of interpreted features currently loaded
        into the tree.
        """
        if tree is None:
            tree = self.prc.trees[0][0]
        elif isinstance(tree, int):
            tree = self.prc.trees[0][tree]

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

    def draw_tree(
        self,
        tree: Tree | int | None = None,
        features: bool = False,
        all_nodes: bool = False,
    ) -> None:
        """Prints out the given dichotomic tree with mapped elements."""
        if tree is None:
            tree = self.prc.trees[0][-1]
        elif isinstance(tree, int):
            tree = self.prc.trees[tree][-1]

        st = str(tree)

        if tree.stance:
            st += f" at {tree.stance}"
        st += "\n"
        st += tree.draw(features=features, all_nodes=all_nodes)

        print(st)

        return

    def gloss(
        self,
        tree: Tree | int | None = None,
        to_gloss: str | list[Node] | None = None,
    ) -> None:
        """Iterates the terminal nodes of the tree and replaces the representations
        of their contents with the gloss strings defined by their features.
        If to_gloss is a string, processes it first.
        If to_gloss is a list of nodes, glosses them.
        """
        if tree is None:
            tree = self.prc.trees[0][-1]
        elif isinstance(tree, int):
            tree = self.prc.trees[0][tree]

        if to_gloss is None:
            logger.debug(f"{tree}")
            items = tree.get_interpretable_nodes(complexes=True)
        elif isinstance(to_gloss, str):
            self.prc.process(to_gloss)
            self.gloss(self.prc.trees[0][-1])
        elif isinstance(to_gloss, list):
            items = to_gloss
        else:
            raise ValueError(f"Invalid input to gloss: {to_gloss}")

        glosses, current_glossless = [], ""
        for item in items:
            if isinstance(item, list):
                if current_glossless:
                    glosses.append(f"{current_glossless}-")
                    current_glossless = ""
                glosses.append(f"[{self.gloss(to_gloss=item)}]")
            elif item.feature.argument_gloss:
                if current_glossless:
                    glosses.append(f"{current_glossless}-")
                    current_glossless = ""
                glosses.append(f"{item.feature.argument_gloss}-")
            else:
                current_glossless += "".join(str(e) for e in item.content)

        gloss = "".join(glosses).strip("-")

        return gloss
