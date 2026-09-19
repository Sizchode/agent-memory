"""Binding-aware beam search over extracted facts and their original sources.

The scorer ranks existing facts; shared variables and known endpoints constrain
candidate generation. Beam search and greedy set cover are standard operators,
not new guarantees of semantic correctness or optimal evidence selection.
"""

from collections import defaultdict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Match:
    bindings: dict[str, str]
    facts: tuple[tuple[int, int], ...]
    score: float


class FactJoinSearch:
    def __init__(self, index, normalize):
        self.index, self.normalize = index, normalize
        self.facts = {row[0]: tuple(row[1:]) for row in index.connection.execute(
            "SELECT id, subject, predicate, object FROM facts ORDER BY id")}
        self.subjects, self.objects = defaultdict(set), defaultdict(set)
        for fact, (subject, _, obj) in self.facts.items():
            self.subjects[normalize(subject)].add(fact)
            self.objects[normalize(obj)].add(fact)
        self.supports = defaultdict(set)
        for fact, source in index.connection.execute("SELECT DISTINCT fact, source FROM occurrences"):
            self.supports[fact].add(source)
        self.source_order = {row[0]: order for order, row in enumerate(
            index.connection.execute("SELECT id FROM sources ORDER BY rowid"))}

    @staticmethod
    def value(term, bindings):
        return bindings.get(term) if term.startswith("?") else term

    def search(self, patterns, score, beam_width=8):
        """score(pattern, subject, object, fact_ids) ranks the bound atom's candidates."""
        if not patterns:
            return []
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        states = [Match({}, (), 0.0)]
        for _ in patterns:
            expanded = []
            for state in states:
                used = {atom for atom, _ in state.facts}
                remaining = [index for index in range(len(patterns)) if index not in used]
                # Evaluate the most bound remaining atom first, as in join planning.
                position = max(remaining, key=lambda i: sum(self.value(term, state.bindings) is not None
                    for term in (patterns[i].subject, patterns[i].object)))
                pattern = patterns[position]
                subject, obj = (self.value(term, state.bindings) for term in (pattern.subject, pattern.object))
                selected = None
                for value, lookup in ((subject, self.subjects), (obj, self.objects)):
                    if value is not None:
                        candidates = lookup.get(self.normalize(value), set())
                        selected = set(candidates) if selected is None else selected & candidates
                candidates = sorted(self.facts if selected is None else selected)
                if pattern.subject == pattern.object:
                    candidates = [fact for fact in candidates if self.normalize(self.facts[fact][0]) ==
                                  self.normalize(self.facts[fact][2])]
                if not candidates:
                    continue
                values = score(pattern, subject, obj, candidates)
                if len(values) != len(candidates) or not all(math.isfinite(value) for value in values):
                    raise ValueError("The scorer must return one finite value per candidate")
                order = sorted(range(len(values)), key=lambda i: (-values[i], candidates[i]))[:beam_width]
                for rank in order:
                    fact = candidates[rank]
                    bindings = dict(state.bindings)
                    for term, value in ((pattern.subject, self.facts[fact][0]), (pattern.object, self.facts[fact][2])):
                        if term.startswith("?"):
                            if term in bindings and self.normalize(bindings[term]) != self.normalize(value):
                                raise AssertionError("Candidate generation broke a shared binding")
                            bindings[term] = value
                    expanded.append(Match(bindings, state.facts + ((position, fact),), state.score + float(values[rank])))
            states = sorted(expanded, key=lambda state: (-state.score, state.facts))[:beam_width]
            if not states:
                break
        return states

    def source_cover(self, match, budget):
        """Greedy unweighted set cover; reject incomplete covers at the budget."""
        if budget <= 0:
            raise ValueError("budget must be positive")
        remaining = {fact for _, fact in match.facts}
        sources = set().union(*(self.supports[fact] for fact in remaining)) if remaining else set()
        selected = []
        while remaining and len(selected) < budget:
            coverage = {source: {fact for fact in remaining if source in self.supports[fact]} for source in sources}
            source = min(sources, key=lambda source: (-len(coverage[source]), self.source_order[source]))
            if not coverage[source]:
                break
            selected.append(source)
            remaining -= coverage[source]
            sources.remove(source)
        return tuple(selected) if not remaining else ()
