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
        return bindings.get(term, None if term.startswith("?") else term)

    def search(self, patterns, score, beam_width=8, node_distances=None):
        """Rank bound atoms, optionally subtracting each known node's distance once.

        node_distances maps each literal to normalized candidate names and L2
        distances. Combined with negative relation L2 scores this uses SimGRAG's
        distance sum, but retains directed beam search, not its full algorithm.
        """
        if not patterns:
            return []
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        literal_candidates = {}
        if node_distances is not None:
            literals = {term for pattern in patterns for term in (pattern.subject, pattern.object)
                        if not term.startswith("?")}
            if set(node_distances) != literals:
                raise ValueError("Semantic matching requires candidates for every literal")
            for term, distances in node_distances.items():
                if any(self.normalize(name) != name or not math.isfinite(distance) or distance < 0
                       for name, distance in distances.items()):
                    raise ValueError("Node candidates require normalized names and finite nonnegative distances")
                for role, lookup in (("subject", self.subjects), ("object", self.objects)):
                    literal_candidates[term, role] = set().union(*(lookup.get(name, set()) for name in distances))
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
                for term, role, value, lookup in ((pattern.subject, "subject", subject, self.subjects),
                                                 (pattern.object, "object", obj, self.objects)):
                    if value is not None:
                        candidates = (literal_candidates[term, role]
                                      if node_distances is not None and term not in state.bindings
                                      else lookup.get(self.normalize(value), set()))
                        selected = set(candidates) if selected is None else selected & candidates
                candidates = sorted(self.facts if selected is None else selected)
                if pattern.subject == pattern.object:
                    candidates = [fact for fact in candidates if self.normalize(self.facts[fact][0]) ==
                                  self.normalize(self.facts[fact][2])]
                if not candidates:
                    continue
                values = list(score(pattern, subject, obj, candidates))
                if len(values) != len(candidates) or not all(math.isfinite(value) for value in values):
                    raise ValueError("The scorer must return one finite value per candidate")
                if node_distances is not None:
                    for i, fact in enumerate(candidates):
                        assignments = {pattern.subject: self.facts[fact][0], pattern.object: self.facts[fact][2]}
                        values[i] -= sum(node_distances[term][self.normalize(value)]
                                         for term, value in assignments.items()
                                         if term in node_distances and term not in state.bindings)
                order = sorted(range(len(values)), key=lambda i: (-values[i], candidates[i]))[:beam_width]
                for rank in order:
                    fact = candidates[rank]
                    bindings = dict(state.bindings)
                    for term, value in ((pattern.subject, self.facts[fact][0]), (pattern.object, self.facts[fact][2])):
                        if term.startswith("?") or node_distances is not None:
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
