"""Query-to-pattern prototype; inspired by SimGRAG, not its reproduction."""

from dataclasses import dataclass
import json


PROMPT = """Describe the factual evidence needed to answer the user's question as a graph pattern.
Do not answer the question or supply facts from your own knowledge.
Return a JSON object with exactly one field, atoms, containing a list of objects.
Each object has exactly four string fields: subject, relation, object, query.
Copy known entities and values directly from the question into subject or object as literals.
NEVER replace a known entity with a variable. Do not add naming or title relations to represent known names.
Use named variables starting with ? ONLY for unknown entities or values. Variable names contain no whitespace.
Use the same variable wherever the same unknown entity or value is needed.
relation describes the directed factual relation from subject to object in ordinary language.
Keep the subject and object in the order expressed by the relation; do not reverse their roles.
query is a natural-language request for that single relation, retaining relevant known entities and the type of unknown information.
Break relational dependencies into separate atoms. Do not put an unknown answer into a literal field.
Retrieve the facts needed for comparisons, exclusions, counting, and time constraints. Do not invent equality,
inequality, or comparison edges: those operations belong to answer generation, not the stored graph.
The original question will still be given to the answer model.
If the question cannot be represented with factual relations, return an empty atoms list.
Do not add explanations or Markdown."""


@dataclass(frozen=True)
class PatternAtom:
    subject: str
    relation: str
    object: str
    query: str


def messages(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question is required")
    return [dict(role="system", content=PROMPT), dict(role="user", content=question)]


def parse_pattern(content, question):
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"atoms"} or not isinstance(value["atoms"], list):
        raise ValueError("Expected a JSON object with an atoms list")
    result = []
    fields = {"subject", "relation", "object", "query"}
    for atom in value["atoms"]:
        if not isinstance(atom, dict) or set(atom) != fields:
            raise ValueError("Incorrect pattern fields")
        if not all(isinstance(atom[key], str) and atom[key].strip() for key in fields):
            raise ValueError("Pattern fields must be nonempty strings")
        for term in (atom["subject"], atom["object"]):
            if term.startswith("?"):
                if len(term) == 1 or any(character.isspace() for character in term):
                    raise ValueError("Variables require nonempty names without whitespace")
            elif term.casefold() not in question.casefold():
                raise ValueError("A literal was not copied from the question")
        result.append(PatternAtom(**atom))
    return tuple(result)
