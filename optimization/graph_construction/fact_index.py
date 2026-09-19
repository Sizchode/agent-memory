"""Source-preserving triple occurrences and exact joins over candidate facts.

The index stores extracted assertions, not verified truths. Joins enforce shared
variable bindings; selecting relevant predicates remains the retriever's job.
"""

from collections import defaultdict
from dataclasses import dataclass
import sqlite3


@dataclass(frozen=True)
class Atom:
    subject: str
    object: str
    fact_ids: tuple[int, ...]


@dataclass(frozen=True)
class Witness:
    bindings: tuple[tuple[str, str], ...]
    fact_ids: tuple[int, ...]
    sources: tuple[tuple[str, ...], ...]


class FactIndex:
    def __init__(self, connection):
        self.connection = connection

    @classmethod
    def build(cls, path, documents):
        """Consume OpenIE documents only; preserve roles, strings and all sources."""
        if path.exists():
            raise FileExistsError(path)
        connection = sqlite3.connect(path)
        try:
            connection.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE sources (id TEXT PRIMARY KEY, passage TEXT NOT NULL);
                CREATE TABLE facts (id INTEGER PRIMARY KEY, subject TEXT NOT NULL,
                    predicate TEXT NOT NULL, object TEXT NOT NULL,
                    UNIQUE(subject, predicate, object));
                CREATE TABLE occurrences (source TEXT REFERENCES sources(id),
                    ordinal INTEGER NOT NULL, fact INTEGER REFERENCES facts(id),
                    PRIMARY KEY(source, ordinal));
                CREATE INDEX subject_index ON facts(subject);
                CREATE INDEX object_index ON facts(object);
                CREATE INDEX predicate_index ON facts(predicate);
                CREATE INDEX fact_sources ON occurrences(fact);
            """)
            for document in documents:
                source, passage = document["idx"], document["passage"]
                if not isinstance(source, str) or not isinstance(passage, str):
                    raise ValueError("Sources require string IDs and text")
                connection.execute("INSERT INTO sources VALUES (?, ?)", (source, passage))
                for ordinal, triple in enumerate(document["extracted_triples"]):
                    if not isinstance(triple, (list, tuple)) or len(triple) != 3 or not all(
                            isinstance(value, str) for value in triple):
                        raise ValueError("Expected an extracted subject/predicate/object triple")
                    connection.execute("INSERT OR IGNORE INTO facts(subject, predicate, object) VALUES (?, ?, ?)", triple)
                    fact, = connection.execute("SELECT id FROM facts WHERE subject=? AND predicate=? AND object=?", triple).fetchone()
                    connection.execute("INSERT INTO occurrences VALUES (?, ?, ?)", (source, ordinal, fact))
            connection.commit()
        except BaseException:
            connection.close()
            path.unlink()
            raise
        return cls(connection)

    @classmethod
    def open(cls, path):
        return cls(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True))

    def documents(self):
        """Reconstruct every input triple occurrence, including duplicates."""
        for source, passage in self.connection.execute("SELECT id, passage FROM sources ORDER BY rowid"):
            triples = self.connection.execute("""SELECT f.subject, f.predicate, f.object
                FROM occurrences o JOIN facts f ON f.id=o.fact
                WHERE o.source=? ORDER BY o.ordinal""", (source,)).fetchall()
            yield dict(idx=source, passage=passage, extracted_triples=[list(row) for row in triples])

    def joins(self, atoms):
        """Execute a conjunctive query over caller-supplied candidate fact IDs.

        A '?' prefix denotes a variable. Other strings are exact constants.
        Distinct sources of one fact are alternatives, not additional conjuncts.
        No similarity scores, dataset labels or question answers enter this step.
        """
        if not atoms:
            raise ValueError("A query must contain at least one atom")
        tables, conditions, parameters, bindings = [], [], [], {}
        candidate_ids = set()
        for index, atom in enumerate(atoms):
            if not isinstance(atom, Atom) or not all(isinstance(term, str) and term for term in (atom.subject, atom.object)):
                raise ValueError("Each atom needs subject/object terms and candidate fact IDs")
            ids = tuple(dict.fromkeys(atom.fact_ids))
            if not ids:
                return
            if not all(type(value) is int and value > 0 for value in ids):
                raise ValueError("Candidate fact IDs must be positive integers")
            existing = self.connection.execute(
                "SELECT count(*) FROM facts WHERE id IN (" + ",".join("?" for _ in ids) + ")", ids).fetchone()[0]
            if existing != len(ids):
                raise ValueError("Candidate fact IDs are absent from this index")
            candidate_ids.update(ids)
            alias = f"f{index}"
            tables.append(f"facts {alias}")
            conditions.append(f"{alias}.id IN (" + ",".join("?" for _ in ids) + ")")
            parameters.extend(ids)
            for term, field in ((atom.subject, "subject"), (atom.object, "object")):
                column = f"{alias}.{field}"
                if term.startswith("?"):
                    if term == "?":
                        raise ValueError("Variables require names")
                    if term in bindings:
                        conditions.append(f"{column}={bindings[term]}")
                    else:
                        bindings[term] = column
                else:
                    conditions.append(f"{column}=?")
                    parameters.append(term)
        fact_columns = [f"f{index}.id" for index in range(len(atoms))]
        columns = fact_columns + list(bindings.values())
        query = ("SELECT " + ", ".join(columns) + " FROM " + ", ".join(tables) +
                 " WHERE " + " AND ".join(conditions))
        sources = defaultdict(list)
        selected = tuple(sorted(candidate_ids))
        source_query = ("SELECT DISTINCT fact, source FROM occurrences WHERE fact IN (" +
                        ",".join("?" for _ in selected) + ") ORDER BY fact, source")
        for fact, source in self.connection.execute(source_query, selected):
            sources[fact].append(source)
        for row in self.connection.execute(query, parameters):
            facts = tuple(row[:len(atoms)])
            yield Witness(tuple(zip(bindings, row[len(atoms):], strict=True)), facts,
                          tuple(tuple(sources[fact]) for fact in facts))

    def close(self):
        self.connection.close()
