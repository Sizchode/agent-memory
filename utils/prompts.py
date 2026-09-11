"""Official MemoryAgentBench memory and query prompt templates."""

SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."

# Cross-domain adapter for methods whose released extractor is written for
# personal dialogue.  It preserves the original document text and supplies
# LightMem's released JSON/source-id interface; it does not summarize, label,
# or curate the benchmark data.
DOCUMENT_FACT_EXTRACTION_PROMPT = """
You are a document fact extractor. Extract factual statements from the provided
document passages so they can be retrieved to answer later questions. Preserve
all named entities, numbers, dates, relations, and qualifications. Do not add
facts that are not stated in the passages. State each fact exactly once, using
the shortest self-contained wording that preserves those details; do not repeat
or elaborate the source text. Each extracted fact must use the
integer prefix before `.user` in the rendered input line as its source_id.
The `--- Topic N ---` header is internal grouping metadata: never copy N into
source_id.  Each document extraction batch contains one rendered line prefixed
`0.user`, so every returned source_id must be 0.  Numbers inside the document
text are not source IDs either.

Return exactly one JSON object:
{"data": [{"source_id": <integer>, "fact": "<standalone factual statement>"}]}
"""

DOCUMENT_FACT_EXTRACTION_INSTRUCTIONS = (
    "The input is a document collection rather than a personal conversation. "
    "Extract all factual statements that may answer future questions; preserve "
    "entities, numbers, dates, relations, and qualifications, and do not infer "
    "facts absent from the input. State each fact exactly once, using the shortest "
    "self-contained wording that preserves those details; do not repeat or elaborate "
    "the source text."
)

_QUERY = {
    "ruler": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
    "eventqa": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
    "factconsolidation": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
}


def hipporag_qa_messages(passages: list[str], question: str) -> list[dict[str, str]]:
    """Render HippoRAG 2's released RAG-QA demonstration for direct answering."""

    import importlib.util
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "baseline_algorithms" / "HippoRAG" / "src" / "hipporag" / "prompts" / "templates" / "rag_qa_musique.py"
    spec = importlib.util.spec_from_file_location("_official_hipporag_rag_qa", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official HippoRAG QA template from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_user = "".join(f"Wikipedia Title: {passage}\n\n" for passage in passages)
    prompt_user += f"Question: {question}\nAnswer: "
    one_shot_input = module.one_shot_rag_qa_input.replace("\nThought: ", "\nAnswer: ")
    one_shot_answer = module.one_shot_rag_qa_output.rsplit("Answer:", 1)[-1].strip()
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using the provided text passages. "
                "Return only the concise answer without reasoning or additional explanation."
            ),
        },
        {"role": "user", "content": one_shot_input},
        {"role": "assistant", "content": one_shot_answer},
        {"role": "user", "content": prompt_user},
    ]


def template_family(source: str) -> str:
    if source.startswith("ruler_qa"):
        return "ruler"
    if source.startswith("eventqa_"):
        return "eventqa"
    if source.startswith("factconsolidation_"):
        return "factconsolidation"
    raise ValueError(f"no official prompt family for source {source!r}")


def query_prompt(source: str, question: str) -> str:
    return _QUERY[template_family(source)].format(question=question)
