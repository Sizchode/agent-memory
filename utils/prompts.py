"""Official MemoryAgentBench memory and query prompt templates."""

SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."

_MEMORIZE = {
    "ruler": "Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
    "eventqa": "Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
    "factconsolidation": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
}

_QUERY = {
    "ruler": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
    "eventqa": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
    "factconsolidation": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
}


def template_family(source: str) -> str:
    if source.startswith("ruler_qa"):
        return "ruler"
    if source.startswith("eventqa_"):
        return "eventqa"
    if source.startswith("factconsolidation_"):
        return "factconsolidation"
    raise ValueError(f"no official prompt family for source {source!r}")


def memorize_prompt(source: str, context: str, time_stamp: str = "") -> str:
    return _MEMORIZE[template_family(source)].format(context=context, time_stamp=time_stamp)


def query_prompt(source: str, question: str) -> str:
    return _QUERY[template_family(source)].format(question=question)
