import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
llm = ChatOpenAI(
    model="meta/llama-4-maverick-17b-128e-instruct",      
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0,
    streaming=True
)  

class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False."
    )

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that can directly answer the question."
    )

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs."
    )

class SelfRAG:
    def __init__(self, pdf_paths: List[str]):
        self.stream_handler = None
        docs = []
        for path in pdf_paths:
            docs.extend(PyPDFLoader(path).load())
            
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=150
        ).split_documents(docs)
        
        embeddings = NVIDIAEmbeddings(
            model="nvidia/llama-nemotron-embed-1b-v2",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        self.app = self._build_graph()

    def decide_retrieval(self, state: State):
        return {"need_retrieval": True}

    def route_after_decide(self, state: State) -> Literal["generate_direct", "retrieve"]:
        return "retrieve" if state.get("need_retrieval") else "generate_direct"

    def generate_direct(self, state: State):
        direct_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Answer using only your general knowledge.\n"
             "If it requires specific information from the uploaded documents, say:\n"
             "'I don't know based on my general knowledge. Please ask about the uploaded documents.'"),
            ("human", "{question}"),
        ])
        kwargs = {"config": {"callbacks": [self.stream_handler]}} if self.stream_handler else {}
        out = llm.invoke(direct_generation_prompt.format_messages(question=state["question"]), **kwargs)
        return {"answer": out.content}

    def retrieve(self, state: State):
        q = state.get("retrieval_query") or state["question"]
        return {"docs": self.retriever.invoke(q)}

    def is_relevant(self, state: State):
        return {"relevant_docs": state.get("docs", [])}

    def route_after_relevance(self, state: State) -> Literal["generate_from_context", "no_answer_found"]:
        if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
            return "generate_from_context"
        return "no_answer_found"

    def generate_from_context(self, state: State):
        rag_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a strict AI assistant answering questions based ONLY on the provided CONTEXT.\n"
             "Task:\n"
             "Provide a highly detailed answer to the question STRICTLY based on the context.\n"
             "If the context does not contain the answer, you MUST reply EXACTLY with 'I cannot answer this based on the provided documents.'\n"
             "UNDER NO CIRCUMSTANCES should you use outside knowledge, guess, or hallucinate information."),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ])
        valid_chunks = [d.page_content.strip() for d in state.get("relevant_docs", []) if d.page_content.strip()]
        context = "\n\n---\n\n".join(valid_chunks)
        if not context:
            return {"answer": "No answer found. The uploaded documents might be empty or scanned images without readable text.", "context": ""}
        kwargs = {"config": {"callbacks": [self.stream_handler]}} if self.stream_handler else {}
        out = llm.invoke(rag_generation_prompt.format_messages(question=state["question"], context=context), **kwargs)
        return {"answer": out.content, "context": context}

    def no_answer_found(self, state: State):
        return {"answer": "No answer found.", "context": ""}

    def is_sup(self, state: State):
        issup_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
             "Return JSON with keys: issup, evidence.\n"
             "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
             "How to decide issup:\n"
             "- fully_supported:\n"
             "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
             "  any qualitative/interpretive words that are not present in CONTEXT.\n"
             "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
             "  supports professional development, best-in-class, employee-first, etc.)\n\n"
             "- partially_supported:\n"
             "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
             "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
             "  or inferring outcomes like 'supports professional development').\n\n"
             "- no_support:\n"
             "  The key claims are not supported by CONTEXT.\n\n"
             "Rules:\n"
             "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
             "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
             "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
             "- Do not use outside knowledge."),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"),
        ])
        issup_llm = llm.with_structured_output(IsSUPDecision)
        decision = issup_llm.invoke(issup_prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")))
        return {"issup": decision.issup, "evidence": decision.evidence}

    def route_after_issup(self, state: State) -> Literal["accept_answer", "revise_answer"]:
        if state.get("issup") == "fully_supported":
            return "accept_answer"
        MAX_RETRIES = 2
        if state.get("retries", 0) >= MAX_RETRIES:
            return "accept_answer"
        return "revise_answer"

    def accept_answer(self, state: State):
        return {}

    def revise_answer(self, state: State):
        revise_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a STRICT reviser.\n\n"
             "Your previous answer contained unsupported claims, hallucinated phrasing, or outside knowledge.\n"
             "Task:\n"
             "Rewrite the answer STRICTLY based ONLY on the provided CONTEXT.\n"
             "Rules:\n"
             "- Explain using ONLY the facts explicitly found in the CONTEXT.\n"
             "- ABSOLUTELY NO outside knowledge. If the context doesn't say it, do not include it.\n"
             "- If the context doesn't contain enough info to answer, simply say 'I cannot answer this based on the provided documents.'\n"),
            ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
        ])
        kwargs = {"config": {"callbacks": [self.stream_handler]}} if self.stream_handler else {}
        out = llm.invoke(revise_prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")), **kwargs)
        return {"answer": out.content, "retries": state.get("retries", 0) + 1}

    def is_use(self, state: State):
        isuse_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
             "Goal:\n"
             "- Decide if the answer actually addresses what the user asked.\n\n"
             "Return JSON with keys: isuse, reason.\n"
             "isuse must be one of: useful, not_useful.\n\n"
             "Rules:\n"
             "- useful: The answer directly answers the question or provides the requested specific info.\n"
             "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
             "- Do NOT use outside knowledge.\n"
             "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
             "- Keep reason to 1 short line."),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
        ])
        isuse_llm = llm.with_structured_output(IsUSEDecision)
        decision = isuse_llm.invoke(isuse_prompt.format_messages(question=state["question"], answer=state.get("answer", "")))
        return {"isuse": decision.isuse, "use_reason": decision.reason}

    def route_after_isuse(self, state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
        if state.get("isuse") == "useful":
            return "END"
        MAX_REWRITE_TRIES = 1
        if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
            return "no_answer_found"
        return "rewrite_question"

    def rewrite_question(self, state: State):
        rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Rewrite the user's QUESTION into a query optimized for vector retrieval over the uploaded documents.\n\n"
             "Rules:\n"
             "- Keep it short (6-16 words).\n"
             "- Preserve key entities.\n"
             "- Add 2-5 high-signal keywords.\n"
             "- Remove filler words.\n"
             "- Do NOT answer the question.\n"
             "- Output JSON with key: retrieval_query\n\n"
             "Examples:\n"
             "Q: 'What is the main topic of chapter 2?'\n"
             "-> {{'retrieval_query': 'main topic chapter 2 summary overview'}}\n\n"
             "Q: 'How does the algorithm work?'\n"
             "-> {{'retrieval_query': 'algorithm working mechanism explanation step by step'}}"),
            ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}"),
        ])
        rewrite_llm = llm.with_structured_output(RewriteDecision)
        decision = rewrite_llm.invoke(rewrite_for_retrieval_prompt.format_messages(question=state["question"], retrieval_query=state.get("retrieval_query", ""), answer=state.get("answer", "")))
        return {
            "retrieval_query": decision.retrieval_query,
            "rewrite_tries": state.get("rewrite_tries", 0) + 1,
            "docs": [],
            "relevant_docs": [],
            "context": "",
        }

    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("decide_retrieval", self.decide_retrieval)
        g.add_node("generate_direct", self.generate_direct)
        g.add_node("retrieve", self.retrieve)
        g.add_node("is_relevant", self.is_relevant)
        g.add_node("generate_from_context", self.generate_from_context)
        g.add_node("no_answer_found", self.no_answer_found)
        g.add_node("is_sup", self.is_sup)
        g.add_node("revise_answer", self.revise_answer)
        g.add_node("is_use", self.is_use)
        g.add_node("rewrite_question", self.rewrite_question)

        g.add_edge(START, "decide_retrieval")
        g.add_conditional_edges("decide_retrieval", self.route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
        g.add_edge("generate_direct", END)
        g.add_edge("retrieve", "is_relevant")
        g.add_conditional_edges("is_relevant", self.route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
        g.add_edge("no_answer_found", END)
        g.add_edge("generate_from_context", "is_sup")
        g.add_conditional_edges("is_sup", self.route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
        g.add_edge("revise_answer", "is_sup")
        g.add_conditional_edges("is_use", self.route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
        g.add_edge("rewrite_question", "retrieve")

        return g.compile()

    def query(self, question: str, stream_handler=None):
        self.stream_handler = stream_handler
        initial_state = {
            "question": question,
            "retrieval_query": question, 
            "rewrite_tries": 0,                                        
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "issup": "",
            "evidence": [],
            "retries": 0,
            "isuse": "not_useful",
            "use_reason": "",
        }
        result = self.app.invoke(initial_state, config={"recursion_limit": 80})
        return result
