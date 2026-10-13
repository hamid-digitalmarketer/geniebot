"""
QA interface combining RAG system with LLM
"""

import time
from typing import Dict, Tuple
from .system import RAGSystem
from .llm import OllamaLLM
from utils.logger import setup_logger
from utils.cache import QueryCache

logger = setup_logger(__name__)


class RAGQA:
    """RAG-based Question Answering system"""
    
    def __init__(
        self,
        rag_system: RAGSystem,
        llm: OllamaLLM,
        use_cache: bool = True
    ):
        """
        Initialize RAG QA system
        
        Args:
            rag_system: RAG retrieval system
            llm: LLM for generation
            use_cache: Cache query responses
        """
        self.rag = rag_system
        self.llm = llm
        self.query_cache = QueryCache() if use_cache else None
        
        logger.info("RAG QA system initialized")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate for latency logging and prompt budgeting."""
        return max(1, int(len(text) / 4))

    @staticmethod
    def _build_context_from_chunks(chunks: list[tuple[str, str]]) -> tuple[str, Dict[str, list[str]]]:
        """Build context and source map from one retrieval call."""
        if not chunks:
            return "No relevant documents found in the knowledge base.", {}

        lines = ["Relevant context from documents:", ""]
        sources: Dict[str, list[str]] = {}
        for i, (chunk_text, source) in enumerate(chunks, 1):
            lines.append(f"[Source {i}: {source}]")
            lines.append(chunk_text)
            lines.append("")
            sources.setdefault(source, []).append(chunk_text[:100] + "...")

        return "\n".join(lines).strip(), sources
    
    def answer_question(
        self,
        question: str,
        use_cache: bool = True,
        conversation_context: str = ""
    ) -> Dict[str, any]:
        """
        Answer a question using RAG

        Args:
            question: User question
            use_cache: Use cache if available
            conversation_context: Optional formatted text of the user's recent
                prior interactions (e.g. from UserMemory.get_context()), used
                to let the LLM resolve conversational follow-ups ("what about
                the enterprise plan?"). Kept separate from the retrieved
                document chunks. The cache is keyed on (question, context)
                together, so a question asked with context is never confused
                with the same question asked without it, but a returning
                user whose context is unchanged (e.g. repeating a question
                in the same conversation state) still benefits from caching
                instead of unconditionally bypassing it.

        Returns:
            Dictionary with answer, sources, and metadata
        """
        overall_start = time.time()
        logger.info(f"Processing question: {question[:50]}...")

        # Check cache - keyed on (question, conversation_context) so cache
        # eligibility doesn't depend on whether context exists at all.
        if use_cache and self.query_cache:
            cached_answer = self.query_cache.get(question, conversation_context)
            if cached_answer:
                logger.debug("Returning cached answer")
                return {
                    "answer": cached_answer,
                    "sources": [],
                    "cached": True
                }
        
        # Check if documents are loaded
        if not self.rag.has_documents():
            logger.warning("No documents loaded")
            return {
                "answer": "No documents loaded. Please add documents first.",
                "sources": [],
                "error": True
            }
        
        # Retrieve relevant chunks once (avoid duplicate query embedding + scoring)
        retrieval_start = time.time()
        retrieved = self.rag.retrieve_chunks(question, top_k=2)
        context, source_chunks = self._build_context_from_chunks(retrieved)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieval time: {retrieval_time:.2f}s | top_k=2")
        
        # Build a strict prompt to avoid meta responses like
        # "based on the given context" in final user-facing output.
        prompt_start = time.time()
        system_prompt = (
            "You are GenieBot, a concise assistant for document-grounded answers. "
            "Use only the provided context. "
            "Return only the direct answer for the user. "
            "Do not mention prompts, context, documents, or your reasoning process. "
            "If the answer is not in the provided context, reply exactly: "
            "I could not find that in the loaded documents."
        )

        conversation_section = ""
        if conversation_context:
            conversation_section = (
                "Recent conversation with this user (for resolving follow-up "
                "questions like \"what about X?\" only - the Context below is "
                "still the sole source of facts for the answer):\n"
                f"{conversation_context}\n"
            )

        prompt = f"""{conversation_section}Context:
{context}

Question: {question}

Instructions:
    - Answer in 2 to 4 short sentences.
    - Be specific and factual.
    - Do not include prompt labels like "Question:" or "Instructions:" in the output.
    - Do not add preambles or explanations about how you answered.

Final Answer:"""
        prompt_time = time.time() - prompt_start
        prompt_chars = len(prompt) + len(system_prompt)
        prompt_tokens_est = self._estimate_tokens(prompt) + self._estimate_tokens(system_prompt)
        logger.info(f"Prompt construction time: {prompt_time:.2f}s")
        logger.info(f"Prompt length: {prompt_chars} chars | est_tokens={prompt_tokens_est}")
        
        # Generate answer using LLM
        llm_start = time.time()
        answer = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            # Reasoning models (e.g. GLM-4.6) spend part of this budget on
            # hidden reasoning before the visible answer, so this needs more
            # headroom than a plain non-reasoning model would.
            max_tokens=800,
        )
        llm_time = time.time() - llm_start
        logger.info(f"LLM response time: {llm_time:.2f}s")

        # Clean up answer - remove any preamble if LLM included it
        if isinstance(answer, str):
            if "Final Answer:" in answer:
                answer = answer.split("Final Answer:", 1)[-1].strip()

            # Remove common preambles the LLM might add
            for preamble in ["**Answer:**", "Answer:", "Final Answer:", "Based on the context:"]:
                if answer.strip().lower().startswith(preamble.lower()):
                    answer = answer[len(preamble):].strip()
            # Take only first 1000 chars to prevent context bleeding
            answer = answer[:1000]

        # Do not cache or return failed model responses as normal answers.
        if isinstance(answer, str) and answer.strip().lower().startswith("error:"):
            logger.error(f"LLM generation failed for question '{question[:50]}...': {answer}")
            return {
                "answer": (
                    "I could not generate an answer right now because the LLM backend is unavailable. "
                    "Please check the provider/model status and try again."
                ),
                "sources": {},
                "cached": False,
                "error": True,
                "details": answer,
            }
        
        # Cache the result, keyed on (question, conversation_context) - see
        # the cache lookup above for why context is part of the key.
        if use_cache and self.query_cache:
            self.query_cache.put(question, answer, conversation_context)
        
        total_time = time.time() - overall_start
        logger.info(f"Total request time: {total_time:.2f}s")
        logger.info(f"Answer length: {len(answer)} chars | est_tokens={self._estimate_tokens(answer)}")
        logger.info("Question answered successfully")
        
        return {
            "answer": answer,
            "sources": source_chunks,
            "cached": False
        }
    
    def get_answer_with_sources(self, question: str) -> Tuple[str, Dict]:
        """
        Get answer with formatted source information
        
        Args:
            question: User question
        
        Returns:
            Tuple of (answer, sources_dict)
        """
        result = self.answer_question(question)
        return result["answer"], result.get("sources", {})
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "rag_stats": self.rag.get_stats(),
            "llm_model": self.llm.model,
            "llm_available": self.llm.is_available(),
            "cache_size": self.query_cache.size() if self.query_cache else 0
        }
