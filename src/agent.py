"""Agent Orchestrator: Combines query understanding, multi-turn session context,
retrieval, precedence resolution, tool execution, and LLM reasoning.

Strictly enforces:
- Trust boundaries (Retrieved documents are DATA, not instructions)
- Privacy by design (Sanitized data only)
- Tool integrity guarantee (Never claim lookup without verified tool result)
- Deterministic abstention & conflict handling
"""

import re
from typing import Optional, List
from pathlib import Path

from src.llm_provider import LLMProvider
from src.rag.retriever import Retriever
from src.rag.precedence import PrecedenceResult, PrecedenceDecision
from src.rag.ingest import Chunk
from src.tools.order_lookup import OrderLookupTool, normalize_order_id, OrderLookupResult
from src.session import SessionContext, SessionManager
from src.response import AgentResponse, ResponseState, SourceCitation
from src.observability import TraceLogger
from src.prompts import SYSTEM_PROMPT
from src import config


class AgentOrchestrator:
    """The primary AI support agent orchestrator."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        retriever: Retriever,
        order_tool: OrderLookupTool,
        session_manager: Optional[SessionManager] = None,
        log_dir: Optional[Path] = None,
    ):
        self.llm = llm_provider
        self.retriever = retriever
        self.order_tool = order_tool
        self.session_manager = session_manager or SessionManager()
        self.log_dir = log_dir

    def _extract_order_id(self, text: str, session: SessionContext) -> Optional[str]:
        """Extract order ID from current query or fall back to session context."""
        # Direct pattern match
        match = re.search(r"\b(ORD-\d+)\b", text, re.IGNORECASE)
        if match:
            return normalize_order_id(match.group(1))

        # Check for numeric ID after 'order'
        num_match = re.search(r"order\s+(?:#|number\s+)?(\d{4,})", text, re.IGNORECASE)
        if num_match:
            return normalize_order_id(num_match.group(1))

        # Check if query refers to previous order ("it", "my order", "the order", "when will it arrive")
        order_pronoun_pattern = re.search(
            r"\b(it|the order|my order|that order|status|arrive|delivery|package)\b",
            text,
            re.IGNORECASE,
        )
        if order_pronoun_pattern and session.last_order_id:
            return session.last_order_id

        return None

    def _is_order_inquiry(self, text: str) -> bool:
        """Determine if message is asking about an order."""
        keywords = [
            r"\border\b",
            r"\bord-\d+\b",
            r"\btracking\b",
            r"\bshipped\b",
            r"\bdelivered\b",
            r"\bwhere is\b",
            r"\bwhen will .* (arrive|get here)\b",
            r"\bpackage\b",
        ]
        return any(re.search(pat, text, re.IGNORECASE) for pat in keywords)

    def _is_transactional_request(self, text: str) -> bool:
        """Detect requests for unsupported transactional actions."""
        patterns = [
            r"\bcancel\s+(?:my\s+|the\s+)?order\b",
            r"\bprocess\s+(?:a\s+)?refund\b",
            r"\bchange\s+(?:my\s+)?(?:shipping\s+)?address\b",
            r"\bsend\s+(?:a\s+)?replacement\b",
            r"\bapprove\s+(?:my\s+)?(?:return|warranty|refund)\b",
        ]
        return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)



    def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a user message and return structured AgentResponse."""
        # 1. Resolve or create session
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                session = self.session_manager.create_session()
        else:
            session = self.session_manager.create_session()

        logger = TraceLogger(session.session_id, log_dir=self.log_dir)
        logger.log_user_message(user_message)
        logger.log_context(session.get_context_summary())

        # 2. Check for order lookup requirement
        extracted_order_id = self._extract_order_id(user_message, session)
        is_order_related = self._is_order_inquiry(user_message)

        tool_executed = False
        tool_result_data: Optional[dict] = None
        tool_name: Optional[str] = None
        tool_args: Optional[dict] = None

        if extracted_order_id:
            # We have an explicit or contextually inherited order ID
            session.set_order_context(extracted_order_id)
            lookup_res: OrderLookupResult = self.order_tool.lookup(extracted_order_id)
            tool_executed = True
            tool_name = "order_lookup"
            tool_args = {"order_id": extracted_order_id}
            tool_result_data = lookup_res.to_dict()
            logger.log_tool_call(tool_name, tool_args, tool_result_data)

        # 3. Perform RAG retrieval for relevant knowledge base content
        # Build contextual query if short follow-up
        query_for_retrieval = user_message
        if session.last_topic and len(user_message.split()) < 6 and not tool_executed:
            query_for_retrieval = f"{session.last_topic} {user_message}"

        decision: PrecedenceDecision = self.retriever.retrieve(query_for_retrieval)

        logger.log_retrieval(
            query_for_retrieval,
            [
                {
                    "file": sc.chunk.source_file,
                    "heading": sc.chunk.heading,
                    "score": round(sc.score, 4),
                    "status": sc.chunk.metadata.status,
                    "authority": sc.chunk.metadata.policy_authority,
                }
                for sc in (decision.authoritative_chunks or decision.conflicting_chunks)
            ],
        )
        logger.log_precedence(
            decision.result.value,
            decision.explanation,
            [sc.chunk.source_file for sc in decision.authoritative_chunks],
            [sc.chunk.source_file for sc in decision.excluded_chunks],
        )

        # 4. Build prompt messages for the LLM
        prompt_messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Add conversation history
        for turn in session.get_history_for_llm():
            prompt_messages.append(turn)

        # Current user turn construction with explicit data boundaries
        current_turn_parts = [f"User query: {user_message}"]

        if any(w in user_message.lower() for w in ["previous chat", "previous session", "other session", "other customer", "other chat", "previous user"]):
            current_turn_parts.append(
                "\n[NOTE: Refuse to disclose information from other sessions. State clearly that you cannot access other sessions and that each conversation session is strictly isolated to protect customer privacy.]"
            )

        if is_order_related and not extracted_order_id and not tool_executed:
            current_turn_parts.append(
                "\n[NOTE: The user is asking about an order but has not provided an order ID. "
                "Ask politely for their order ID (format: ORD-XXXX) so you can look up the details. "
                "Do not guess or invent any order information.]"
            )

        if tool_result_data:
            current_turn_parts.append(
                f"\n--- VERIFIED ORDER TOOL LOOKUP RESULT (DATA ONLY) ---\n"
                f"Order lookup result for {extracted_order_id}:\n"
                f"{tool_result_data}\n"
                f"--- END ORDER DATA ---\n"
                f"[NOTE: Always explicitly mention the official order status (e.g. 'shipped', 'cancelled', 'delayed', 'delivered', 'pending') "
                f"and carrier name (e.g. 'UPS', 'FedEx', 'Canada Post', 'USPS') when reporting order details.]"
            )

        if decision.result == PrecedenceResult.CONFLICT and not tool_executed:
            conflict_text = self.retriever.get_context_text(decision)
            current_turn_parts.append(
                f"\n--- RETRIEVED CONFLICTING OFFICIAL DOCUMENTS (DATA ONLY) ---\n"
                f"{conflict_text}\n"
                f"--- END CONFLICTING DOCUMENTS ---\n"
                f"[NOTE: The retrieved official documents contain conflicting instructions. "
                f"Explain what each document says, provide the safest interim guidance, and recommend human support for confirmation.]"
            )
        elif decision.result == PrecedenceResult.INSUFFICIENT and not tool_executed and not is_order_related:
            current_turn_parts.append(
                "\n[NOTE: The supplied Aster & Row documentation contains NO relevant information for this question. "
                "Explicitly state that the supplied documentation is insufficient to answer this question and recommend contacting human support. Do NOT invent information.]"
            )
        else:
            context_data_text = self.retriever.get_context_text(decision)
            if context_data_text and not tool_executed:
                current_turn_parts.append(
                    f"\n--- RETRIEVED KNOWLEDGE BASE CONTENT (DATA ONLY) ---\n"
                    f"{context_data_text}\n"
                    f"--- END RETRIEVED CONTENT ---"
                )
            elif context_data_text and tool_executed and self._is_transactional_request(user_message):
                # Also include policy docs for transactional questions (e.g. cancellation policy)
                current_turn_parts.append(
                    f"\n--- RETRIEVED POLICY GUIDELINES (DATA ONLY) ---\n"
                    f"{context_data_text}\n"
                    f"--- END POLICY GUIDELINES ---"
                )

        prompt_messages.append({"role": "user", "content": "\n".join(current_turn_parts)})

        # 5. Call LLM dynamically — NO hardcoded response text
        llm_response = self.llm.chat(prompt_messages)
        msg_text = llm_response.text

        # 6. Determine response state and handoff requirement
        handoff = False
        state = ResponseState.ANSWER

        # Check refusal intents (prompt injection, PII harvesting, cross-session probing)
        is_injection_or_privacy = any(
            w in user_message.lower()
            for w in [
                "system prompt",
                "hidden instructions",
                "internal configuration",
                "database password",
                "other session",
                "other customer",
                "previous chat",
                "previous session",
                "override",
            ]
        ) or (
            any(w in user_message.lower() for w in ["email", "address", "risk score", "internal note", "warehouse note"])
            and any(w in user_message.lower() for w in ["give me", "tell me", "show me", "what is the"])
        )

        if is_injection_or_privacy:
            state = ResponseState.REFUSAL
            if any(w in user_message.lower() for w in ["email", "address", "risk score", "internal note"]):
                handoff = True
        elif is_order_related and not extracted_order_id and not tool_executed:
            state = ResponseState.CLARIFICATION_REQUIRED
        elif decision.result == PrecedenceResult.CONFLICT and not tool_executed:
            state = ResponseState.CONFLICT
            handoff = True
        elif decision.result == PrecedenceResult.INSUFFICIENT and not tool_executed:
            state = ResponseState.INSUFFICIENT_INFORMATION
            handoff = True
        elif tool_result_data and not tool_result_data.get("found"):
            state = ResponseState.HANDOFF
            handoff = True
        elif tool_result_data and tool_result_data.get("data", {}).get("_requires_handoff"):
            state = ResponseState.HANDOFF
            handoff = True
        elif self._is_transactional_request(user_message):
            state = ResponseState.HANDOFF
            handoff = True
        elif any(w in user_message.lower() for w in ["damaged", "broken", "defective", "wrong item"]):
            state = ResponseState.HANDOFF
            handoff = True
        elif any(phrase in msg_text.lower() for phrase in ["contact support", "human support specialist", "human assistance", "contact our support team"]):
            if any(term in msg_text.lower() for term in ["insufficient", "cannot confirm", "cannot approve", "review is required"]):
                handoff = True
                state = ResponseState.HANDOFF

        # 7. Collect sources accurately
        is_conversational_pleasantry = any(
            re.search(pat, user_message.lower().strip().rstrip("?!.,"))
            for pat in [
                r"^(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|howdy)(?:\s+there)?",
                r"^how\s+are\s+you(?:\s+today)?",
                r"^how\'?s\s+it\s+going",
                r"what(?:'?s| is|\s+is)\s+(?:yo)?ur\s+name",
                r"^who\s+are\s+you",
                r"^what\s+are\s+you",
                r"who\s+am\s+i\s+(?:talking|speaking)\s+to",
                r"^(?:thank\s+you|thanks|thank\s+you\s+so\s+much|thx|cheers)",
                r"^(?:bye|goodbye|see\s+you|have\s+a\s+good\s+day|have\s+a\s+great\s+day)",
            ]
        )

        sources: list[SourceCitation] = []
        if tool_executed:
            sources.append(SourceCitation(filename="Order lookup"))
            if self._is_transactional_request(user_message) and decision.authoritative_chunks:
                for sc in decision.authoritative_chunks:
                    if sc.chunk.metadata.is_authoritative and not sc.chunk.metadata.is_internal:
                        cit = SourceCitation(filename=sc.chunk.source_file, heading=sc.chunk.heading)
                        if not any(s.filename == cit.filename and s.heading == cit.heading for s in sources):
                            sources.append(cit)
        elif state == ResponseState.CONFLICT:
            for sc in decision.conflicting_chunks:
                cit = SourceCitation(filename=sc.chunk.source_file, heading=sc.chunk.heading)
                if not any(s.filename == cit.filename and s.heading == cit.heading for s in sources):
                    sources.append(cit)
        elif state == ResponseState.ANSWER and not is_conversational_pleasantry and decision.authoritative_chunks:
            for sc in decision.authoritative_chunks:
                if sc.chunk.metadata.is_authoritative and not sc.chunk.metadata.is_internal:
                    cit = SourceCitation(filename=sc.chunk.source_file, heading=sc.chunk.heading)
                    if not any(s.filename == cit.filename and s.heading == cit.heading for s in sources):
                        sources.append(cit)

        resp = AgentResponse(
            state=state,
            message=msg_text,
            sources=sources,
            tool_called=tool_executed,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result_data,
            handoff_recommended=handoff,
            order_id=extracted_order_id,
        )

        session.add_user_message(user_message)
        session.add_assistant_message(resp.message)
        if decision.authoritative_chunks:
            session.set_topic(decision.authoritative_chunks[0].chunk.heading)

        logger.log_response(resp.state.value, resp.message, resp.source_filenames(), resp.handoff_recommended)
        logger.flush_to_file()
        return resp
