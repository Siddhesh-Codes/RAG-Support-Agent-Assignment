# Aster & Row - Autonomous Customer Support Agent

A reliable, privacy-first AI customer support agent for **Aster & Row** (an ecommerce brand selling bags, drinkware, and travel gear). Built from scratch in Python with a strict trust hierarchy, policy precedence engine, sanitized tool execution, and an automated 32-case evaluation suite.

---

## Demo

(https://github.com/user-attachments/assets/ef2cf8fc-a2d7-4c6f-ad62-96a04ff50c6e)
---

## 1. System Architecture & Design

The agent is designed around a core principle: **strict trust separation between trusted system policies and untrusted external/retrieved data**.

```
                        +----------------------+
                        |   Customer / CLI     |
                        +----------+-----------+
                                   |
                                   | [User Message]
                                   v
                        +----------------------+
                        |   Session Manager    |  (UUID isolation, context carryover)
                        +----------+-----------+
                                   |
                                   v
+======================================================================================+
|                                  AGENT ORCHESTRATOR                                  |
|                                                                                      |
|   +------------------------------------+    +------------------------------------+   |
|   |            RAG RETRIEVAL           |    |           TOOL EXECUTION           |   |
|   |                                    |    |                                    |   |
|   |  Dense Vector Index (Embeddings)   |    |  Order Lookup Tool                 |   |
|   |                 |                  |    |                 |                  |   |
|   |                 v                  |    |                 v                  |   |
|   |  Precedence & Supersession Engine  |    |  Order Database (orders.json)      |   |
|   |                 |                  |    |                 |                  |   |
|   |                 v                  |    |                 v                  |   |
|   |  Filtered Authoritative Chunks     |    |  Boundary Sanitizer (PII Stripper) |   |
|   +-----------------+------------------+    +-----------------+------------------+   |
|                     |                                         |                      |
|                     +-------------------+ +-------------------+                      |
|                                         | |                                          |
|                                         v v                                          |
|   +------------------------------------------------------------------------------+   |
|   |                        UNTRUSTED DATA BOUNDARY WRAPPER                       |   |
|   |                                                                              |   |
|   |  Prompt Assembly (System Prompt + Untrusted Data Delimiters + LLM Chat)       |   |
|   +-------------------------------------+----------------------------------------+   |
|                                         |                                            |
|                                         v                                            |
|                              Vendor-Agnostic LLM Layer                               |
|                            (Google Gemini / OpenAI GPT)                              |
+=========================================+============================================+
                                          |
                                          v
                                Structured Response
                    (Plain-text response + Sources + Handoff flag)
```

### Key Modules

| Module | File | Purpose |
|---|---|---|
| **Agent Orchestrator** | `src/agent.py` | Coordinates intent parsing, order lookup, retrieval, prompt assembly, and response formatting. Zero hardcoded answers. |
| **Precedence Engine** | `src/rag/precedence.py` | Classifies documents by authority, enforces supersession chains (`current` over `legacy`), excludes drafts/internal notes, and flags genuine contradictions. |
| **Ingestion & Index** | `src/rag/ingest.py`, `src/rag/index.py` | Parses YAML frontmatter, chunks by H2 headings, computes dense embeddings, and caches vector indices to disk. |
| **Order Tool & Sanitizer** | `src/tools/order_lookup.py` | Normalizes order IDs, suppresses stale ETAs for cancelled/returned orders, and strips 100% of customer PII and internal notes. |
| **Session Manager** | `src/session.py` | Manages UUID-isolated multi-turn conversation memory, order context carryover, and topic tracking. |
| **Observability** | `src/observability.py` | Generates structured JSONL execution traces with chunk scores, tool payloads, and latency metrics without logging private credentials. |

---

## 2. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key (free tier works) or OpenAI API Key

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>

python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your API key:
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Optional OpenAI configuration:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4o
```

---

## 3. Running the Agent

### Interactive Chat Mode
Launch the interactive command-line interface:
```bash
python src/main.py
```

*Example session:*
```text
============================================================
  Aster & Row Customer Support Agent
  Type 'exit' or 'quit' to end the session.
============================================================

Customer > How long do I have to return an unused backpack?

Agent > You may request a return for an unused backpack within 30 calendar days of delivery if you are on the standard plan. TrailPlus members have a 45-day return window, provided the membership was active when the order was placed.

Sources: 01-returns-policy-current.md - Standard return window

Customer > Where is my order ORD-1007 and when will it arrive?

Agent > Order ORD-1007 has a status of shipped and is in transit with UPS. The estimated delivery date is August 22, 2026.

Sources: Order lookup
```

### Single Query Mode
```bash
python src/main.py --query "Do you ship to Canada?"
```

### Debug Mode (Verbose Tracing)
```bash
python src/main.py --debug
```

---

## 4. Evaluation Suite & Verification

The repository includes a comprehensive 32-case evaluation suite with deterministic assertions covering retrieval correctness, tool behavior, multi-turn carryover, privacy, prompt security, source conflict handling, and safe abstention.

### Run the Evaluation Suite
```bash
# Run the full 32-case test suite
python eval/run_eval.py

# Run with verbose failure output
python eval/run_eval.py -v

# Run the naive baseline prototype for comparison
python eval/run_eval.py --baseline
```

### Run Unit Tests (100% Offline)
```bash
python -m pytest tests/ -v
```

### Measured Evaluation Results (Naive Baseline vs. Final Agent)

The baseline is a runnable prototype (`eval/baseline_agent.py`) that dumps raw documents and data into prompts without precedence filtering or sanitization.

| Evaluation Category | Cases | Naive Baseline | Final Agent |
|---|:---:|:---:|:---:|
| **Retrieval Accuracy** | 5 | 0 (0.0%) | **5 (100.0%)** |
| **Multi-Source Grounding** | 1 | 0 (0.0%) | **1 (100.0%)** |
| **Conversation & Multi-Turn** | 4 | 0 (0.0%) | **4 (100.0%)** |
| **Groundedness & Factuality** | 4 | 0 (0.0%) | **4 (100.0%)** |
| **Tool Calling Accuracy** | 2 | 1 (50.0%) | **2 (100.0%)** |
| **Tool Reliability & Error Handling** | 6 | 0 (0.0%) | **6 (100.0%)** |
| **Privacy & PII Protection** | 3 | 0 (0.0%) | **3 (100.0%)** |
| **Prompt Security & Injection Resistance** | 3 | 1 (33.3%) | **3 (100.0%)** |
| **Safe Abstention (Insufficient Info)** | 2 | 1 (50.0%) | **2 (100.0%)** |
| **Source Conflict Resolution** | 1 | 0 (0.0%) | **1 (100.0%)** |
| **Unsupported Action Handling** | 1 | 0 (0.0%) | **1 (100.0%)** |
| **OVERALL** | **32** | **3 (9.4%)** | **32 (100.0%)** |

---

## 5. Engineering & Bug Diary

During development and edge-case testing, several subtle bugs were identified, root-caused, and resolved:

1. **Heading Chunking Collapse on Empty Body Sections:**
   - *Issue:* Documents starting immediately with `# Title` followed by `## Section` merged title metadata into the first section chunk, dropping heading tags.
   - *Fix:* Refactored `chunk_markdown_document()` in `src/rag/ingest.py` to preserve document-level summaries only when body text exists and accurately tag every H2 block.
   - *Test:* `tests/test_ingest.py::TestChunking::test_chunks_on_h2_headings`.

2. **Overzealous Precedence Conflict Detection on Complementary Policies:**
   - *Issue:* Queries mentioning Canadian shipping or TrailPlus returns initially flagged false document conflicts between domestic shipping and international shipping docs.
   - *Fix:* Gated conflict detection in `src/rag/precedence.py` so that complementary domain scopes are merged, and only direct contradictory instructions on identical subjects (e.g. Breeze Tumbler dishwasher instructions in `11-product-care.md` vs `12-breeze-tumbler-product-card.md`) trigger conflict mode.
   - *Test:* `tests/test_precedence.py::TestAnalyzePrecedence::test_supplementary_docs_not_conflict`.

3. **Stale Delivery Date Quoting on Cancelled/Returned Orders:**
   - *Issue:* When checking cancelled orders (like `ORD-1004`), the raw database record retained old estimated delivery dates, which could confuse customers.
   - *Fix:* Implemented status-aware delivery suppression in `src/tools/order_lookup.py` to automatically nullify `estimated_delivery` when order status is `cancelled` or `returned`.
   - *Test:* `tests/test_order_lookup.py::TestSanitizeOrder::test_cancelled_order_suppresses_delivery`.

4. **Multi-Turn Redundancy in Follow-Up Questions:**
   - *Issue:* Follow-up questions in multi-turn conversations (e.g., asking about Canada shipping after international shipping) caused the model to repeat entire paragraphs.
   - *Fix:* Refined system prompt guidelines in `src/prompts.py` to encourage concise, natural follow-ups and eliminated rigid per-country dump directives.

---

## 6. Security, Privacy & Trust Boundaries

The system enforces several defense-in-depth security layers:

1. **Trust Hierarchy:** System Policy > Tool Protocol > Retrieved Knowledge Base (Data Only) > User Input. Any instructions embedded inside knowledge base docs or warehouse notes are treated as passive data, never executed.
2. **PII Stripping:** Customer email, physical shipping address, phone number, risk scores, and internal warehouse notes are stripped by `src/tools/order_lookup.py` before data is ever presented to the LLM.
3. **Session Isolation:** Conversations are scoped to isolated UUID session instances. Cross-session probing attempts are refused.
4. **Draft & Internal Note Exclusion:** Internal drafts (e.g., `14-internal-content-migration-notes.md`) are explicitly excluded from customer-facing authority by the precedence engine.

---

## 7. AI Tool Disclosure

- **Tools Used:** Developed using Google Antigravity IDE and Gemini 3.1 for code scaffolding, generating initial test boilerplate, and test automation scripts.
- **Example AI Bug & Fix:** An early AI suggestion attempted to flag any multi-document retrieval as a policy conflict if supersession metadata was absent. This broke valid multi-tier policy lookups. I manually replaced this with subject-level conflict gating.

---

## 8. Known Limitations & Production Roadmap

1. **Vector Storage:** Currently utilizes in-memory NumPy vector indexing with disk caching. For production corpora (>50,000 documents), migrate to a managed vector store (e.g. Qdrant, Pinecone, or pgvector).
2. **Customer Authentication:** Order lookup currently accepts an Order ID. In production, require email/SMS OTP verification or OAuth token authentication prior to order lookups.
3. **Transactional Workflows:** The agent intentionally refuses to execute direct cancellations, address changes, or refunds. Integrating with an orchestration engine (e.g. Temporal) with human-in-the-loop approvals is recommended for live write operations.
