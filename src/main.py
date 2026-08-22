import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.rag.ingest import ingest_knowledge_base
from src.rag.index import VectorIndex
from src.rag.retriever import Retriever
from src.tools.order_lookup import OrderLookupTool
from src.agent import AgentOrchestrator
from src.providers.factory import get_llm_provider, get_embedding_provider
from src.session import SessionManager


def initialize_agent(debug: bool = False) -> AgentOrchestrator:
    """Initialize all dependencies and return configured AgentOrchestrator."""
    if debug:
        config.DEBUG_TRACE = True

    # 1. Ingest Knowledge Base
    chunks = ingest_knowledge_base(config.KNOWLEDGE_BASE_DIR)

    # 2. Build Embedding Vector Index
    emb_provider = get_embedding_provider()
    index = VectorIndex(cache_dir=config.INDEX_CACHE_DIR)
    index.build(chunks, emb_provider)

    # 3. Retriever
    retriever = Retriever(index=index, embedding_provider=emb_provider, top_k=config.RETRIEVAL_TOP_K)

    # 4. Order Tool
    order_tool = OrderLookupTool(config.ORDERS_JSON_PATH)

    # 5. LLM Provider
    llm_provider = get_llm_provider()

    # 6. Session Manager & Orchestrator
    session_manager = SessionManager()
    log_dir = Path("logs")

    orchestrator = AgentOrchestrator(
        llm_provider=llm_provider,
        retriever=retriever,
        order_tool=order_tool,
        session_manager=session_manager,
        log_dir=log_dir,
    )
    return orchestrator


def run_interactive():
    """Run an interactive multi-turn support chat session in the terminal."""
    print("=" * 60)
    print("  Aster & Row Customer Support Agent")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 60)

    agent = initialize_agent()
    session = agent.session_manager.create_session()

    while True:
        try:
            user_input = input("\nCustomer > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nThank you for contacting Aster & Row support. Goodbye!")
                break

            response = agent.process_message(user_input, session_id=session.session_id)
            print("\nAgent > " + response.format_for_user())

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\n[Error]: {str(e)}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Aster & Row AI Support Agent")
    parser.add_argument("query", nargs="?", default=None, help="Single query mode (positional)")
    parser.add_argument("--query", "-q", dest="query_flag", default=None, help="Single query mode (flag)")
    parser.add_argument("--debug", action="store_true", help="Enable debug trace logging to stderr")
    args = parser.parse_args()

    query = args.query_flag or args.query
    if query:
        agent = initialize_agent(debug=args.debug)
        response = agent.process_message(query)
        print(response.format_for_user())
    else:
        run_interactive()


if __name__ == "__main__":
    main()
