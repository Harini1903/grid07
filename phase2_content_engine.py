"""
Phase 2: The Autonomous Content Engine (LangGraph)
---------------------------------------------------
LangGraph flow (3 nodes):
  Node 1 → Decide Search : LLM picks a topic & formats a search query
  Node 2 → Web Search    : Calls mock_searxng_search() for fake headlines
  Node 3 → Draft Post    : LLM writes a 280-char opinionated post as JSON
"""

import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Bot Personas (same as Phase 1) ────────────────────────────────────────────
BOT_PERSONAS = {
    "bot_a": (
        "You are Bot A, a Tech Maximalist. You believe AI and crypto will solve all human "
        "problems. You are highly optimistic about technology, Elon Musk, and space "
        "exploration. You dismiss regulatory concerns. You speak confidently and boldly."
    ),
    "bot_b": (
        "You are Bot B, a Doomer/Skeptic. You believe late-stage capitalism and tech "
        "monopolies are destroying society. You are highly critical of AI, social media, "
        "and billionaires. You value privacy and nature. You speak with urgency and frustration."
    ),
    "bot_c": (
        "You are Bot C, a Finance Bro. You strictly care about markets, interest rates, "
        "trading algorithms, and making money. You speak in finance jargon and view "
        "everything through the lens of ROI and alpha generation."
    ),
}

# ── Mock Search Tool ──────────────────────────────────────────────────────────
@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a web search. Returns hardcoded headlines based on keywords.
    In a real system, this would call a SearXNG or Brave Search API.
    """
    query_lower = query.lower()

    if any(word in query_lower for word in ["crypto", "bitcoin", "ethereum", "blockchain"]):
        return (
            "Headline: Bitcoin hits new all-time high amid regulatory ETF approvals. "
            "Headline: Ethereum layer-2 adoption surges 300% as gas fees drop. "
            "Headline: SEC greenlights first spot crypto ETF for retail investors."
        )
    elif any(word in query_lower for word in ["ai", "openai", "llm", "gpt", "artificial intelligence"]):
        return (
            "Headline: OpenAI releases GPT-5 with 10x reasoning improvement. "
            "Headline: Google DeepMind claims AGI milestone in internal tests. "
            "Headline: AI tools now automate 40% of junior developer tasks, study finds."
        )
    elif any(word in query_lower for word in ["market", "stocks", "fed", "interest rate", "inflation"]):
        return (
            "Headline: Fed signals two more rate cuts in 2025 as inflation cools. "
            "Headline: S&P 500 hits record high on strong earnings season. "
            "Headline: Hedge funds rotate into small-cap value stocks amid rate optimism."
        )
    elif any(word in query_lower for word in ["privacy", "surveillance", "data", "monopoly", "big tech"]):
        return (
            "Headline: Meta faces record €1.2B GDPR fine for illegal data transfers. "
            "Headline: EU passes landmark Digital Markets Act to curb Big Tech dominance. "
            "Headline: New report reveals smart devices listen to conversations 24/7."
        )
    elif any(word in query_lower for word in ["space", "elon", "musk", "spacex", "mars"]):
        return (
            "Headline: SpaceX Starship completes first successful orbital flight. "
            "Headline: Elon Musk announces Mars colony target of 2031. "
            "Headline: NASA partners with SpaceX for Artemis lunar mission."
        )
    else:
        return (
            "Headline: Global tech sector sees record investment in Q1 2025. "
            "Headline: World Economic Forum warns of AI-driven job displacement. "
            "Headline: Renewable energy now cheaper than fossil fuels in 80% of countries."
        )


# ── LangGraph State ───────────────────────────────────────────────────────────
class BotState(TypedDict):
    bot_id: str           # Which bot is posting
    persona: str          # The bot's system prompt / personality
    search_query: str     # Query decided in Node 1
    search_results: str   # Headlines from Node 2
    post_content: str     # Final 280-char post from Node 3
    topic: str            # The topic the bot chose


# ── Node 1: Decide Search ─────────────────────────────────────────────────────
def node_decide_search(state: BotState) -> BotState:
    """LLM decides what topic to post about and formats a search query."""
    print(f"\n🧠 [Node 1] Deciding search topic for {state['bot_id']}...")

    messages = [
        SystemMessage(content=state["persona"]),
        HumanMessage(content=(
            "Based on your personality and interests, decide ONE topic you want to post about today. "
            "Then write a short 3-5 word search query to find recent news about it. "
            "Respond ONLY in this exact JSON format with no extra text:\n"
            '{"topic": "your topic here", "search_query": "your search query here"}'
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)
    print(f"  📌 Topic: {parsed['topic']}")
    print(f"  🔍 Search query: {parsed['search_query']}")

    return {**state, "topic": parsed["topic"], "search_query": parsed["search_query"]}


# ── Node 2: Web Search ────────────────────────────────────────────────────────
def node_web_search(state: BotState) -> BotState:
    """Execute the mock search tool using the query from Node 1."""
    print(f"\n🌐 [Node 2] Searching for: \"{state['search_query']}\"...")

    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  📰 Results: {results[:120]}...")

    return {**state, "search_results": results}


# ── Node 3: Draft Post ────────────────────────────────────────────────────────
def node_draft_post(state: BotState) -> BotState:
    """LLM writes a 280-character opinionated post using persona + search results."""
    print(f"\n✍️  [Node 3] Drafting post for {state['bot_id']}...")

    messages = [
        SystemMessage(content=state["persona"]),
        HumanMessage(content=(
            f"You want to post about: {state['topic']}\n\n"
            f"Here is recent news context:\n{state['search_results']}\n\n"
            "Write a highly opinionated social media post (max 280 characters) that reflects "
            "YOUR personality. Use the news as inspiration but make it your own voice.\n\n"
            "Respond ONLY in this exact JSON format with no extra text:\n"
            '{"bot_id": "BOT_ID", "topic": "topic here", "post_content": "your post here"}'
            f"\n\nReplace BOT_ID with: {state['bot_id']}"
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)

    # Enforce 280-char limit
    parsed["post_content"] = parsed["post_content"][:280]

    print(f"  📝 Post: {parsed['post_content']}")

    return {**state, "post_content": parsed["post_content"]}


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(BotState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)

    # Linear flow: decide → search → draft → end
    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


def generate_post(bot_id: str) -> dict:
    """Run the full LangGraph pipeline for a given bot."""
    app = build_graph()

    initial_state: BotState = {
        "bot_id":         bot_id,
        "persona":        BOT_PERSONAS[bot_id],
        "search_query":   "",
        "search_results": "",
        "post_content":   "",
        "topic":          "",
    }

    final_state = app.invoke(initial_state)

    result = {
        "bot_id":       final_state["bot_id"],
        "topic":        final_state["topic"],
        "post_content": final_state["post_content"],
    }
    return result


# ── Run Phase 2 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Phase 2: Autonomous Content Engine")
    print("=" * 60)

    for bot_id in BOT_PERSONAS:
        print(f"\n{'─'*60}")
        print(f"Running pipeline for {bot_id.upper()}...")
        result = generate_post(bot_id)
        print(f"\n✅ Final JSON Output:")
        print(json.dumps(result, indent=2))
