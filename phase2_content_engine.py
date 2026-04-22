import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

BOT_PERSONAS = {
    "bot_a": (
        "You are Bot A, a Tech Maximalist. You believe AI and crypto will solve human "
        "problems. You are extremely optimistic about technology, Elon Musk, and space "
        "exploration. You dismiss regulatory concerns. You are outspoken."
    ),
    "bot_b": (
        "You are Bot B, a Doomer/Skeptic. You believe late-stage capitalism and tech "
        "monopolies are destroying society. You are highly critical of AI, social media, "
        "and billionaires. You love privacy and nature. You speak with urgency and frustration."
    ),
    "bot_c": (
        "You are Bot C, a Finance Bro. You are strictly focused on markets, interest rates, "
        "trading algorithms, and making money. You speak in finance jargon and view "
        "everything through the lens of ROI and alpha generation."
    ),
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a web search by returning hardcoded headlines based on keywords.
    In a real system this would call a SearXNG or Brave Search API.
    """
    query_lower = query.lower()

    if any(word in query_lower for word in ["crypto", "bitcoin", "ethereum", "blockchain"]):
        return (
            "Headline: Bitcoin surges to new all-time high following regulatory ETF approvals. "
            "Headline: Ethereum layer-2 adoption up 300% as gas fees decline. "
            "Headline: SEC greenlights first spot crypto ETF for retail investors."
        )
    elif any(word in query_lower for word in ["ai", "openai", "llm", "gpt", "artificial intelligence"]):
        return (
            "Headline: OpenAI unveils GPT-5 with 10x reasoning improvements. "
            "Headline: Google DeepMind announces AGI milestone in internal tests. "
            "Headline: Study finds 40% of junior developer tasks now automated by AI."
        )
    elif any(word in query_lower for word in ["market", "stocks", "fed", "interest rate", "inflation"]):
        return (
            "Headline: Fed signals two more rate cuts by 2025 as inflation cools. "
            "Headline: S&P 500 hits record high on strong earnings season. "
            "Headline: Hedge funds rotate into small-cap value stocks on rate optimism."
        )
    elif any(word in query_lower for word in ["privacy", "surveillance", "data", "monopoly", "big tech"]):
        return (
            "Headline: Meta fined record 1.2B in GDPR data transfer violation. "
            "Headline: EU passes landmark Digital Markets Act to curb Big Tech dominance. "
            "Headline: New report reveals smart devices recording conversations around the clock."
        )
    elif any(word in query_lower for word in ["space", "elon", "musk", "spacex", "mars"]):
        return (
            "Headline: SpaceX Starship completes first successful orbital flight. "
            "Headline: Elon Musk targets 2031 for first Mars colony. "
            "Headline: NASA and SpaceX deepen collaboration on Artemis lunar mission."
        )
    else:
        return (
            "Headline: Global tech sector sees record investment in Q1 2025. "
            "Headline: World Economic Forum warns of accelerating AI-driven job displacement. "
            "Headline: Renewable energy now cheaper than fossil fuels in 80% of countries."
        )


class BotState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    post_content: str
    topic: str


def node_decide_search(state: BotState) -> BotState:
    print(f"[Node 1] Select search topic of {state['bot_id']}...")

    messages = [
        SystemMessage(content=state["persona"]),
        HumanMessage(content=(
            "Based on your personality and interests, decide ONE topic you want to post about today. "
            "Write a short 3-5 word search query to find recent news about it. "
            "Please answer in the following JSON format and no more text: "
            '{"topic": "your topic here", "search_query": "your search query here"}'
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)
    print(f"  Topic: {parsed['topic']}")
    print(f"  Search query: {parsed['search_query']}")

    return {**state, "topic": parsed["topic"], "search_query": parsed["search_query"]}


def node_web_search(state: BotState) -> BotState:
    print(f"\n[Node 2] Searching for: \"{state['search_query']}\"...")

    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  Results: {results[:120]}...")

    return {**state, "search_results": results}


def node_draft_post(state: BotState) -> BotState:
    print(f"[Node 3] Writing post on behalf of {state['bot_id']}...")

    messages = [
        SystemMessage(content=state["persona"]),
        HumanMessage(content=(
            f"You would like to write about: {state['topic']}\n"
            f"The most up-to-date news: \n{state['search_results']}\n\n"
            "Write a very opinionated social media post (max 280 characters) that reflects "
            "YOUR personality. Use the news as a guide but make it your voice. "
            "Please answer in the following format, in JSON, and without any extra text: "
            '{"bot_id": "BOT_ID", "topic": "topic here", "post_content": "your post here"}'
            f"\n\nReplace BOT_ID with: {state['bot_id']}"
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = {"bot_id": state["bot_id"], "topic": state["topic"], "post_content": raw[:280]}

    parsed["post_content"] = parsed["post_content"][:280]
    print(f"  Post: {parsed['post_content']}")

    return {**state, "post_content": parsed["post_content"]}


def build_graph():
    graph = StateGraph(BotState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search", node_web_search)
    graph.add_node("draft_post", node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


def create_post(bot_id: str) -> dict:
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

    return {
        "bot_id":       final_state["bot_id"],
        "topic":        final_state["topic"],
        "post_content": final_state["post_content"],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Autonomous Content Engine")
    print("=" * 60)

    for bot_id in BOT_PERSONAS:
        print(f"\n{'─' * 60}")
        print(f"Initiating pipeline of bot {bot_id.upper()}...")
        result = create_post(bot_id)
        print("Final JSON Response:")
        print(json.dumps(result, indent=2))
