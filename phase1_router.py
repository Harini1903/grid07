"""
Phase 1: Vector-Based Persona Matching (The Router)
----------------------------------------------------
This module:
1. Creates an in-memory ChromaDB vector store
2. Stores the 3 bot personas as embeddings
3. Routes incoming posts to the right bots based on cosine similarity
"""

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ── Bot Personas ──────────────────────────────────────────────────────────────
BOT_PERSONAS = {
    "bot_a": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns."
    ),
    "bot_b": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}

# ── Setup ChromaDB with a sentence-transformer embedding function ─────────────
# Using the default all-MiniLM-L6-v2 model (downloads automatically, no API key needed)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create an in-memory ChromaDB client
client = chromadb.Client()

# Create (or get) a collection to store bot personas
collection = client.get_or_create_collection(
    name="bot_personas",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # Use cosine similarity
)

def setup_personas():
    """Embed each bot persona and store it in ChromaDB."""
    print("📦 Loading bot personas into ChromaDB...")
    for bot_id, persona_text in BOT_PERSONAS.items():
        # Upsert so re-running the script doesn't duplicate entries
        collection.upsert(
            documents=[persona_text],
            ids=[bot_id],
        )
    print(f"✅ {len(BOT_PERSONAS)} personas stored.\n")


def route_post_to_bots(post_content: str, threshold: float = 0.4) -> list[dict]:
    """
    Given a post, return the bots whose persona matches it above the threshold.

    ChromaDB cosine distance ranges from 0 (identical) to 2 (opposite).
    We convert: similarity = 1 - distance, so similarity is in [-1, 1].
    A threshold of 0.4 means "at least 40% similar" — tune this as needed.

    Args:
        post_content: The text of the incoming social media post.
        threshold:    Minimum cosine similarity to consider a bot "interested".

    Returns:
        List of dicts with bot_id and their similarity score.
    """
    results = collection.query(
        query_texts=[post_content],
        n_results=len(BOT_PERSONAS),  # Check all bots
        include=["distances"],
    )

    matched_bots = []
    distances = results["distances"][0]    # List of distances for each bot
    bot_ids   = results["ids"][0]          # Corresponding bot IDs

    for bot_id, distance in zip(bot_ids, distances):
        # ChromaDB cosine distance → similarity
        similarity = 1 - distance
        print(f"  🤖 {bot_id}: similarity = {similarity:.4f}")
        if similarity >= threshold:
            matched_bots.append({"bot_id": bot_id, "similarity": round(similarity, 4)})

    return matched_bots


# ── Run Phase 1 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_personas()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "Electric vehicles are destroying the environment with battery waste.",
    ]

    for post in test_posts:
        print(f"📢 Post: \"{post}\"")
        matched = route_post_to_bots(post)
        if matched:
            print(f"  ✅ Matched bots: {matched}")
        else:
            print("  ❌ No bots matched above threshold.")
        print()
