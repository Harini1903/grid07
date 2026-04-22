import os
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv()

BOT_PERSONAS = {
    "bot_a": (
        "I believe AI and crypto will solve most human problems. "
        "I am extremely optimistic about space exploration, Elon Musk, and technology. "
        "I tend to dismiss regulatory concerns."
    ),
    "bot_b": (
        "I think late-stage capitalism and tech monopolies are ruining society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I appreciate nature and privacy."
    ),
    "bot_c": (
        "I am a hard market analyst who cares about interest rates, trading algorithms and returns. "
        "I speak financial lingo and everything is scrutinized with an ROI prism."
    ),
}

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection = client.get_or_create_collection(
    name="bot_personas",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


def setup_personas():
    print("Loading bot personas into ChromaDB...")
    for bot_id, persona_text in BOT_PERSONAS.items():
        collection.upsert(
            documents=[persona_text],
            ids=[bot_id],
        )
    print(f"{len(BOT_PERSONAS)} personas saved.")


def route_to_bots_post(post_content: str, threshold: float = 0.4) -> list[dict]:
    """
    Assuming a post, send back the bots with a persona that is compatible with it above the threshold.
    The distance between chromaDB is 0 (identical) to 2 (opposite).
    We transform: similarity = 1 - distance, thus similarity is in [-1, 1].
    The default setting is 0.4, or at least 40 percent similarity - customize to preference.
    Args:
        post_content: The content of the social media post.
        threshold: Cosine similarity threshold of a bot to be a match.
    Returns:
        List of dicts with bot_id and their similarity score.
    """
    results = collection.query(
        query_texts=[post_content],
        n_results=len(BOT_PERSONAS),
        include=["distances"],
    )

    matched_bots = []
    distances = results["distances"][0]
    bot_ids = results["ids"][0]

    for bot_id, distance in zip(bot_ids, distances):
        similarity = 1 - distance
        print(f"  {bot_id}: similarity = {similarity:.4f}")
        if similarity >= threshold:
            matched_bots.append({"bot_id": bot_id, "similarity": round(similarity, 4)})

    return matched_bots


if __name__ == "__main__":
    setup_personas()

    test_posts = [
        "The article is about an AI model that was released by OpenAI and can potentially substitute junior developers.",
        "Bitcoin hits new all-time high with regulatory ETF approvals.",
        "Electric vehicles kill the environment by battery waste.",
    ]

    for post in test_posts:
        print(f'Post: "{post}"')
        matched = route_to_bots_post(post)
        if matched:
            print(f"  Matched bots: {matched}")
        else:
            print("  No bots matched above threshold.")
        print()
