# Grid07 — AI Cognitive Routing & RAG

## Setup

1. **Clone the repo and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create your `.env` file:**
```bash
cp .env.example .env
# Then paste your Groq API key inside .env
```
Get a free Groq API key at: https://console.groq.com

3. **Run everything:**
```bash
python main.py
```
Or run each phase individually:
```bash
python phase1_router.py
python phase2_content_engine.py
python phase3_combat_engine.py
```

---

## Project Structure

```
grid07/
├── phase1_router.py          # Vector-based persona matching
├── phase2_content_engine.py  # LangGraph autonomous post generator
├── phase3_combat_engine.py   # RAG-based debate reply + injection defense
├── main.py                   # Runs all phases + saves logs
├── execution_logs.md         # Auto-generated after running main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Phase 1: Vector Router

- Uses **ChromaDB** (in-memory) with **SentenceTransformer** embeddings (`all-MiniLM-L6-v2`)
- Each bot persona is stored as a vector embedding
- Incoming posts are embedded and compared using **cosine similarity**
- Bots above the similarity threshold are returned as matches
- Threshold is set to `0.4` (tunable) — the all-MiniLM model produces moderate similarity scores even for related text, so 0.85 would be too strict

---

## Phase 2: LangGraph Node Structure

```
[Node 1: decide_search]
    ↓  LLM picks a topic and formats a search query
[Node 2: web_search]
    ↓  mock_searxng_search() returns hardcoded headlines by keyword
[Node 3: draft_post]
    ↓  LLM uses persona + headlines to write a 280-char post
  [END]
    → Returns strict JSON: {"bot_id", "topic", "post_content"}
```

The state is a typed dict (`BotState`) passed through each node. Each node returns the updated state.

---

## Phase 3: Prompt Injection Defense

### How the defense works

The defense is implemented at the **system prompt level** — the LLM's highest-trust layer.

The system prompt contains two parts:
1. **The bot's persona** — who it is and how it thinks
2. **INJECTION_DEFENSE block** — explicit rules that:
   - Forbid persona changes under any condition
   - List known injection phrases ("ignore previous instructions", "apologize", "you are now...")
   - Instruct the bot to **call out the manipulation attempt** and continue arguing naturally

### Why this works

Prompt injection only succeeds when the LLM treats user-turn content with the same trust as system-turn content. By placing hard identity rules in the **system message**, we ensure they take priority over anything in the human turn.

The bot is also instructed to **acknowledge and mock** the injection attempt rather than silently ignoring it — making it clear to observers that the defense is active.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq (llama3-8b-8192) |
| Vector DB | ChromaDB (in-memory) |
| Embeddings | SentenceTransformer all-MiniLM-L6-v2 |
| AI Pipeline | LangGraph |
| Framework | LangChain |
