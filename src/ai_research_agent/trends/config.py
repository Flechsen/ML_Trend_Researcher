DEFAULT_CONFIG: dict = {
    "hn_queries": [
        "LLM", "MCP", "AI agent", "Claude", "GPT", "RAG",
        "fine-tuning", "open model", "autoresearch", "agent skills",
    ],
    "min_hn_points": 40,
    "github_topics": ["mcp", "llm", "ai-agents", "rag", "llm-inference", "fine-tuning"],
    "github_keywords": ["mcp server", "llm agent", "agent skills"],
    "min_github_stars": 100,
    "github_days": 14,
    "subreddits": ["LocalLLaMA", "MachineLearning"],
    "max_items_per_source": 30,
}


def load_config(interests: dict) -> dict:
    """Merge the optional `trends:` block of interests.yaml over DEFAULT_CONFIG."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(interests.get("trends") or {})
    return cfg
