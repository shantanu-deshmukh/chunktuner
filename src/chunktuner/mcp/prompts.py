"""MCP prompt templates."""


def register_prompts(mcp: object) -> None:
    from mcp.server.fastmcp import FastMCP

    if not isinstance(mcp, FastMCP):
        raise TypeError("Expected FastMCP instance")

    @mcp.prompt()
    def explain_chunking_results(
        best_strategy: str = "",
        metrics_json: str = "",
    ) -> str:
        """Help an assistant interpret evaluation / recommendation output."""
        return (
            "You are helping a user interpret chunktuner evaluation results.\n\n"
            f"Best strategy summary:\n{best_strategy}\n\n"
            f"Key metrics (JSON or text):\n{metrics_json}\n\n"
            "Explain in plain language: (1) what the winning configuration means for chunk size, "
            "(2) whether retrieval overlap looks healthy, (3) any caveats about the trivial "
            "evaluation dataset if one was used."
        )

    @mcp.prompt()
    def design_eval_questions(
        corpus_excerpt: str = "",
        domain: str = "general",
    ) -> str:
        """Guide creation of domain-specific evaluation questions."""
        return (
            f"The corpus domain is: {domain}.\n\n"
            "Here is an excerpt from the corpus:\n"
            f"{corpus_excerpt[:4000]}\n\n"
            "Propose 5 evaluation questions for RAG-style retrieval. For each question, "
            "describe which document sections should contain the answer and what span "
            "characteristics to validate (offsets must slice the original text exactly)."
        )
