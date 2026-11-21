"""Equal context budgeting for retrieval depth."""


def compute_effective_k(
    avg_chunk_tokens: float,
    budget_tokens: int = 2000,
) -> int:
    if avg_chunk_tokens <= 0:
        return 1
    return max(1, int(budget_tokens / avg_chunk_tokens))
