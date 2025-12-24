from app.rag.retriever import PolicyIndex

_policy_index = None

def get_policy_index() -> PolicyIndex:
    global _policy_index
    if _policy_index is None:
        _policy_index = PolicyIndex()
        _policy_index.build_or_load(force=False)
    return _policy_index
