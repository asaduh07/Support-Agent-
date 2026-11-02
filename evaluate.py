"""Evaluation: Task 5 chunking precision/recall."""
import rag

# Same 5 in-scope queries as Task 4, with hand-labelled relevant parent documents.
RAG_QUERIES = [
    ("How quickly must a P1 ticket be resolved?", {"02_sla_by_severity"}),
    ("When will my refund reach my bank account?", {"04_refund_compensation"}),
    ("Who approves a refund above 2,000 rupees?", {"03_escalation_matrix", "04_refund_compensation"}),
    ("What happens if I complain about the same issue again within two weeks?", {"07_repeat_complaints"}),
    ("How long are support tickets kept before they are deleted?", {"12_data_retention"}),
]
OUT_OF_SCOPE = ["What is the capital of France?", "Can you give me a good recipe for chicken biryani?",
                "Who won the cricket world cup in 2011?", "What is the weather in Mumbai today?",
                "How do I reset my Netflix password?", "Tell me a joke"]


def precision_recall(strategy, k=rag.TOP_K):
    """Doc-level P/R: map top-k chunks to parent docs, dedup, then score. Prints per-query arithmetic."""
    print(f"\n=== Collection: {rag.COLLECTIONS[strategy]} ({strategy} chunking, top-{k}) ===")
    ps, rs = [], []
    for q, relevant in RAG_QUERIES:
        retrieved = list(dict.fromkeys(h["doc_id"] for h in rag.retrieve(q, strategy, k)))  # dedup, keep rank order
        hit = relevant & set(retrieved)
        p, r = len(hit) / len(retrieved), len(hit) / len(relevant)
        ps.append(p); rs.append(r)
        print(f"Q: {q}\n   retrieved docs={retrieved}\n   relevant={sorted(relevant)}"
              f"\n   P = {len(hit)}/{len(retrieved)} = {p:.2f}   R = {len(hit)}/{len(relevant)} = {r:.2f}")
    mp, mr = sum(ps) / len(ps), sum(rs) / len(rs)
    print(f"Mean P = ({' + '.join(f'{p:.2f}' for p in ps)}) / {len(ps)} = {mp:.2f}")
    print(f"Mean R = ({' + '.join(f'{r:.2f}' for r in rs)}) / {len(rs)} = {mr:.2f}")
    return mp, mr
