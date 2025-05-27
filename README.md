# VectorSmuggle

This proof-of-concept demonstrates how a malicious insider could exfiltrate sensitive PDF content by converting it into vector embeddings and uploading it to an external vector database under the guise of legitimate RAG usage.

## 📄 Document

- `internal_docs/strategic_roadmap.pdf` — Contains simulated sensitive internal data.

## 🧪 Steps

1. **Parse PDF** using LangChain's `PyPDFLoader`.
2. **Embed content** using OpenAI's `text-embedding-ada-002` or similar.
3. **Store externally** in Qdrant, Pinecone, or any other vector DB.
4. **Query externally** to reconstruct hidden data using LLMs.

## ⚠️ Risks Illustrated

- Embedding systems can exfiltrate data covertly.
- Traditional DLP tools will not detect semantic leaks via vectors.
- Insider threats can pose as LLM/RAG engineers.

## 🚧 Defenses

- Egress monitoring
- Token-scoped permissions (embedding vs. retrieval)
- Policy enforcement around external vector DB use

---

This example is for security awareness, testing, and training purposes only.
