import os
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

FAQ_PATH = os.path.join("data", "faq.json")
FEES_PATH = os.path.join("data", "fees.json")  # change if your file name is different


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    faq_data = load_json(FAQ_PATH)
    fees_data = load_json(FEES_PATH)

    docs = []

    # ---------- 1) FAQ items ----------
    for item in faq_data:
        # If it's a dictionary like {"question": "...", "answer": "..."}
        if isinstance(item, dict):
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
        else:
            # If it's a plain string or something else
            q = str(item).strip()
            a = str(item).strip()

        if not q:
            continue

        docs.append(
            Document(
                page_content=q,
                metadata={
                    "answer": a or q,
                    "source": "faq",
                },
            )
        )

    # ---------- 2) Fees items ----------
    for item in fees_data:
        if isinstance(item, dict):
            # Try several possible keys – adjust if your JSON uses different ones
            q = (
                item.get("question")
                or item.get("title")
                or item.get("name")
                or item.get("course")
                or ""
            )
            a = item.get("answer") or item.get("details") or ""
        else:
            q = str(item)
            a = str(item)

        q = q.strip()
        a = a.strip()

        if not q:
            continue

        docs.append(
            Document(
                page_content=q,
                metadata={
                    "answer": a or q,
                    "source": "fees",
                },
            )
        )

    if not docs:
        print("No documents found – check your JSON files.")
        return

    print(f"Loaded {len(docs)} documents into the vector store.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(docs, embeddings)

    vectordb.save_local("vector_store")
    print("Vector store saved to ./vector_store")


if __name__ == "__main__":
    main()
