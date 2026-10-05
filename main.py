from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import requests
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)

app = Flask(__name__)

# ---- Config (read from environment / .env via App Service settings) ----
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "documents")

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

CHUNK_SIZE = 1000
TOP_K = 6

index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))


def ensure_index():
    """Create the search index once, if it doesn't already exist."""
    existing = [i.name for i in index_client.list_indexes()]
    if SEARCH_INDEX in existing:
        return
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="doc_slot", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="doc_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]
    index_client.create_index(SearchIndex(name=SEARCH_INDEX, fields=fields))


def extract_text(file_stream) -> str:
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text: str, size: int = CHUNK_SIZE):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def clear_index():
    """Wipe existing docs so each new pair of uploads starts fresh (demo-friendly)."""
    results = search_client.search(search_text="*", select=["id"], top=1000)
    ids = [{"id": r["id"]} for r in results]
    if ids:
        search_client.delete_documents(ids)


def call_azure_openai(question: str, context_blocks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['doc_name']}]\n{c['content']}" for c in context_blocks
    )
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer isn't in the context, say you "
        "couldn't find it in either document. Mention which document the answer "
        "came from.\n\nContext:\n" + context
    )
    url = (
        f"{OPENAI_ENDPOINT}/openai/deployments/{OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={OPENAI_API_VERSION}"
    )
    resp = requests.post(
        url,
        headers={"api-key": OPENAI_KEY, "Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        ensure_index()
        clear_index()

        docs_to_index = []
        uploaded_names = {}
        for slot in ("doc1", "doc2"):
            file = request.files.get(slot)
            if not file or file.filename == "":
                continue
            text = extract_text(file.stream)
            chunks = chunk_text(text)
            uploaded_names[slot] = file.filename
            for idx, chunk in enumerate(chunks):
                docs_to_index.append({
                    "id": str(uuid.uuid4()),
                    "content": chunk,
                    "doc_slot": slot,
                    "doc_name": file.filename,
                    "chunk_index": idx,
                })

        if not docs_to_index:
            return jsonify({"error": "Upload at least one PDF."}), 400

        search_client.upload_documents(docs_to_index)
        return jsonify({"status": "indexed", "documents": uploaded_names, "chunks": len(docs_to_index)})
    except Exception as e:
        app.logger.exception("Upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question is required."}), 400

        results = list(search_client.search(
            search_text=question,
            top=TOP_K,
            include_total_count=True,
            search_mode="any",
            query_type="simple",
        ))
        if not results:
            return jsonify({"error": "No results matched your question. Try rephrasing with more specific terms from the documents."}), 400

        top_context = results[:3]
        answer = call_azure_openai(question, top_context)

        # Tally which document contributed more to the top matches -> the "winner"
        tally = {}
        for r in top_context:
            tally[r["doc_name"]] = tally.get(r["doc_name"], 0) + 1
        winner = max(tally, key=tally.get)

        sources = [
            {"doc_name": r["doc_name"], "doc_slot": r["doc_slot"], "score": round(r["@search.score"], 2)}
            for r in top_context
        ]

        return jsonify({"answer": answer, "winner": winner, "sources": sources})
    except Exception as e:
        app.logger.exception("Ask failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))