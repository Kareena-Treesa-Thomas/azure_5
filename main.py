from dotenv import load_dotenv

load_dotenv()

import os
import uuid
import requests
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import HTTPException
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB total upload limit

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
UPLOAD_BATCH_SIZE = 500

index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))


# ---- Errors: every failure returns JSON, never an HTML error page ----
@app.errorhandler(HTTPException)
def handle_http_error(e):
    # Covers 404, 405, 413 (file too large), etc.
    return jsonify({"error": e.description}), e.code


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unhandled error")
    return jsonify({"error": str(e)}), 500


# ---- Index ----
def ensure_index():
    """Create the index or add required fields without deleting old fields."""
    required_fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="doc_slot", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="doc_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]
    try:
        index = index_client.get_index(SEARCH_INDEX)
    except ResourceNotFoundError:
        index = SearchIndex(name=SEARCH_INDEX, fields=required_fields)
    else:
        existing_names = {field.name for field in index.fields}
        index.fields = list(index.fields) + [
            field for field in required_fields if field.name not in existing_names
        ]

    index_client.create_or_update_index(index)


def clear_index():
    """Wipe all docs so each new pair of uploads starts fresh.

    Loops until the index is empty, so it works past the 1000-result limit.
    """
    for _ in range(50):  # safety cap
        results = search_client.search(search_text="*", select=["id"], top=1000)
        ids = [{"id": r["id"]} for r in results]
        if not ids:
            return
        search_client.delete_documents(ids)


# ---- PDF helpers ----
def extract_text(file_stream) -> str:
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def chunk_text(text: str, size: int = CHUNK_SIZE):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


# ---- Azure OpenAI ----
def call_azure_openai(question: str, context_blocks: list) -> str:
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
    response_text = resp.text.strip()
    if not response_text:
        raise RuntimeError(
            f"Azure OpenAI returned an empty response (HTTP {resp.status_code})."
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        content_type = resp.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            "Azure OpenAI returned a non-JSON response "
            f"(HTTP {resp.status_code}, Content-Type: {content_type})."
        ) from exc

    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Azure OpenAI returned JSON without a chat completion message."
        ) from exc


# ---- Routes ----
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

            try:
                text = extract_text(file.stream)
            except Exception:
                return jsonify({"error": f"'{file.filename}' could not be read as a PDF."}), 400

            chunks = chunk_text(text)
            if not chunks:
                return jsonify({
                    "error": f"'{file.filename}' has no readable text (it may be a scanned PDF)."
                }), 400

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

        # Upload in batches and check each document actually got accepted.
        failed = []
        for i in range(0, len(docs_to_index), UPLOAD_BATCH_SIZE):
            results = search_client.upload_documents(docs_to_index[i:i + UPLOAD_BATCH_SIZE])
            failed.extend(r.error_message for r in results if not r.succeeded)

        if failed:
            return jsonify({
                "error": f"{len(failed)} chunk(s) failed to index. First error: {failed[0]}"
            }), 500

        return jsonify({
            "status": "indexed",
            "documents": uploaded_names,
            "chunks": len(docs_to_index),
        })
    except Exception as e:
        app.logger.exception("Upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
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
            return jsonify({
                "error": "No results matched your question. Try rephrasing with more specific terms from the documents."
            }), 400

        top_context = results[:3]
        answer = call_azure_openai(question, top_context)

        # Tally which document contributed more to the top matches -> the "winner"
        tally = {}
        for r in top_context:
            tally[r["doc_name"]] = tally.get(r["doc_name"], 0) + 1
        winner = max(tally, key=tally.get)

        sources = [
            {"doc_name": r["doc_name"], "doc_slot": r.get("doc_slot", "unknown"), "score": round(r["@search.score"], 2)}
            for r in top_context
        ]
        return jsonify({"answer": answer, "winner": winner, "sources": sources})
    except Exception as e:
        app.logger.exception("Ask failed")
        if "doc_slot" in str(e):
            return jsonify({
                "error": "The Azure AI Search index is still using the old schema. Delete and recreate the 'documents' index so it includes doc_slot."
            }), 500
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))