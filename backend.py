from dotenv import load_dotenv
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

FAISS_PATH = "faiss_index"

def clean_text(text):
    return " ".join(text.split())

def clean_query(query):
    return query.replace("?", "").replace(".", "").strip()

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

def load_data():
    all_docs = []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            path = os.path.join("data", file)
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = file

            all_docs.extend(docs)

    return get_splitter().split_documents(all_docs)

EMBEDDINGS = None

def get_embeddings():
    global EMBEDDINGS
    if EMBEDDINGS is None:
        EMBEDDINGS = HuggingFaceEmbeddings()
    return EMBEDDINGS

VECTORSTORE = None

def get_vectorstore():
    global VECTORSTORE

    if VECTORSTORE:
        return VECTORSTORE

    embeddings = get_embeddings()

    if os.path.exists(FAISS_PATH):
        VECTORSTORE = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return VECTORSTORE

    chunks = load_data()
    VECTORSTORE = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE.save_local(FAISS_PATH)

    return VECTORSTORE

def load_docs_from_upload(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = file.name

    os.remove(path)

    return docs

def build_vectorstore_from_docs(docs):
    splitter = get_splitter()
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")


# FOLLOW-UP HANDLING
def rewrite_with_history(query, history):

    query_clean = query.lower().strip()
    user_msgs = [m["content"] for m in history if m["role"] == "user"]

    if len(user_msgs) < 2:
        return query

    previous = user_msgs[-2].strip()

    if query_clean.startswith("and"):
        new_subject = query_clean.replace("and", "", 1).strip()

        prev_words = previous.split()

        if len(prev_words) >= 4:
            prev_words[-1] = new_subject
            return " ".join(prev_words)

        return new_subject

    return query


def get_context(vectorstore, query, history=None, last_query=None):

    query = clean_query(query)

    if history and len(history) > 1:
        query = rewrite_with_history(query, history)

    elif last_query:
        query = rewrite_with_history(query, [{"role": "user", "content": last_query}])

    print("\n--- FINAL QUERY ---")
    print(query)

    results = vectorstore.similarity_search(query, k=12)

    docs = []
    for doc in results:
        docs.append({
            "content": doc.page_content,
            "page": doc.metadata.get("page", "?"),
            "source": doc.metadata.get("source", "?")
        })

    return docs, query


def generate_answer(llm, context, query):

    ctx = ""
    for c in context:
        ctx += f"(File: {c['source']}, Page {c['page']}): {c['content']}\n\n"

    prompt = f"""
You are a strict AI assistant.

Answer ONLY using the provided context.

If not found say:
"This information is not available in the provided documents."

Context:
{ctx}

Question:
{query}
"""

    return llm.invoke(prompt).content
