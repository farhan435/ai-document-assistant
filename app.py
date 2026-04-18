import streamlit as st
from backend import *

st.set_page_config(page_title="AI Document Assistant", layout="wide")

# ---------------------------
# STYLING
# ---------------------------
st.markdown("""
<style>
.user-msg {background:#DCF8C6;padding:10px;border-radius:10px;margin-bottom:5px;}
.bot-msg {background:#F1F0F0;padding:10px;border-radius:10px;margin-bottom:10px;}
.source-box {font-size:12px;color:gray;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD DEFAULT
# ---------------------------
@st.cache_resource
def load_default():
    return get_vectorstore(), get_llm()

default_vs, llm = load_default()

# ---------------------------
# SESSION
# ---------------------------
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {"Default Documents": default_vs}

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Default Documents"

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if st.session_state.current_mode not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.current_mode] = []

messages = st.session_state.chat_histories[st.session_state.current_mode]
current = st.session_state.current_mode

# ---------------------------
# HEADER
# ---------------------------
st.markdown("# 📄 AI Document Assistant")
st.caption(f"📂 Currently using: **{current}**")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:

    selected = st.selectbox(
        "Choose document",
        list(st.session_state.vectorstores.keys())
    )

    if selected != current:
        st.session_state.current_mode = selected
        if selected not in st.session_state.chat_histories:
            st.session_state.chat_histories[selected] = []
        st.rerun()

    st.markdown("---")

    file = st.file_uploader("Upload PDF", type="pdf")

    if file:
        key = file.name
        if key not in st.session_state.vectorstores:
            with st.spinner("Processing PDF..."):
                docs = load_docs_from_upload(file)
                vs = build_vectorstore_from_docs(docs)
                st.session_state.vectorstores[key] = vs
                st.session_state.current_mode = key
                st.session_state.chat_histories[key] = []
            st.rerun()

vectorstore = st.session_state.vectorstores[current]

# ---------------------------
# CHAT
# ---------------------------
if not messages:
    st.info("💡 Ask questions about your documents")

for m in messages:
    cls = "user-msg" if m["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="{cls}">{m["content"]}</div>', unsafe_allow_html=True)

query = st.chat_input("Ask...")

if query:

    # Store meaningful queries for follow-up
    if not query.lower().strip().startswith("and"):
        st.session_state.last_query = query

    messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        ctx, final_query = get_context(
            vectorstore,
            query,
            messages,
            st.session_state.last_query
        )

        if not ctx:
            res = "This information is not available in the provided documents."
        else:
            res = generate_answer(llm, ctx, final_query)

    st.markdown(f'<div class="bot-msg">{res}</div>', unsafe_allow_html=True)

    sources = set([c["source"] for c in ctx])
    st.markdown(
        f'<div class="source-box">📄 Sources: {", ".join(sources)}</div>',
        unsafe_allow_html=True
    )

    messages.append({"role": "assistant", "content": res})

    st.markdown("---")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("👨‍💻 **Made by Farhan**")