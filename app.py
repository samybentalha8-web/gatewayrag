import streamlit as st
import requests

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="GatewayRAG", page_icon="🚪", layout="wide")

st.title("🚪 GatewayRAG")
st.caption("Multi-provider LLM gateway with RAG")

with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Model",
        [
            "auto (fallback chain)",
            "gpt-5-mini",
            "gpt-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "grok-4",
            "solar-pro3",
        ],
    )

    use_rag = st.toggle("Use RAG (search uploaded docs)", value=False)

    st.divider()
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("PDF only", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Ingest document"):
            with st.spinner("Chunking and embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                r = requests.post(f"{API_URL}/upload", files=files)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ Ingested {data['chunks_created']} chunks from {data['filename']}")
                else:
                    st.error(f"Upload failed: {r.text}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {"message": prompt, "use_rag": use_rag}
            if model != "auto (fallback chain)":
                payload["model"] = model

            r = requests.post(f"{API_URL}/chat", json=payload)

            if r.status_code == 200:
                data = r.json()
                reply = data["reply"]
                meta = f"🤖 {data['model_used']}"
                if data.get("rag_used"):
                    meta += f" · 📄 {data['context_chunks_retrieved']} chunks"
                if data.get("fallbacks_attempted"):
                    meta += f" · ⚠️ {len(data['fallbacks_attempted'])} fallback(s)"
                st.markdown(reply)
                st.caption(meta)
                st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
            else:
                err = f"❌ Error: {r.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
