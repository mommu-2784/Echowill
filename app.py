import re
import os
import uuid

import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("❌ Groq API key not found.")
    st.info(
        "Create a .env file in the same folder as app.py "
        "and add: GROQ_API_KEY=your_new_api_key"
    )
    st.stop()

st.set_page_config(
    page_title="EchoWill",
    page_icon="🕯️",
    layout="centered"
)

DB_PATH = "./echowill_db"
COLLECTION_NAME = "interview_chunks"

MODEL_NAME = "openai/gpt-oss-120b"

@st.cache_resource
def get_groq_client():
    return Groq(api_key=API_KEY)


client = get_groq_client()


@st.cache_resource
def get_chroma_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"❌ ChromaDB could not start:\n\n{e}")
        st.stop()


collection = get_chroma_collection()


def chunk_transcript(text, max_words=80):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        word_count = len(sentence.split())

        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def add_transcript_to_db(collection, transcript, person_name):
    chunks = chunk_transcript(transcript)
    if not chunks:
        return 0

    person_name = person_name.strip()
    upload_id = uuid.uuid4().hex[:12]

    ids = []
    metadatas = []

    for index in range(len(chunks)):
        ids.append(f"{person_name}-{upload_id}-{index}")
        metadatas.append({
            "person": person_name,
            "chunk_index": index,
            "upload_id": upload_id
        })

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def query_values_model(collection, question, person_name, n_results=4):
    person_name = person_name.strip()

    try:
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            where={"person": person_name}
        )
    except Exception as e:
        return (f"❌ Database search error: {e}", [])

    documents = results.get("documents", [])

    if not documents or not documents[0]:
        return (f"No recorded material found for {person_name} yet.", [])

    excerpts = documents[0]

    numbered_excerpts = "\n\n".join(
        f"[Excerpt {i + 1}] {excerpt}"
        for i, excerpt in enumerate(excerpts)
    )

    system_prompt = f"""
You are EchoWill.

You are helping someone understand what
{person_name} might say based ONLY on their
recorded interview material.

IMPORTANT RULES:

1. Use ONLY the transcript excerpts provided.
2. Never invent memories or opinions.
3. Never create facts that are not present.
4. Do not pretend to know {person_name} personally.
5. If the transcript does not provide enough information,
   clearly say that there is not enough recorded material.
6. Give a thoughtful and natural answer.
7. Always mention the relevant excerpt number(s).
8. Make it clear that the response is based on recorded
   material, not a literal new statement from the person.

Recorded transcript excerpts:

{numbered_excerpts}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )

        answer = response.choices[0].message.content
        return answer, excerpts

    except Exception as e:
        return (f"❌ Groq API error:\n\n{e}", excerpts)


def ask_general_assistant(question):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are EchoWill, a friendly general-purpose AI assistant "
                        "inside the EchoWill app. If asked your name or what you are, "
                        "say you are EchoWill's General Assistant. Never say you are "
                        "ChatGPT, OpenAI, or any other product."
                    )
                },
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Groq API error:\n\n{e}"


st.title("🕯️ EchoWill")
st.caption("Preserve how someone thinks — not just what they said.")

tab1, tab2, tab3 = st.tabs(["📼 Add an interview", "💬 Ask", "🤖 General Assistant"])

with tab1:
    st.subheader("📼 Add a recorded interview")

    person_name = st.text_input("Person's name", placeholder="Example: Dad")

    manual_text = st.text_area(
        "Paste a transcript directly",
        height=250,
        placeholder="Paste the interview transcript here..."
    )

    process_button = st.button(
        "Process Interview",
        type="primary",
        disabled=not person_name.strip()
    )

    if process_button:
        transcript = manual_text.strip()

        if not transcript:
            st.warning("⚠️ Please paste a transcript first.")
        else:
            with st.spinner("🧠 Building values model..."):
                try:
                    number_of_chunks = add_transcript_to_db(collection, transcript, person_name)

                    if number_of_chunks > 0:
                        st.success(f"✅ Successfully added {number_of_chunks} transcript chunks for {person_name}.")
                    else:
                        st.warning("⚠️ No usable transcript content found.")

                except Exception as e:
                    st.error(f"❌ Could not process interview:\n\n{e}")

with tab2:
    st.subheader("💬 Ask what they would say")

    ask_person = st.text_input(
        "Whose wisdom are you asking for?",
        key="ask_person",
        placeholder="Example: Dad"
    )

    question = st.text_input(
        "Your question",
        key="personal_question",
        placeholder="Example: What would you advise me about my career?"
    )

    ask_button = st.button(
        "Ask",
        key="personal_ask_button",
        type="primary",
        disabled=not (ask_person.strip() and question.strip())
    )

    if ask_button:
        with st.spinner("🕯️ Thinking..."):
            answer, excerpts = query_values_model(collection, question, ask_person)

        st.markdown("### 🕯️ EchoWill's Answer")
        st.write(answer)

        if excerpts:
            with st.expander("📖 View grounded excerpts"):
                for i, excerpt in enumerate(excerpts):
                    st.markdown(f"**Excerpt {i + 1}**")
                    st.write(excerpt)

with tab3:
    st.subheader("🤖 Ask EchoWill Anything")
    st.caption("General AI help — this is not grounded in a specific person's archive.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask anything...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_general_assistant(user_input)
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.divider()
st.caption("🕯️ EchoWill — Preserving memories, values and wisdom.")