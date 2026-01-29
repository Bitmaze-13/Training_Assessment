# # making Simple RAG Chatbot using small dataset

# import streamlit as st
# import re
# import numpy as np
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader


# @st.cache_resource
# def load_models():
#   embed_model = SentenceTransformer('all-MiniLM-L6-v2')
#   llm = pipeline(
#       "text-generation",

#       model = "gpt2",
#       temperature = .8,
#       truncation=True,
#       max_new_tokens = 2000
#   )
#   return embed_model , llm

# embed_model , llm = load_models()

# # advanced_rag_insights = [
# #     "Hybrid Search merges keyword-based BM25 algorithms with vector-based semantic search to capture both exact terminology and conceptual meaning.",
# #     "Reranking models act as a second-pass filter to re-order retrieved chunks, placing the most statistically relevant context at the top for the LLM.",
# #     "GraphRAG utilizes knowledge graphs to map complex relationships between entities, enabling the AI to synthesize answers across disconnected documents.",
# #     "Query Expansion techniques use an LLM to generate multiple versions of a user's prompt, increasing the surface area for finding relevant data.",
# #     "The RAG Triad framework evaluates system performance by measuring context relevance, faithfulness to the source, and answer relevance.",
# #     "Agentic RAG patterns allow the system to autonomously decide whether to search local databases, use web tools, or ask for clarification.",
# #     "Contextual Compression reduces noise by summarizing retrieved chunks before passing them to the generator, saving on token costs and latency.",
# #     "Corrective RAG (CRAG) implements a self-grading mechanism that triggers fallback searches if the initial vector retrieval returns low-confidence results."
# # ]

# # embs = embed_model.encode(advanced_rag_insights)

# def retrieval(query, top_k:int=3, doc=None, embs=None):
#   query_emb = embed_model.encode(query)
#   if isinstance(top_k, list):

#     top_k = int(top_k[0])
#   else:
#     top_k = int(top_k)


#   sim = np.dot(embs, query_emb)
#   part = np.argpartition(sim, -top_k)[-top_k:]
#   top_idx = part[np.argsort(sim[part])[::-1]]
#   return [doc[i] for i in top_idx]



# def pdf_to_text(file_path):
#     reader = PdfReader(file_path)
#     pages_text = []
#     for page in reader.pages:
#         txt = page.extract_text() or ""
#         pages_text.append(txt)  # append strings, not lists
#     return "\n\n".join(pages_text)

# def chunk(text, max_char= 800):
#   parts = [p.strip() for p in text.split("\n") if p.strip()]
#   chunks, buff = [], ""
#   for p in parts:
#     if len(buff) + len(p) <= max_char:
#       buff = (buff + "\n" + p).strip() 
#     else:
#       if buff:
#         chunks.append(buff)
#       buff = p
#   if buff:
#     chunks.append(buff)
#   return chunks



  
# text = pdf_to_text("D:\Python Programs\Training_Assessment\Week 4\DPDP_Act_2023_Reference.pdf")
# docs = chunk(text, max_char=800)
# embs = embed_model.encode(docs)


# def generate_ans(query):
#   retrieved_docs = retrieval(query,top_k=2,doc=text,embs=embs)
#   context = " \n".join(retrieved_docs)

#   prompt = f"""
#   You are an AI assistant.
# Answer the question using ONLY the context below.

# Context:
# {context}

# Question:
# {query}

# Answer :
#   """
#   prompt = prompt[-2000:]
#   response = llm(prompt,max_new_tokens=150, truncation=True)[0]["generated_text"]
#   return response.split("Answer: ")[-1].strip(), retrieved_docs


# st.set_page_config(page_title= "RAG CHATBOT", page_icon="🤖")

# st.title("🤖 RAG Chatbot (GenAI + Deep Learning)")

# st.write("Ask questions based on the internal knowledge base.")

# user_query = st.text_input("Enter your Question Here:")

# if user_query:
#   with st.spinner("Thinking...."):
#     answer, sources = generate_ans(user_query)
  
#   st.subheader("🧠 Answer")
#   st.write(answer)
#   st.subheader("📚 Retrieved Context")
#   for src in sources:
#     src += " "
#     st.markdown(f"- {src}")


# making Simple RAG Chatbot using small dataset

import streamlit as st
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = pipeline(
        "text-generation",
        model="gpt2",
        temperature=0.8,
        truncation=True,
        max_new_tokens=150,           # keep small for GPT-2
        return_full_text=False        # <-- don't echo the prompt in output
    )
    # GPT-2 has no pad token by default
    tok = llm.tokenizer
    mdl = llm.model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.config.pad_token_id = tok.eos_token_id
    return embed_model, llm

embed_model, llm = load_models()

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages_text.append(txt)  # append strings, not lists
    return "\n\n".join(pages_text)

def chunk(text, max_char=800):
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buff = [], ""
    for p in parts:
        # FIX: use <= (not &lt;=)
        if len(buff) + len(p) + 1 <= max_char:
            buff = (buff + "\n" + p).strip()
        else:
            if buff:
                chunks.append(buff)
            buff = p
    if buff:
        chunks.append(buff)
    return chunks

def retrieval(query, top_k: int = 3, doc=None, embs=None):
    # doc must be a list[str] of chunks; embs must be (N, D)
    if not isinstance(doc, list) or embs is None:
        raise ValueError("retrieval expects doc=list[str] and embs=(N,D) array.")
    query_emb = embed_model.encode(query)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(top_k, len(doc)))

    # sim: (N,)
    sim = np.dot(embs, query_emb)
    part = np.argpartition(sim, -top_k)[-top_k:]
    top_idx = part[np.argsort(sim[part])[::-1]]
    return [doc[i] for i in top_idx]

# Use a raw string for Windows path (or double backslashes)
text = pdf_to_text(r"D:\Python Programs\Training_Assessment\Week 4\DPDP_Act_2023_Reference.pdf")

docs = chunk(text, max_char=800)  # list[str] chunks
# Make embeddings for CHUNKS, not the whole text
embs = embed_model.encode(docs, convert_to_numpy=True)

def generate_ans(query):
    # IMPORTANT: pass doc=docs (the chunk list), not the raw text
    retrieved_docs = retrieval(query, top_k=2, doc=docs, embs=embs)
    # Short context for GPT-2
    context = "\n\n---\n\n".join(retrieved_docs)
    context = context[-2000:]  # guard for small-context models

    prompt = f"""You are an AI assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:"""

    # return_full_text=False ensures we only get the new completion
    response = llm(prompt, max_new_tokens=150, truncation=True)[0]["generated_text"]
    answer = response.strip()
    return answer, retrieved_docs

st.set_page_config(page_title="RAG CHATBOT", page_icon="🤖")
st.title("🤖 RAG Chatbot (GenAI + Deep Learning)")
st.write("Ask questions based on the internal knowledge base.")

user_query = st.text_input("Enter your Question Here:")

if user_query:
    with st.spinner("Thinking...."):
        answer, sources = generate_ans(user_query)

    st.subheader("🧠 Answer")
    st.write(answer)
    st.subheader("📚 Retrieved Context")
    for src in sources:
        st.markdown(f"- {src}")