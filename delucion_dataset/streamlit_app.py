"""Streamlit demo of the winning pipeline from `17_embedding_generator_goldilocks.ipynb`.

Pipeline (validated in 11-17): sentence + metadata chunks @ 64t -> HyDE query
transform -> hybrid BM25+dense retrieval (alpha=0.3, k=20) with bge-m3
embeddings -> Cohere Rerank 4-Pro to top 2 -> generation with microsoft/phi-4.
Composite score 0.793 on the fixed 20-question eval set (vs. 0.690 baseline).

Lets you pick one of the fixed evaluation questions, run it through the live
pipeline, and compare the generated answer against the dataset's reference answer.
"""

import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragbench_lib.data_loading import load_rag_bench_data
from ragbench_lib.models import get_embedding_model, get_generation_llm
from ragbench_lib.generation_prompt import RAG_GENERATION_PROMPT
from ragbench_lib.chunking import count_tokens, get_sentences
from ragbench_lib.retrievers import HybridRetriever, OpenRouterReranker

load_dotenv()
openrouter_token = os.environ.get("OPENROUTER_TOKEN")

DATASET_NAME = "delucionqa"
EVAL_SAMPLE_SIZE = 20
CHUNK_TARGET_TOKENS = 64
CHUNK_OVERLAP_FRACTION = 0.10
RETRIEVER_ALPHA = 0.3
CANDIDATE_K = 20
FINAL_K = 2
EMBEDDING_MODEL = "baai/bge-m3"
GENERATOR_MODEL = "microsoft/phi-4"

# Reuse the already-embedded bge-m3 store built by 17_embedding_generator_goldilocks.ipynb's
# Phase 2 run (same dataset_name/num_samples/chunking config, so the 456 chunks line up
# exactly) instead of re-embedding via the API on every app start.
EXISTING_VECTOR_STORE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vector_stores", "chroma_delucionqa_emb_winner_baai_bge_m3_9g56e6sb",
)

st.set_page_config(page_title="DelucionQA RAG Demo (Notebook 17 winner)", layout="wide")
st.title("🔍 DelucionQA RAG Pipeline — Winning Config (Notebook 17)")

with st.sidebar:
    st.write("### Configuration")
    st.write("- **Chunking:** Sentence + metadata, 64 tokens, 10% overlap")
    st.write("- **Query transform:** HyDE (1 doc, concatenated)")
    st.write("- **Retrieval:** Hybrid BM25+Dense, alpha=0.3, k=20")
    st.write(f"- **Embedding:** `{EMBEDDING_MODEL}`")
    st.write("- **Reranker:** Cohere Rerank 4-Pro → top 2")
    st.write(f"- **Generator:** `{GENERATOR_MODEL}`")
    st.markdown("---")
    st.caption("Composite score 0.793 vs. 0.690 baseline (11-17 eval series)")


def sentence_level_chunks(docs_df, target_tokens=64, overlap_fraction=0.1):
    from langchain_core.documents import Document

    documents = []
    for _, doc in docs_df.iterrows():
        sentences = get_sentences(doc["text"])
        if not sentences:
            continue
        sent_tokens = [count_tokens(s) for s in sentences]
        n = len(sentences)
        i, chunk_idx = 0, 0
        while i < n:
            chunk_sents, chunk_tok_count, j = [], 0, i
            while j < n and (not chunk_sents or chunk_tok_count + sent_tokens[j] <= target_tokens):
                chunk_sents.append(sentences[j])
                chunk_tok_count += sent_tokens[j]
                j += 1
            chunk_text = " ".join(chunk_sents)
            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": f"{doc['doc_id']}_sent{chunk_idx}",
                    "row_id": doc["row_id"],
                    "tokens": count_tokens(chunk_text),
                },
            ))
            chunk_idx += 1
            overlap_count = int(len(chunk_sents) * overlap_fraction)
            step = max(len(chunk_sents) - overlap_count, 1)
            i += step
    return documents


def metadata_enhanced_chunks(docs_df, target_tokens=64, overlap_fraction=0.1):
    from langchain_core.documents import Document

    documents = sentence_level_chunks(docs_df, target_tokens=target_tokens, overlap_fraction=overlap_fraction)

    doc_id_to_heading = {}
    for _, doc in docs_df.iterrows():
        sentences = get_sentences(doc["text"])
        heading = sentences[0] if sentences else doc["text"][:80]
        doc_id_to_heading[doc["doc_id"]] = " ".join(heading.split()[:12])

    enhanced = []
    for d in documents:
        source_doc_id = d.metadata["chunk_id"].rsplit("_sent", 1)[0]
        heading = doc_id_to_heading.get(source_doc_id, "")
        new_text = f"[Context: {heading}]\n{d.page_content}"
        enhanced.append(Document(
            page_content=new_text,
            metadata={**d.metadata, "tokens": count_tokens(new_text)},
        ))
    return enhanced


def apply_hyde(query, llm, n_docs=1, concat_with_query=True):
    pseudo_docs = []
    for _ in range(n_docs):
        p = f"""Write a comprehensive document that would answer this question:

'{query}'

The document should be detailed and informative, covering all aspects of the answer."""
        try:
            pseudo_docs.append(llm.invoke(p).content.strip())
        except Exception:
            pass
    if not pseudo_docs:
        return query
    combined = " ".join(pseudo_docs)
    return f"{query} {combined}" if concat_with_query else combined


@st.cache_resource
def load_data():
    docs_df = load_rag_bench_data(DATASET_NAME, num_samples=max(EVAL_SAMPLE_SIZE + 10, 30))
    eval_questions_df = docs_df.drop_duplicates(subset=["row_id"]).head(EVAL_SAMPLE_SIZE).reset_index(drop=True)
    return docs_df, eval_questions_df


@st.cache_resource
def build_pipeline():
    docs_df, eval_questions_df = load_data()

    chunks = metadata_enhanced_chunks(docs_df, target_tokens=CHUNK_TARGET_TOKENS, overlap_fraction=CHUNK_OVERLAP_FRACTION)

    embedding_model = get_embedding_model(openrouter_token, model=EMBEDDING_MODEL)
    retriever = HybridRetriever(
        chunks, embedding_model, k=CANDIDATE_K, alpha=RETRIEVER_ALPHA,
        dataset_name=DATASET_NAME, config_name="emb_winner_baai_bge_m3",
        persist_directory=EXISTING_VECTOR_STORE if os.path.isdir(EXISTING_VECTOR_STORE) else None,
    )

    query_llm = get_generation_llm(openrouter_token, temperature=0.7, max_tokens=256)
    reranker = OpenRouterReranker("cohere/rerank-4-pro", openrouter_token)
    generator_llm = get_generation_llm(openrouter_token, model=GENERATOR_MODEL)

    return retriever, query_llm, reranker, generator_llm


docs_df, eval_questions_df = load_data()
retriever, query_llm, reranker, generator_llm = build_pipeline()

questions = eval_questions_df["question"].tolist()
ground_truths = eval_questions_df["response"].tolist()
question_map = {f"Q{i + 1}: {q[:70]}...": i for i, q in enumerate(questions)}

col1, col2 = st.columns([3, 1])
with col1:
    selected_question = st.selectbox("Select a question:", options=list(question_map.keys()), key="question_select")
with col2:
    generate_button = st.button("🔄 Generate Answer", use_container_width=True)

selected_idx = question_map[selected_question]
question = questions[selected_idx]
ground_truth = ground_truths[selected_idx]

st.markdown("---")
st.markdown("### ❓ Question")
st.write(question)

if generate_button:
    with st.spinner("Running HyDE → hybrid retrieval → rerank → generation..."):
        try:
            t0 = time.time()
            retrieval_query = apply_hyde(question, query_llm, n_docs=1, concat_with_query=True)
            candidates = retriever.retrieve(retrieval_query)[:CANDIDATE_K]
            reranked_docs = reranker.rerank(question, candidates, top_k=FINAL_K)
            context = "\n\n".join(d.page_content for d in reranked_docs)

            rag_chain = RAG_GENERATION_PROMPT | generator_llm | StrOutputParser()
            generated_answer = rag_chain.invoke({"context": context, "question": question})
            latency = time.time() - t0

            st.session_state.generated_answer = generated_answer
            st.session_state.retrieved_docs = reranked_docs
            st.session_state.latency = latency
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("### 🤖 Generated Answer")
if "generated_answer" in st.session_state:
    st.success(st.session_state.generated_answer)
    st.caption(f"✓ {len(st.session_state.retrieved_docs)} chunks used · {st.session_state.latency:.2f}s")
    with st.expander("Show retrieved & reranked chunks"):
        for i, d in enumerate(st.session_state.retrieved_docs):
            st.markdown(f"**Chunk {i + 1}** (`{d.metadata.get('chunk_id', '')}`)")
            st.text(d.page_content)
else:
    st.info("Click 'Generate Answer' to run the pipeline")

st.markdown("---")
st.markdown("### ✓ Ground Truth Answer (dataset reference)")
st.info(ground_truth)

st.markdown("---")
st.caption(
    "Pipeline: Sentence+metadata 64t chunks → HyDE → Hybrid retrieval (bge-m3, alpha=0.3) "
    "→ Cohere Rerank 4-Pro → Phi-4 generation. Winning config from 17_embedding_generator_goldilocks.ipynb."
)
