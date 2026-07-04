import time
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_chroma import Chroma
from langchain_text_splitters import TextSplitter

from reliablerag.chain import build_rag_chain, TimingCallbackHandler
from reliablerag.evaluation import evaluate
from reliablerag.retriever import get_or_build_vector_store


def run_rag_experiment(
    samples: list,
    retriever_factory: Callable[[Chroma, list[Document]], Runnable],
    embeddings: Embeddings,
    generator_llm: BaseChatModel,
    splitter: TextSplitter,
    persist_dir: str,
    collection_tag: str,
    retrieve_label: str = "retrieve",
    prompt_template: str | None = None,
) -> list[dict]:
    """Run the RAG pipeline over a list of RAGBench samples and return result dicts.

    retriever_factory(vector_store, chunks) → Runnable
      Build the retriever for one sample. Bind any extra args (top_k, reranker, etc.)
      in a lambda or functools.partial before passing in.

    retrieve_label customises the [timing] line, e.g. "hyde retrieve" or "retrieve+rerank".
    generator_llm is the model used to generate the final answer.
    prompt_template pins the prompt for this experiment (import PROMPT_V1 / PROMPT_V2 from
      reliablerag.chain). Defaults to _RAG_PROMPT_TEMPLATE if not provided.
    """
    results = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        raw_doc  = sample["documents"][0]
        coll_name = f"cuad_{i}_{collection_tag}"

        print(f"\n[{i+1}/{len(samples)}] {question[:90]}...")

        chunks = splitter.create_documents(
            [raw_doc],
            metadatas=[{"source": f"cuad_sample_{i}"}],
        )

        t0 = time.perf_counter()
        vector_store, cached = get_or_build_vector_store(
            chunks, embeddings,
            persist_directory=persist_dir,
            collection_name=coll_name,
        )
        print(f"[timing] vector store : {time.perf_counter() - t0:.3f}s  "
              f"({'cache hit' if cached else f'{len(chunks)} chunks embedded'})")

        retriever = retriever_factory(vector_store, chunks)

        t0 = time.perf_counter()
        retrieved_chunks = retriever.invoke(question)
        print(f"[timing] {retrieve_label} : {time.perf_counter() - t0:.3f}s")

        rag_chain    = build_rag_chain(retriever, llm=generator_llm, prompt_template=prompt_template) if prompt_template else build_rag_chain(retriever, llm=generator_llm)
        our_response = rag_chain.invoke(question, config=RunnableConfig(callbacks=[TimingCallbackHandler()]))

        print(f"  our: {our_response}")
        print(f"  ref: {sample['response']}")

        results.append({
            "question"            : question,
            "context"             : raw_doc,
            "retrieved_chunks"    : retrieved_chunks,
            "our_response"        : our_response,
            "ref_response"        : sample["response"],
            "ref_adherence"       : bool(sample["adherence_score"]),
            "ref_relevance"       : sample["relevance_score"],
            "ref_utilization"     : sample["utilization_score"],
            "ref_completeness"    : sample["completeness_score"],
            "ragas_faithfulness"  : sample["ragas_faithfulness"],
            "trulens_groundedness": sample["trulens_groundedness"],
        })

    print(f"\nDone — {len(results)} samples processed.")
    return results


def evaluate_results(
    results: list[dict],
    judge_llm: BaseChatModel,
    n_runs: int = 3,
) -> dict[str, float]:
    """Run TRACe evaluation over results in-place and return aggregate metrics.

    Fills our_adherence, our_relevance, our_utilization, our_completeness,
    our_adherence_explanation, our_relevance_explanation, our_parsed_llm_response
    on each dict in results.

    Returns {"adherence_rate", "avg_relevance", "avg_utilization", "avg_completeness"}.
    """
    for i, r in enumerate(results):
        scores = evaluate(judge_llm, r["question"], r["retrieved_chunks"], r["our_response"], n_runs=n_runs)

        r["our_adherence"]              = scores.adherence
        r["our_adherence_explanation"]  = scores.adherence_explanation
        r["our_relevance"]              = scores.relevance
        r["our_relevance_explanation"]  = scores.relevance_explanation
        r["our_utilization"]            = scores.utilization
        r["our_completeness"]           = scores.completeness
        r["our_parsed_llm_response"]    = scores.parsed_llm_response

        status = "PASS" if scores.adherence else "FAIL"
        print(f"[{i+1}/{len(results)}] [{status}] {r['question'][:70]}...")
        print(f"  Adherence   : {status}  — {scores.adherence_explanation[:90]}")
        print(f"  Relevance   : {scores.relevance:.3f} — {scores.relevance_explanation[:90]}")
        print(f"  Utilization : {scores.utilization:.3f}")
        print(f"  Completeness: {scores.completeness:.3f}")
        print()

    n = len(results)
    return {
        "adherence_rate"  : sum(r["our_adherence"]    for r in results) / n,
        "avg_relevance"   : sum(r["our_relevance"]    for r in results) / n,
        "avg_utilization" : sum(r["our_utilization"]  for r in results) / n,
        "avg_completeness": sum(r["our_completeness"] for r in results) / n,
    }