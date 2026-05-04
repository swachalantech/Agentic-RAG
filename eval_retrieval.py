import os
import pandas as pd
from settings import settings
from local_oracle import get_retriever

# --- 1. MODERN IMPORTS WITH IDE SUPPRESSION ---
# These paths are verified for LangChain 1.2.15 'classic' architecture.
# We use # noinspection to tell PyCharm to ignore 'Unresolved Reference' ghost errors.

# noinspection PyUnresolvedReferences
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
# noinspection PyUnresolvedReferences
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank


def run_retrieval_benchmark():
    """
    Evaluates the 'Smarter Librarian' using MMR (Diversity) and
    FlashRank (Re-ranking) to measure Hit Rate and MRR.
    """

    # 1. Load the Golden Dataset from Config
    # Uses single-quote safe paths from settings.py
    dataset_path = settings.GOLDEN_DATASET
    if dataset_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    # 2. Initialize the Base Librarian
    # This returns a 'VectorStoreRetriever' object directly.
    retriever_obj = get_retriever(force_reload=False)

    # Extract params from config (Ensure these exist in your config.yaml!)
    k_fetch = settings.config['retrieval'].get('fetch_k', 100)
    k_final = settings.config['retrieval'].get('k_final', 15)
    lambda_val = settings.config['retrieval'].get('lambda_mult', 0.5)
    rerank_model = settings.config['retrieval'].get('rerank_model', 'ms-marco-MiniLM-L-12-v2')

    # --- UPGRADE 1: DIVERSITY (MMR) ---
    # FIX: We configure retriever_obj directly because it is the base retriever.
    retriever_obj.search_type = "mmr"
    retriever_obj.search_kwargs = {
        "k": k_fetch,
        "fetch_k": 100,
        "lambda_mult": lambda_val
    }

    # --- UPGRADE 2: LOGIC (Reranker) ---
    # We wrap the MMR retriever in a Reranker to get the "Smart" version.
    compressor = FlashrankRerank(model=rerank_model, top_n=k_final)

    # We pass the configured 'retriever_obj' as the base_retriever here.
    smart_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever_obj
    )

    results = []
    print(f"\n--- 🎯 STARTING SMART RETRIEVAL BENCHMARK ---")
    print(f"--- Config: MMR Lambda={lambda_val} | Rerank Top={k_final} ---")

    for index, row in df.iterrows():
        question = row['Question']
        # Handle cases where source name might be slightly different in the Excel
        expected_source = str(row.get('Source', row.get('Expected_Source', ''))).strip()

        print(f"[{index + 1}/{len(df)}] Query: {question[:50]}...")

        try:
            # The "Smart" retrieval call (MMR + FlashRank)
            docs = smart_retriever.invoke(question)

            found_at_rank = -1
            retrieved_sources = []

            for i, doc in enumerate(docs):
                source_file = os.path.basename(doc.metadata.get('source', ''))
                retrieved_sources.append(source_file)

                # Case-insensitive check for the expected document name
                if expected_source.lower() in source_file.lower():
                    found_at_rank = i + 1
                    break

            hit = 1 if found_at_rank > 0 else 0
            mrr = 1 / found_at_rank if found_at_rank > 0 else 0

            results.append({
                "Question": question,
                "Expected": expected_source,
                "Hit": hit,
                "MRR": mrr,
                "Rank": found_at_rank if hit else "N/A",
                "Top_Result": retrieved_sources[0] if retrieved_sources else "None"
            })

            status = f"✅ HIT at Rank {found_at_rank}" if hit else "❌ MISS"
            print(f"   Verdict: {status}")

        except Exception as e:
            print(f"   [!] Error processing query: {e}")
            continue

    # 3. STATISTICAL SUMMARY
    total = len(results)
    if total > 0:
        avg_hit_rate = sum(r['Hit'] for r in results) / total
        avg_mrr = sum(r['MRR'] for r in results) / total

        # Save results to the CSV path specified in config
        output_path = settings.QUALITY_REPORT.replace(".csv", "_retrieval_benchmark.csv")
        pd.DataFrame(results).to_csv(output_path, index=False)

        print("\n" + "=" * 40)
        print("METICULOUS RETRIEVAL SCORECARD")
        print("=" * 40)
        print(f"Hit Rate @{k_final}:           {avg_hit_rate:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
        print(f"Benchmark Report Saved:     {output_path}")
        print("=" * 40)


if __name__ == "__main__":
    # Ensure dependencies are installed: pip install flashrank langchain-classic
    run_retrieval_benchmark()