import pandas as pd
import time
import json
import os
from AgentState import app  # Importing your refined LangGraph pipeline
from langchain_groq import ChatGroq
from settings import settings

# --- 1. CONFIGURATION (Now driven by settings.py) ---
# All paths and keys are now pulled from config.yaml via the settings hub
EXCEL_PATH = settings.GOLDEN_DATASET
OUTPUT_CSV = settings.QUALITY_REPORT

# The Judge uses the QUALITY_MODEL (70B) for superior technical oversight
judge_llm = ChatGroq(
    temperature=0,
    model_name=settings.QUALITY_MODEL,
    groq_api_key=settings.GROQ_API_KEY
)


def get_rag_scores(question, context, answer):
    """Single call to get the RAG Triad scores using LLM-as-a-Judge."""
    prompt = f"""
    ROLE: Scientific Evaluator for RAG Systems.

    INPUT DATA:
    Question: {question}
    Context Chunks: {context}
    Final Answer: {answer}

    SCORING TASKS (0.0 to 1.0):
    1. **Faithfulness**: Is the Answer based ONLY on the Context? (0 if hallucinated).
    2. **Relevance**: Does the Answer directly address the User's Question?
    3. **Precision**: How much of the Context was actually useful for this answer?

    Respond ONLY in JSON format like this:
    {{
        "faithfulness": score, 
        "relevance": score, 
        "precision": score, 
        "reasoning": "one sentence why"
    }}
    """

    try:
        response = judge_llm.invoke(prompt).content
        # Extraction logic to handle potential LLM conversational filler
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception as e:
        print(f"   [!] Judge Error: {e}")
        # Default return to prevent script crashing during batch processing
        return {
            "faithfulness": 0,
            "relevance": 0,
            "precision": 0,
            "reasoning": f"Evaluation Failed: {str(e)}"
        }


# --- 2. THE EVALUATION ENGINE ---

def run_evaluation():
    # 1. Load your gold data (Handles both .xls and .csv formats)
    if EXCEL_PATH.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(EXCEL_PATH)
    else:
        df = pd.read_csv(EXCEL_PATH)

    # --- METICULOUS RESUME LOGIC ---
    if os.path.exists(OUTPUT_CSV):
        # Load existing progress to avoid re-running expensive/long queries
        existing_df = pd.read_csv(OUTPUT_CSV)
        results = existing_df.to_dict('records')
        start_index = len(results)
        print(f"--- RESUMING FROM QUESTION {start_index + 1} ({start_index} already completed) ---")
    else:
        results = []
        start_index = 0
        print(f"--- STARTING NEW EVALUATION: {len(df)} QUESTIONS ---")

    # 2. Loop only through the remaining rows
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        question = row['Question']

        print(f"\n[{i + 1}/{len(df)}] Querying Oracle: {question[:60]}...")

        # A. Run your actual Oracle Pipeline
        # Initializing the state keys to ensure the Responder/Auditor nodes don't KeyError
        inputs = {
            "input": question,
            "docs_available": [],  # Handled internally by AgentState/settings
            "intermediate_steps": [],
            "summaries": [],
            "context": []
        }

        try:
            # Execute the LangGraph pipeline
            final_state = app.invoke(inputs)
            oracle_answer = final_state.get('response', 'No Answer Generated')

            # Extract retrieved context for the Judge
            # If summaries exist, we join them; otherwise use raw context
            retrieved_context = "\n\n".join(final_state.get('summaries', []))
            if not retrieved_context:
                retrieved_context = "\n\n".join(final_state.get('context', []))

            # B. Get the Judge's Verdict (Using the 70B Model)
            scores = get_rag_scores(question, retrieved_context, oracle_answer)

            # C. Collect Data
            result_entry = {
                "Question": question,
                "Gold_Answer": row.get('Answer', 'N/A'),
                "Oracle_Answer": oracle_answer,
                "Faithfulness": scores.get('faithfulness', 0),
                "Relevance": scores.get('relevance', 0),
                "Precision": scores.get('precision', 0),
                "Reasoning": scores.get('reasoning', 'N/A')
            }
            results.append(result_entry)

            # D. CHECKPOINT: Save after every question to prevent data loss
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"   [+] Score: F:{scores.get('faithfulness')} | R:{scores.get('relevance')}")

        except Exception as e:
            # Check for Rate Limits (429) specifically to save your API quota
            print(f"   [X] Pipeline Error on Q{i + 1}: {e}")
            if "429" in str(e):
                print("!!! Rate Limit (TPM/RPM) Detected. Stopping script to preserve quota. !!!")
                break

            # For other errors, we save progress and continue to the next question
            continue

        # E. RATE LIMIT PROTECTION (Uses sleep_time from config.yaml)
        # Standardize timing to prevent bursting on the Groq API
        time.sleep(settings.SLEEP_TIME)

    # --- 3. FINAL SUMMARY ---
    if results:
        final_df = pd.DataFrame(results)
        print("\n" + "=" * 40)
        print("METICULOUS QUALITY SCORECARD")
        print("=" * 40)
        print(f"Avg Faithfulness: {final_df['Faithfulness'].mean():.2%}")
        print(f"Avg Relevance:    {final_df['Relevance'].mean():.2%}")
        print(f"Avg Precision:    {final_df['Precision'].mean():.2%}")
        print(f"Total Processed:  {len(final_df)}/{len(df)}")
        print(f"Report saved to:  {OUTPUT_CSV}")
        print("=" * 40)

if __name__ == "__main__":
    run_evaluation()