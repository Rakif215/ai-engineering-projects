from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
import json
import os
import sys

# Optional: Ensure Gemini API Key is loaded since RAGAS will use the default LLM (which we can configure to use Gemini)
if "GEMINI_API_KEY" not in os.environ:
    print("Error: GEMINI_API_KEY not found. RAGAS requires an LLM for evaluation.")
    sys.exit(1)
    
os.environ["OPENAI_API_KEY"] = "mock_key_if_ragas_defaults_to_openai_which_we_override" # RAGAS sometimes looks for this by default

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def load_dataset(filepath="eval/golden_dataset.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Formatting for RAGAS
    # RAGAS expects a huggingface dataset with dicts containing: question, answer, contexts, ground_truth
    # Since we are just evaluating an existing pipeline, we mock the `answer` if generating on the fly,
    # or assuming the pipeline generated answers already. Here we mock for CI demonstrations.
    
    formatted_data = {
        "question": [item["question"] for item in data],
        "answer": [item["ground_truth"] for item in data], # Pre-generated answers (mocking the pipeline's output)
        "contexts": [item["context"] for item in data],
        "ground_truth": [item["ground_truth"] for item in data],
    }
    return Dataset.from_dict(formatted_data)

def run_evaluation():
    print("Loading Golden Dataset...")
    dataset = load_dataset()
    
    print("Initializing Gemini for Evaluation...")
    eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
    
    print("Running RAGAS Evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=ragas_llm,
        embeddings=ragas_emb
    )
    
    print("Evaluation Results:")
    print(result)
    
    # Simple threshold gate for CI/CD
    faithfulness_score = result.get('faithfulness', 0.0)
    if faithfulness_score < 0.85:
        print(f"FAILED: Faithfulness score {faithfulness_score} is below threshold 0.85")
        sys.exit(1)
    else:
        print("PASSED: Quality metrics meet production standards.")

if __name__ == "__main__":
    run_evaluation()
