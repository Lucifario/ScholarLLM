import json, re, hashlib, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

eg_codeop={
    "code_snippet": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\ndef run_experiment():\n    data = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])\n    print(f'Using a sample size of {len(data)}.')\n    X_train, X_test, y_train, y_test = train_test_split(data[['A', 'B']], data['C'], test_size=0.2)\n    print(f'Training set size: {len(X_train)}')\n\nif __name__ == '__main__':\n    run_experiment()",
    "justification": "This code tests the effect of different sample sizes on the performance of a scikit-learn model, which is a key consideration in statistical power analysis.",
    "dependencies": ["pandas", "scikit-learn", "numpy"],
    "test_case": "The code should print the sample sizes of the training and test sets."
}

execution_results={
    "stdout": "Using a sample size of 100.\nTraining set size: 80\n",
    "stderr": "",
    "exit_code": 0,
    "success": True,
    "timestamp": 1722513410.975411
}

simulated_hypothesis_output={
    "hypothesis_statement": "A novel graph-based attention mechanism can improve the accuracy of multimodal sentiment analysis by 5% over existing fusion methods on the MIMIC-III dataset.",
    "justification": "This hypothesis builds on existing work on attention mechanisms in the context of multi-modal data fusion as discussed in the papers.",
    "related_concepts": ["Graph-based attention", "multimodal fusion", "sentiment analysis", "MIMIC-III dataset"],
    "potential_methods": ["Transformer-based models", "GNNs", "cross-modal attention"],
    "gaps_addressed": ["Lack of explicit relational context in existing fusion methods."]
}

simulated_abstract_content=[
    "This paper presents a new attention-based fusion model for social media data. We achieve a 92.5% accuracy on the public benchmark, outperforming previous methods. The proposed method utilizes a novel graph structure to better model textual relationships.",

]

def isReproducible(execution_results: dict)->dict:
    """
    Check if the execution results indicate a successful and reproducible experiment.
    """
    is_reproducible=execution_results["success"]
    return {
        "score": 1.0 if is_reproducible else 0.0,
        "details": {
            "success": is_reproducible,
            "exit_code": execution_results['exit_code'],
            "execution_log": execution_results['stdout'] + execution_results['stderr']
        }
    }

def novelty(hypothesis_output: dict, existing_abstracts: list[str])->dict:
    """
    Evaluate the novelty of the hypothesis based on its uniqueness and relevance to the abstract content.
    """
    hypothesis_embed=embedding_function.embed_documents([hypothesis_output["hypothesis_statement"]])
    abstract_embeddings=embedding_function.embed_documents(existing_abstracts)
    
    hypo_vector=np.array(hypothesis_embed[0]).reshape(1, -1)
    abs_vectors=np.array(abstract_embeddings)
    
    if len(abs_vectors)==0:
        return {"score": 1.0, "details": "No existing abstracts to compare against."}
    
    similarities=cosine_similarity(hypo_vector, abs_vectors)
    highest_sim=np.max(similarities)
    novelty_score=1.0 - highest_sim
    return {
        "score": float(novelty_score),
        "details": {
            "highest_similarity_score": float(highest_sim),
            "comparison_count": len(existing_abstracts)
        }
    }

def stats_power_heuristic(execution_results: dict, code_snippet: str)->dict:
    """
    Heuristic to evaluate statistical power based on the presence of key statistical terms in the code and execution results."""
    power_keywords=['sample size', 'training set', 'test set', 'p-value', 't-test', 'chi-square', 'ANOVA', 'F-statistic']
    full_text = code_snippet + " " + execution_results.get("stdout", "")
    
    found_keywords=[kw for kw in power_keywords if re.search(r'\b' + re.escape(kw) + r'\b', full_text, re.IGNORECASE)]
    score=0.0
    if found_keywords:
        score=0.5+0.5*len(found_keywords)/len(power_keywords)
    return {
        "score": min(score, 1.0),
        "details": {
            "found_keywords": found_keywords,
            "keyword_count": len(found_keywords)
        }
    }
def evaluate(generated_code: dict,execution_results: dict, generated_hypothesis: dict, existing_abstracts: list[str])->dict:
    """
    Evaluate the generated code and hypothesis against reproducibility, novelty, and statistical power heuristics.
    """
    reproducibility_score=isReproducible(execution_results)
    novelty_score=novelty(generated_hypothesis,existing_abstracts)
    stats_power_score=stats_power_heuristic(execution_results, generated_code["code_snippet"])
    
    final_scorecard = {
        "metadata": {
            "hypothesis_statement": generated_hypothesis['hypothesis_statement'],
            "code_snippet_hash": hashlib.sha256(generated_code['code_snippet'].encode()).hexdigest(),
            "timestamp": execution_results['timestamp']
        },
        "evaluation_scores": {
            "reproducibility": reproducibility_score,
            "novelty": novelty_score,
            "statistical_power_heuristic": stats_power_score,
        }
    }
    return final_scorecard

if __name__ == "__main__":
    print("\n--- Starting PaperForge Evaluation Pipeline ---")
    simulated_existing_abstracts=simulated_abstract_content+[
        "Another abstract about attention mechanisms in NLP models. We evaluate our approach on a new benchmark dataset and show improved results over traditional methods."
    ]

    scorecard = evaluate(
        generated_code=eg_codeop,
        execution_results=execution_results,
        generated_hypothesis=simulated_hypothesis_output,
        existing_abstracts=simulated_existing_abstracts
    )
    
    print("\n--- Final Scorecard (JSON Output) ---")
    print(json.dumps(scorecard, indent=2))