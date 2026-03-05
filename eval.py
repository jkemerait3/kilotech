#!/usr/bin/env python3
"""
Evaluation harness for the Hawaiian RAG system.
Runs a set of test queries and measures retrieval + generation quality.
"""

import json
import sys
from retrieval import SemanticRetriever
from llm.local_llm import query_llm
from utils import rouge_score, retrieval_precision, retrieval_recall

def load_eval_dataset(filepath="eval_dataset.json"):
    """Load evaluation queries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_rag(retriever, eval_data, max_queries=None):
    """
    Run evaluation queries and compute metrics.
    
    Returns dict with:
    - per-query results (retrieval metrics, generation scores)
    - aggregate statistics
    """
    queries = eval_data['eval_queries']
    if max_queries:
        queries = queries[:max_queries]
    
    results = []
    
    for q_data in queries:
        query = q_data['query']
        ideal_answer = q_data['ideal_answer']
        keywords = q_data['relevant_keywords']
        
        print(f"\n[Query {q_data['id']}] {query}")
        
        # Retrieve
        retrieved_chunks = retriever.retrieve(query, top_n=6, max_total_chars=4000)
        
        # Generate
        context_text = "\n\n".join(retrieved_chunks)
        prompt = (
            "You are an expert in Hawaiian culture and agriculture. "
            "Based on the context below, answer this question concisely and accurately. "
            "Do not name sources.\n\n"
            f"--- Context ---\n{context_text}\n\n"
            f"--- Question ---\n{query}\n\n"
            "Answer:"
        )
        generated_answer = query_llm(prompt)
        
        # Evaluate retrieval (basic: check keyword overlap)
        retrieved_text = " ".join(retrieved_chunks).lower()
        keyword_coverage = sum(1 for kw in keywords if kw.lower() in retrieved_text) / len(keywords)
        
        # Evaluate generation
        gen_rouge = rouge_score(ideal_answer, generated_answer)
        
        result = {
            "query_id": q_data['id'],
            "query": query,
            "num_retrieved_chunks": len(retrieved_chunks),
            "keyword_coverage": round(keyword_coverage, 2),
            "generation_rouge_f1": round(gen_rouge, 2),
            "generated_answer": generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer
        }
        results.append(result)
        
        print(f"  Keyword coverage: {keyword_coverage:.2%}")
        print(f"  ROUGE F1: {gen_rouge:.2f}")
        print(f"  Answer: {result['generated_answer']}")
    
    # Aggregate
    avg_keywords = sum(r['keyword_coverage'] for r in results) / len(results)
    avg_rouge = sum(r['generation_rouge_f1'] for r in results) / len(results)
    
    summary = {
        "num_queries": len(results),
        "avg_keyword_coverage": round(avg_keywords, 2),
        "avg_generation_rouge_f1": round(avg_rouge, 2),
        "per_query_results": results
    }
    
    return summary

def main():
    print("Loading evaluation dataset...")
    eval_data = load_eval_dataset()
    
    print("Initializing retriever...")
    retriever = SemanticRetriever(
        ['Data/hawaiian_chunks'],
        max_chunks=600
    )
    print(f"Loaded {len(retriever.chunks)} chunks")
    
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    # Evaluate on all queries (or limit with max_queries=3 for quick test)
    results = evaluate_rag(retriever, eval_data)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Queries evaluated: {results['num_queries']}")
    print(f"Avg keyword coverage: {results['avg_keyword_coverage']:.2%}")
    print(f"Avg ROUGE F1 (generation): {results['avg_generation_rouge_f1']:.2f}")
    
    # Save results
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to eval_results.json")

if __name__ == "__main__":
    main()
