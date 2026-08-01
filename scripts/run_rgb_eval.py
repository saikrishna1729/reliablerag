#!/usr/bin/env python3
import os
import sys
import json
import math
import random
import argparse
import tqdm
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig, DATA_DIR
from src.generation import get_generator
from prompts.rgb import (
    RGB_SYSTEM_PROMPT,
    RGB_INSTRUCTION_TEMPLATE,
    REJECTION_JUDGE_TEMPLATE,
    FACTUAL_JUDGE_TEMPLATE
)

def process_rgb_data(instance, noise_rate, passage_num, dataset_name, correct_rate=0.0):
    """
    Reimplements the data selection logic from RGB's processdata function.
    """
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in dataset_name:
        # Information Integration logic
        shuffled_positives = [list(group) for group in instance['positive']]
        for group in shuffled_positives:
            random.shuffle(group)
        
        docs = [group[0] for group in shuffled_positives if group]
        if len(docs) < pos_num:
            maxnum = max([len(group) for group in shuffled_positives])
            for i in range(1, maxnum):
                for group in shuffled_positives:
                    if len(group) > i:
                        docs.append(group[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in dataset_name:
        # Counterfactual Robustness logic
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        
        # docs get wrong/fake information first
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain, min(len(remain), correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        # Noise Robustness / Negative Rejection logic
        if noise_rate == 1.0:
            neg_num = passage_num
            pos_num = 0
            docs = instance['negative'][:neg_num]
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
            
            positive = instance['positive'][:pos_num]
            negative = instance['negative'][:neg_num]
            docs = positive + negative

    # Ensure we shuffle doc order
    random.shuffle(docs)
    return query, ans, docs

def check_answer(prediction, ground_truth):
    """
    Checks if the ground-truth answer string/list is present in prediction.
    """
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    labels = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def run_llm_judge(judge_generator, prompt_template, **kwargs):
    """
    Queries the LLM judge model with the formatted prompt template.
    """
    prompt = prompt_template.format(**kwargs)
    try:
        response = judge_generator.generate(prompt=prompt, context="")
        return response.strip()
    except Exception as e:
        print(f"Warning: Judge LLM query failed: {e}")
        return "Error"

def main():
    parser = argparse.ArgumentParser(description="RGB English Dataset Evaluation Suite")
    parser.add_argument(
        '--dataset', type=str, default='en',
        choices=['en', 'en_refine', 'en_int', 'en_fact'],
        help='RGB dataset type to evaluate'
    )
    parser.add_argument(
        '--generator', type=str, default='mock',
        help='Generator model name (mock, hf-small, hf-large, llama3:8b, qwen2.5:7b, qwen2.5:14b, etc.)'
    )
    parser.add_argument(
        '--passage_num', type=int, default=5,
        help='Number of retrieved passages to inject'
    )
    parser.add_argument(
        '--noise_rate', type=float, default=0.0,
        help='Ratio of noisy/negative documents in final prompt (0.0 to 1.0)'
    )
    parser.add_argument(
        '--correct_rate', type=float, default=0.0,
        help='Ratio of correct documents (only for fact checking evaluation)'
    )
    parser.add_argument(
        '-n', '--n_records', type=int, default=None,
        help='Number of records to evaluate (None for full dataset)'
    )
    parser.add_argument(
        '--use_llm_judge', action='store_true',
        help='Use LLM-as-a-judge for rejection/counterfactual errors classification'
    )
    parser.add_argument(
        '--judge_generator', type=str, default=None,
        help='Model name for the judge generator (defaults to the same as generator)'
    )
    parser.add_argument(
        '--run_id', type=str, default=None,
        help='Run ID prefix or timestamp to append to output filename. If not provided, a timestamp is generated.'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    data_file_path = os.path.join(DATA_DIR, "rgb", f"{args.dataset}.json")
    if not os.path.exists(data_file_path):
        print(f"Error: Dataset file not found at {data_file_path}")
        print("Please verify that English RGB datasets are in data/rgb/")
        sys.exit(1)
        
    print(f"Loading dataset from: {data_file_path}")
    instances = []
    with open(data_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
                
    if args.n_records is not None and args.n_records > 0:
        instances = instances[:args.n_records]
    print(f"Loaded {len(instances)} records for evaluation.")
    
    # Initialize generators
    print(f"Initializing generator model: {args.generator}...")
    config = ExperimentConfig(generator=args.generator)
    generator = get_generator(config)
    
    judge_generator = None
    if args.use_llm_judge:
        judge_name = args.judge_generator if args.judge_generator else args.generator
        print(f"Initializing judge model: {judge_name}...")
        judge_config = ExperimentConfig(generator=judge_name)
        judge_generator = get_generator(judge_config)

    # Output directory
    output_dir = os.path.join(DATA_DIR.parent, "eval", "results", "rgb")
    os.makedirs(output_dir, exist_ok=True)
    
    import time
    run_id = args.run_id if args.run_id else time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"prediction_{args.dataset}_{args.generator}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}"
    if args.use_llm_judge:
        output_filename += "_judge"
    output_filename += f"_{run_id}"
    output_json_path = os.path.join(output_dir, f"{output_filename}.json")
    output_summary_path = os.path.join(output_dir, f"{output_filename}_summary.json")
    
    print(f"Running evaluation...")
    results = []
    
    for idx, instance in enumerate(tqdm.tqdm(instances)):
        # Apply deterministic seed per instance for doc selection shuffling consistency
        random.seed(2333 + instance.get("id", idx))
        
        if args.passage_num == 0:
            query = instance['query']
            ans = instance['answer']
            docs = []
        else:
            query, ans, docs = process_rgb_data(
                instance, args.noise_rate, args.passage_num, args.dataset, args.correct_rate
            )
            
        # Format context and run generation
        docs_str = '\n'.join(docs) if docs else ""
        prompt_text = RGB_INSTRUCTION_TEMPLATE.format(DOCS=docs_str, QUERY=query)
        
        # Combine system prompt prepended to instructions
        full_input = f"{RGB_SYSTEM_PROMPT}\n\n{prompt_text}"
        
        try:
            prediction = generator.generate(prompt=full_input, context="")
        except Exception as e:
            print(f"\nError generating answer for ID {instance.get('id', idx)}: {e}")
            prediction = f"Error during generation: {e}"
            
        prediction_lower = prediction.lower()
        
        # 1. Answer checking (accurate vs inaccurate)
        is_insufficient = 'insufficient information' in prediction_lower or 'i can not answer' in prediction_lower
        if is_insufficient:
            labels = [-1]
        else:
            labels = check_answer(prediction, ans)
            
        # 2. Factchecking checking
        factlabel = 1 if 'factual errors' in prediction_lower else 0
        
        # 3. Judge-based checks
        judge_rejection = "N/A"
        judge_factcheck = "N/A"
        
        if args.use_llm_judge and judge_generator:
            # Rejection check (only relevant if we evaluate rejection)
            judge_rejection = run_llm_judge(
                judge_generator, REJECTION_JUDGE_TEMPLATE, question=query, answer=prediction
            )
            # Factual error detection check
            judge_factcheck = run_llm_judge(
                judge_generator, FACTUAL_JUDGE_TEMPLATE, answer=prediction
            )
            
        res_instance = {
            'id': instance.get('id', idx),
            'query': query,
            'ans': ans,
            'prediction': prediction,
            'docs': docs,
            'noise_rate': args.noise_rate,
            'labels': labels,
            'factlabel': factlabel,
            'judge_rejection': judge_rejection,
            'judge_factcheck': judge_factcheck
        }
        results.append(res_instance)
        
    # Write detailed predictions
    with open(output_json_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Calculate metrics
    total = len(results)
    correct_count = 0
    rejection_count = 0
    fact_detected_count = 0
    fact_corrected_count = 0
    
    for r in results:
        labels = r['labels']
        # Success definition: 0 not in labels, 1 in labels
        is_correct = (0 not in labels) and (1 in labels)
        
        if is_correct:
            correct_count += 1
            
        # Rejection metric
        if args.use_llm_judge:
            judge_res = r['judge_rejection'].lower()
            if "not addressed" in judge_res or "cannot solve" in judge_res or judge_res.strip().startswith("no"):
                rejection_count += 1
        else:
            if labels == [-1]:
                rejection_count += 1
                
        # Counterfactual metric (Error Detection ED and Error Correction CR)
        has_error_detected = False
        if args.use_llm_judge:
            if "has identified" in r['judge_factcheck'].lower() or "yes" in r['judge_factcheck'].lower():
                has_error_detected = True
        else:
            if r['factlabel'] == 1:
                has_error_detected = True
                
        if has_error_detected:
            fact_detected_count += 1
            if is_correct:
                fact_corrected_count += 1
                
    # Rejection rate (Rej)
    rej_rate = rejection_count / total if total > 0 else 0
    
    # Error Detection rate (ED)
    ed_rate = fact_detected_count / total if total > 0 else 0
    
    # Error Correction rate (CR)
    cr_rate = fact_corrected_count / fact_detected_count if fact_detected_count > 0 else 0
    
    # Overall Accuracy
    # Note: If noise_rate == 1.0 (all noisy, no correct docs), accuracy matches rejection rate.
    if args.noise_rate == 1.0:
        accuracy = rej_rate
    else:
        accuracy = correct_count / total if total > 0 else 0
        
    summary = {
        "dataset": args.dataset,
        "generator": args.generator,
        "passage_num": args.passage_num,
        "noise_rate": args.noise_rate,
        "correct_rate": args.correct_rate,
        "total_records": total,
        "accuracy": accuracy,
        "rejection_rate": rej_rate,
        "error_detection_rate": ed_rate,
        "error_correction_rate": cr_rate
    }
    
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    print("\n=============================================")
    print("             RGB EVALUATION RESULTS          ")
    print("=============================================")
    print(f"Dataset:               {args.dataset}")
    print(f"Generator:             {args.generator}")
    print(f"Total Records:         {total}")
    print(f"Passage Num:           {args.passage_num}")
    print(f"Noise Rate:            {args.noise_rate}")
    if '_fact' in args.dataset:
        print(f"Correct Rate:          {args.correct_rate}")
    print("---------------------------------------------")
    if args.noise_rate == 1.0:
        print(f"Rejection Rate (Rej):  {rej_rate:.4%}")
    else:
        print(f"Accuracy (Acc):        {accuracy:.4%}")
    if '_fact' in args.dataset:
        print(f"Error Detection (ED):  {ed_rate:.4%}")
        print(f"Error Correction (CR): {cr_rate:.4%}")
    print("=============================================")
    print(f"Predictions saved to:  {output_json_path}")
    print(f"Summary saved to:      {output_summary_path}")

if __name__ == "__main__":
    main()
