#!/usr/bin/env python3
"""
Evaluate model using local MMLU dataset
Return accuracy
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_mmlu_dataset(dataset_path: str) -> List[Dict]:
    """Load MMLU dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded MMLU dataset: {len(data)} samples")
    return data


def build_prompt(question: str, choices: List[str]) -> str:
    """Build prompt for MMLU questions"""
    # Use Llama format prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Please answer the following multiple choice question:

Question: {question}

Options:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Please only answer with the option letter (A, B, C, or D).<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def extract_answer(response: str) -> Optional[int]:
    """Extract answer index from model response"""
    response = response.strip().upper()
    
    # Try to match A, B, C, D
    for i, letter in enumerate(['A', 'B', 'C', 'D']):
        if response.startswith(letter) or response.endswith(letter):
            return i
    
    # Try to match numbers 0-3
    for i in range(4):
        if str(i) in response:
            return i
    
    return None


def evaluate_mmlu(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    dataset_path: str = "/root/exp/mmlu_dataset/test_all_subjects.json",
    batch_size: int = 4,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance on MMLU dataset
    
    Returns:
        Dictionary containing accuracy
    """
    print("Starting MMLU evaluation...")
    print(f"Base model: {base_model_path}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    
    # Load dataset
    data = load_mmlu_dataset(dataset_path)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Limited sample count: {len(data)}")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Fix padding warning
    tokenizer.padding_side = 'left'
    
    # Configure 8bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: "20GB", "cpu": "32GB"}  # Increase GPU memory limit, allow CPU offloading
    )
    
    # Load adapter (if exists)
    if adapter_path and os.path.exists(adapter_path):
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    
    # Prepare evaluation data
    questions = []
    correct_answers = []
    
    for item in data:
        prompt = build_prompt(item['question'], item['choices'])
        questions.append(prompt)
        correct_answers.append(item['answer'])
    
    # Batch generate answers
    print("Starting to generate answers...")
    correct_count = 0
    total_processed = 0
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        
        # Encode input
        inputs = tokenizer(
            batch_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate answers - fix generation parameter warnings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Deterministic generation
                temperature=None,  # Remove temperature parameter
                top_p=None,       # Remove top_p parameter
                top_k=None,       # Remove top_k parameter
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Evaluate answers
        for j, output in enumerate(outputs):
            # Ensure input_ids is tensor
            if isinstance(inputs['input_ids'][j], list):
                input_length = len(inputs['input_ids'][j])
            else:
                input_length = inputs['input_ids'][j].shape[0]
            generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            
            predicted_answer = extract_answer(generated_text)
            correct_answer = correct_answers[i + j]
            
            if predicted_answer == correct_answer:
                correct_count += 1
            
            total_processed += 1
            
            # Show progress
            if total_processed % 50 == 0 or total_processed == len(data):
                progress = (total_processed / len(data)) * 100
                print(f"Evaluation progress: {total_processed}/{len(data)} ({progress:.1f}%)")
    
    # Calculate accuracy
    total_questions = len(data)
    overall_accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    # Clean GPU memory
    del model
    torch.cuda.empty_cache()
    
    result = {
        'overall_accuracy': overall_accuracy,
        'total_questions': total_questions,
        'correct_questions': correct_count
    }
    
    print(f"\nEvaluation completed!")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({correct_count}/{total_questions})")
    print(f"Random guess accuracy: 25.00%")
    print(f"Model performance: {'Above random' if overall_accuracy > 0.25 else 'Below random'}")
    
    return result


def evaluate_mmlu_with_model(
    model,
    tokenizer,
    dataset_path: str = "/root/exp/mmlu_dataset/test_all_subjects.json",
    batch_size: int = 4,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate MMLU performance using loaded model instance
    
    Args:
        model: Loaded model instance
        tokenizer: Loaded tokenizer instance
        dataset_path: MMLU dataset path
        batch_size: Batch size
        max_samples: Maximum number of samples (optional)
    
    Returns:
        Dictionary containing accuracy
    """
    print("Starting MMLU evaluation (using model instance)...")
    
    # Ensure tokenizer settings are correct
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Fix padding warning
    
    # Load dataset
    data = load_mmlu_dataset(dataset_path)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Limited sample count: {len(data)}")
    
    # Prepare evaluation data
    questions = []
    correct_answers = []
    
    for item in data:
        prompt = build_prompt(item['question'], item['choices'])
        questions.append(prompt)
        correct_answers.append(item['answer'])
    
    # Batch generate answers
    print("Starting to generate answers...")
    correct_count = 0
    total_processed = 0
    
    model.eval()
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        
        # Encode input
        inputs = tokenizer(
            batch_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate answers
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Evaluate answers
        for j, output in enumerate(outputs):
            # Ensure input_ids is tensor
            if isinstance(inputs['input_ids'][j], list):
                input_length = len(inputs['input_ids'][j])
            else:
                input_length = inputs['input_ids'][j].shape[0]
            generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            
            predicted_answer = extract_answer(generated_text)
            correct_answer = correct_answers[i + j]
            
            if predicted_answer == correct_answer:
                correct_count += 1
            
            total_processed += 1
            
            # Show progress
            if total_processed % 100 == 0 or total_processed == len(data):
                progress = (total_processed / len(data)) * 100
                print(f"Evaluation progress: {total_processed}/{len(data)} ({progress:.1f}%)")
    
    # Calculate accuracy
    total_questions = len(data)
    overall_accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    result = {
        'overall_accuracy': overall_accuracy,
        'total_questions': total_questions,
        'correct_questions': correct_count
    }
    
    print(f"\nEvaluation completed!")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({correct_count}/{total_questions})")
    print(f"Random guess accuracy: 25.00%")
    print(f"Model performance: {'Above random' if overall_accuracy > 0.25 else 'Below random'}")
    
    return result


if __name__ == "__main__":
    # Example usage: prioritize reading MODEL_NAME environment variable
    base_model = os.environ.get(
        "MODEL_NAME",
        "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    )
    
    # Evaluate base model
    print("=== Evaluating Base Model ===")
    result = evaluate_mmlu(
        base_model_path=base_model,
        max_samples=100  # For testing, limit sample count
    )
    
    print(f"Overall accuracy: {result['overall_accuracy']:.4f}")
    print(f"Correct answers: {result['correct_questions']}/{result['total_questions']}") 