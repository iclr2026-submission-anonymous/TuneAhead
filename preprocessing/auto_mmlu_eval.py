#!/usr/bin/env python3
"""
Automated LoRA training and MMLU evaluation script
Using optimized training and evaluation methods
"""

import os
import json
import torch
from datetime import datetime
from train_model import FineTuner

def run_experiment(dataset_path: str, max_samples: int = 1000):
    """
    Run single experiment: train LoRA model and evaluate MMLU performance
    
    Args:
        dataset_path: Training dataset path
        max_samples: Maximum number of samples for MMLU evaluation
    
    Returns:
        Experiment result dictionary
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: {os.path.basename(dataset_path)}")
    print(f"{'='*60}")
    
    try:
        # 1. Training phase
        print("\n--- Training Phase ---")
        fine_tuner = FineTuner()
        
        # Prepare dataset
        print(f"Preparing dataset: {dataset_path}")
        dataset = fine_tuner.prepare_context_dataset(dataset_path)
        
        # Prepare model
        print("Preparing model...")
        model, tokenizer, lora_config_info = fine_tuner.prepare_model()
        
        # Start training
        print("Starting training...")
        training_info = fine_tuner.train(dataset, model, tokenizer, lora_config_info)
        
        # 2. Evaluation phase - directly call evaluate_model.py methods
        print("\n--- Evaluation Phase ---")
        
        # Import evaluation function from evaluate_model.py
        from evaluate_model import evaluate_mmlu_with_model
        
        # Directly use trained model for evaluation (without saving model files)
        print("Evaluating trained model...")
        
        # Use trained model for evaluation
        finetuned_result = evaluate_mmlu_with_model(
            model=model,
            tokenizer=tokenizer,
            max_samples=max_samples
        )
        
        # Clean GPU memory
        del model, training_info['trainer']
        torch.cuda.empty_cache()
        
        # 3. Result collection
        result = {
            "dataset": os.path.basename(dataset_path),
            "dataset_path": dataset_path,
            "training_info": {
                "model_name": fine_tuner.model_name,
                "dataset_samples": len(dataset),
                "run_id": fine_tuner.run_id,
                "training_metrics": {
                    "initial_loss": training_info['training_metrics']['initial_loss'],
                    "final_loss": training_info['training_metrics']['final_loss'],
                    "train_runtime": training_info['training_metrics']['train_runtime'],
                    "train_samples_per_second": training_info['training_metrics']['train_samples_per_second'],
                    "train_steps_per_second": training_info['training_metrics']['train_steps_per_second']
                },
                "training_config": training_info['training_config'],
                "lora_training_params": training_info['lora_config']
            },
            "evaluation_info": {
                "max_samples": max_samples,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": {
                "finetuned_model": {
                    "overall_accuracy": finetuned_result['overall_accuracy'],
                    "total_questions": finetuned_result['total_questions'],
                    "correct_questions": finetuned_result['correct_questions']
                }
            }
        }
        
        print(f"✅ Experiment completed: {os.path.basename(dataset_path)}")
        print(f"Fine-tuned model accuracy: {finetuned_result['overall_accuracy']:.4f} ({finetuned_result['correct_questions']}/{finetuned_result['total_questions']})")
        print(f"Training time: {training_info['training_metrics']['train_runtime']:.2f} seconds")
        print(f"Final training loss: {training_info['training_metrics']['final_loss']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        return {
            "dataset": os.path.basename(dataset_path),
            "error": str(e),
            "status": "failed"
        }


def run_all_experiments():
    """Run all experiments"""
    
    # Experiment configuration
    experiments = [
        {
            "name": "dolly_dataset",
            "dataset_path": "/root/exp/train_data/dolly/split_chunks/chunk_001.json",
            "max_samples": 100  # Reduce sample count to speed up testing
        }
    ]
    
    print("🚀 Starting automated LoRA training and MMLU evaluation experiments")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n📊 Experiment {i}/{len(experiments)}: {exp['name']}")
        
        # Check if dataset exists
        if not os.path.exists(exp['dataset_path']):
            print(f"❌ Dataset does not exist: {exp['dataset_path']}")
            continue
            
        # Run experiment
        result = run_experiment(
            dataset_path=exp['dataset_path'],
            max_samples=exp['max_samples']
        )
        
        results.append(result)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📈 Experiment Summary Report")
    print(f"{'='*60}")
    
    successful_experiments = [r for r in results if 'error' not in r]
    failed_experiments = [r for r in results if 'error' in r]
    
    print(f"Successful experiments: {len(successful_experiments)}/{len(experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")
    
    if successful_experiments:
        print("\n📊 Successful Experiment Results:")
        for exp in successful_experiments:
            print(f"\nDataset: {exp['dataset']}")
            print(f"  Fine-tuned model accuracy: {exp['results']['finetuned_model']['overall_accuracy']:.4f} ({exp['results']['finetuned_model']['correct_questions']}/{exp['results']['finetuned_model']['total_questions']})")
    
    if failed_experiments:
        print("\n❌ Failed Experiments:")
        for exp in failed_experiments:
            print(f"  {exp['dataset']}: {exp['error']}")
    
    # Calculate average accuracy
    if successful_experiments:
        avg_accuracy = sum(exp['results']['finetuned_model']['overall_accuracy'] for exp in successful_experiments) / len(successful_experiments)
        print(f"\n📈 Average accuracy: {avg_accuracy:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create result directory
    result_dir = "/root/exp/result"
    os.makedirs(result_dir, exist_ok=True)
    
    results_file = os.path.join(result_dir, f"mmlu_evaluation_results_{timestamp}.json")
    
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(experiments),
            "successful_experiments": len(successful_experiments),
            "failed_experiments": len(failed_experiments)
        },
        "results": results
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"🎉 All experiments completed!")


if __name__ == "__main__":
    run_all_experiments() 