 #!/usr/bin/env python3
"""
Batch processing of folders in merged_datasets
Call methods from auto_mmlu_eval.py to process all training datasets in specified folders at once
"""

import os
import json
import glob
from datetime import datetime
from auto_mmlu_eval import run_experiment

def get_merged_folders():
    """Get all folders in merged_datasets directory"""
    merged_dir = "/root/exp/train_data/merged_datasets"
    if not os.path.exists(merged_dir):
        print(f"❌ merged_datasets directory does not exist: {merged_dir}")
        return []
    
    # Get all folders
    folders = [d for d in os.listdir(merged_dir) 
               if os.path.isdir(os.path.join(merged_dir, d))]
    
    # Filter and sort chunk_xxx format folders
    chunk_folders = [f for f in folders if f.startswith('chunk_')]
    chunk_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    return chunk_folders

def get_dataset_files_in_folder(folder_path):
    """Get all JSON dataset files in specified folder"""
    pattern = os.path.join(folder_path, "*.json")
    files = sorted(glob.glob(pattern))
    return files

def process_folder(folder_name, max_samples=100):
    """
    Process all dataset files in specified folder
    
    Args:
        folder_name: Folder name
        max_samples: Maximum number of samples for MMLU evaluation
    """
    folder_path = os.path.join("/root/exp/train_data/merged_datasets", folder_name)
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder does not exist: {folder_path}")
        return
    
    # Get all dataset files in folder
    dataset_files = get_dataset_files_in_folder(folder_path)
    
    if not dataset_files:
        print(f"❌ No dataset files found in folder {folder_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting to process folder: {folder_name}")
    print(f"Found {len(dataset_files)} dataset files")
    print(f"{'='*60}")
    
    # Prepare result saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = "/root/exp/result"
    os.makedirs(result_dir, exist_ok=True)
    
    results_file = os.path.join(result_dir, f"batch_mmlu_results_{folder_name}_{timestamp}.json")
    
    # Initialize results list
    results = []
    
    # Process each dataset file one by one
    for i, dataset_file in enumerate(dataset_files, 1):
        print(f"\n📊 Processing file {i}/{len(dataset_files)}: {os.path.basename(dataset_file)}")
        
        try:
            # Call run_experiment function from auto_mmlu_eval.py
            result = run_experiment(
                dataset_path=dataset_file,
                max_samples=max_samples
            )
            
            # Add file information to results
            result['file_name'] = os.path.basename(dataset_file)
            result['folder_name'] = folder_name
            
            results.append(result)
            
            print(f"✅ File processing completed: {os.path.basename(dataset_file)}")
            
            # Save current results in real-time
            summary = {
                "folder_name": folder_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(dataset_files),
                "processed_files": len(results),
                "max_samples": max_samples,
                "results": results
            }
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(results)}/{len(dataset_files)} results")
            
        except Exception as e:
            print(f"❌ File processing failed: {os.path.basename(dataset_file)} - {str(e)}")
            error_result = {
                'file_name': os.path.basename(dataset_file),
                'folder_name': folder_name,
                'error': str(e),
                'status': 'failed'
            }
            results.append(error_result)
            
            # Save results including error information
            summary = {
                "folder_name": folder_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(dataset_files),
                "processed_files": len(results),
                "max_samples": max_samples,
                "results": results
            }
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(results)}/{len(dataset_files)} results (including errors)")
    
    print(f"\n💾 Final results saved to: {results_file}")
    
    # Print summary
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"\n📈 Processing Summary:")
    print(f"  Success: {len(successful_results)}/{len(dataset_files)}")
    print(f"  Failed: {len(failed_results)}")
    
    if successful_results:
        print(f"\n📊 Successful Experiment Results:")
        for result in successful_results:
            if 'results' in result and 'finetuned_model' in result['results']:
                accuracy = result['results']['finetuned_model']['overall_accuracy']
                correct = result['results']['finetuned_model']['correct_questions']
                total = result['results']['finetuned_model']['total_questions']
                print(f"  {result['file_name']}: {accuracy:.4f} ({correct}/{total})")
    
    return results

def main():
    """Main function"""
    import sys
    
    print("🚀 Batch MMLU Evaluation Tool")
    print("=" * 60)
    
    # Get all available folders
    folders = get_merged_folders()
    
    if not folders:
        print("❌ No folders found")
        return
    
    print(f"Found {len(folders)} folders:")
    for i, folder in enumerate(folders, 1):
        print(f"  {i:2d}. {folder}")
    
    # Check if there are command line arguments
    if len(sys.argv) > 1:
        try:
            folder_index = int(sys.argv[1]) - 1
            if 0 <= folder_index < len(folders):
                selected_folder = folders[folder_index]
                print(f"\nUsing command line argument to select folder: {selected_folder}")
            else:
                print(f"❌ Command line argument out of range, please enter a number between 1-{len(folders)}")
                return
        except ValueError:
            print("❌ Command line argument must be a number")
            return
    else:
        # User selects folder
        while True:
            try:
                choice = input(f"\nPlease select folder to process (1-{len(folders)}): ")
                folder_index = int(choice) - 1
                
                if 0 <= folder_index < len(folders):
                    selected_folder = folders[folder_index]
                    break
                else:
                    print(f"❌ Please enter a number between 1-{len(folders)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Exiting program")
                return
    
    # Set MMLU evaluation sample count - use all data
    max_samples = None
    print("Will use all MMLU data for evaluation")
    
    print(f"\nStarting to process folder: {selected_folder}")
    print(f"MMLU evaluation: using all data")
    
    # Process selected folder
    results = process_folder(selected_folder, max_samples)
    
    print(f"\n🎉 Batch processing completed!")

if __name__ == "__main__":
    main()