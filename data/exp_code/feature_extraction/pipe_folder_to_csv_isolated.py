#!/usr/bin/env python3
"""
Isolated version of feature extraction pipeline (package version): Use subprocess to ensure complete isolation of each dataset
Input a dataset folder, process datasets one by one, each dataset runs in independent process
For each dataset:
  - Load model and tokenizer in independent subprocess
  - Use FeatureExtractionPipeline to calculate static + dynamic features
  - Incrementally write results to a unified CSV table
  - Automatically clean up all resources after subprocess ends

Header: First column is dataset_name, remaining columns are feature names (based on feature key set of first successful dataset).
"""

import os
import sys
import csv
import json
import gc
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import torch

# Package relative imports
from .feature_pipeline import FeatureExtractionPipeline
from .data_parsers import HyperParams


def _sum_str_lengths(obj: Any) -> int:
    """Recursively count total character count of all strings in object."""
    if obj is None:
        return 0
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(_sum_str_lengths(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_sum_str_lengths(v) for v in obj)
    return 0


def compute_dataset_stats(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate dataset count and total character count (traverse all string fields)."""
    num_items = len(dataset)
    total_chars = 0
    for item in dataset:
        total_chars += _sum_str_lengths(item)
    return {"num_items": num_items, "total_chars": total_chars}


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"\n🔄 Loading model and tokenizer...\n   Model path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("   ✅ Tokenizer loaded successfully")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("   🚀 Using bf16 precision (recommended)")
        torch_dtype = torch.bfloat16
    else:
        print("   ⚠️ Using fp16 precision (bf16 not supported)")
        torch_dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: "20GB", "cpu": "32GB"},
    )
    print("   ✅ Model loaded successfully")
    return model, tokenizer


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset"""
    path = Path(dataset_path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix.lower() == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return data


def process_single_dataset(
    dataset_path: str,
    model_path: str,
    sample_size: int,
    batch_size: int,
    probe_steps: int,
) -> Dict[str, Any]:
    """Process single dataset in single process (for subprocess call)"""
    try:
        dataset = load_dataset(dataset_path)
        print(f"📂 Loading dataset: {dataset_path} (total {len(dataset)} items)")
        # Calculate dataset scale statistics
        stats = compute_dataset_stats(dataset)

        model, tokenizer = load_model_and_tokenizer(model_path)
        
        try:
            pipeline = FeatureExtractionPipeline(model, tokenizer)
            hyperparams = HyperParams(
                learning_rate=1e-4,
                lora_r=8,
                lora_alpha=16.0,
                lora_dropout=0.1,
            )
            print("🚀 Starting feature extraction...")
            features = pipeline.extract_all_features(
                dataset=dataset,
                hyperparams=hyperparams,
                sample_size=sample_size,
                batch_size=batch_size,
                probe_steps=probe_steps,
            )
            print(f"✅ Feature extraction completed: {len(features)} features")
            
            # Output results to stdout for parent process to capture
            result = {
                "success": True,
                "features": features,
                "dataset_name": Path(dataset_path).stem,
                "num_items": stats["num_items"],
                "total_chars": stats["total_chars"],
            }
            print(json.dumps(result, ensure_ascii=False))
            return result
            
        finally:
            # Clean up model and tokenizer
            try:
                if hasattr(model, "disable_adapter"):
                    model.disable_adapter()
                if hasattr(model, "disable_adapters"):
                    model.disable_adapters()
            except Exception:
                pass
            
            del model
            del tokenizer
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("🧹 Model instance cleanup completed")
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "dataset_name": Path(dataset_path).stem
        }
        print(json.dumps(error_result, ensure_ascii=False))
        return error_result


def process_dataset_in_subprocess(
    dataset_path: Path,
    model_path: str,
    sample_size: int,
    batch_size: int,
    probe_steps: int,
) -> Dict[str, Any]:
    """Process dataset in subprocess to ensure complete isolation"""
    print(f"🔄 Starting subprocess to process: {dataset_path.name}")
    
    # Use package running method to ensure module can be located from any working directory
    module_path = "src.pipe_folder_to_csv_isolated"
    cmd = [
        sys.executable, "-m", module_path,
        "--single_dataset", str(dataset_path),
        "--model_path", model_path,
        "--sample_size", str(sample_size),
        "--batch_size", str(batch_size),
        "--probe_steps", str(probe_steps),
    ]
    
    try:
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=os.environ.copy(),
            cwd=str(Path(__file__).resolve().parents[1])  # Project root directory /root/exp
        )
        
        if result.returncode != 0:
            print(f"❌ Subprocess execution failed (return code: {result.returncode})")
            print(f"Error output: {result.stderr}")
            return {"success": False, "error": f"Subprocess failed: {result.stderr}"}
        
        # Parse output
        try:
            output_lines = result.stdout.strip().split('\n')
            json_line = None
            for line in reversed(output_lines):
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    json_line = line.strip()
                    break
            
            if json_line:
                output_data = json.loads(json_line)
                if output_data.get("success"):
                    print(f"✅ Subprocess processing successful: {len(output_data.get('features', {}))} features")
                    return output_data
                else:
                    print(f"❌ Subprocess processing failed: {output_data.get('error')}")
                    return output_data
            else:
                print("❌ Unable to parse subprocess output")
                return {"success": False, "error": "Unable to parse subprocess output"}
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw output: {result.stdout}")
            return {"success": False, "error": f"JSON parsing failed: {e}"}
            
    except subprocess.TimeoutExpired:
        print(f"❌ Subprocess timeout (30 minutes)")
        return {"success": False, "error": "Subprocess timeout"}
    except Exception as e:
        print(f"❌ Subprocess execution exception: {e}")
        return {"success": False, "error": f"Subprocess exception: {e}"}


def normalize_value(value: Any) -> Any:
    """Serialize non-scalar values to JSON strings to avoid CSV write failures"""
    if isinstance(value, (int, float)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def format_time_duration(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def write_row(csv_path: Path, header: List[str], row: Dict[str, Any], is_first_write: bool):
    """Write CSV row"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_first_write:
            writer.writerow(["dataset_name", *header])
        row_values = [row.get(col, "") for col in header]
        writer.writerow([row.get("dataset_name", "")] + [normalize_value(v) for v in row_values])
        f.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Isolated version: Batch feature extraction from folder to CSV (src package)")
    parser.add_argument("--data_folder", type=str, help="Data folder path")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_csv", type=str, default="/root/exp/extracted_features/summary_isolated.csv", help="Output CSV file")
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size")
    parser.add_argument("--batch_size", type=int, default=1, help="Static feature batch size")
    parser.add_argument("--probe_steps", type=int, default=20, help="Dynamic probe steps")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of datasets to process (0 for all)")
    
    # Internal single dataset mode
    parser.add_argument("--single_dataset", type=str, help="Single dataset processing mode (internal use)")
    
    args = parser.parse_args()
    
    # Single dataset mode (subprocess call)
    if args.single_dataset:
        result = process_single_dataset(
            dataset_path=args.single_dataset,
            model_path=args.model_path,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            probe_steps=args.probe_steps,
        )
        return 0 if result.get("success") else 1
    
    # Batch processing mode
    if not args.data_folder:
        print("❌ Batch processing mode requires --data_folder to be specified")
        return 1
    
    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        print(f"❌ Data folder does not exist: {data_folder}")
        return 1

    dataset_files = []
    for ext in ("*.json", "*.jsonl"):
        dataset_files.extend(sorted(data_folder.glob(ext)))

    if args.limit and len(dataset_files) > args.limit:
        dataset_files = dataset_files[: args.limit]

    if not dataset_files:
        print(f"⚠️ No dataset files found in {data_folder}")
        return 0

    csv_path = Path(args.output_csv)
    # If file exists, delete first, then rewrite header
    if csv_path.exists():
        csv_path.unlink()

    header: List[str] = []
    is_first_write = True

    # Record overall start time
    total_start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"🚀 Starting isolated mode processing, {len(dataset_files)} datasets total, will write to: {csv_path}")
    print(f"⏰ Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔒 Each dataset will run in independent subprocess to ensure complete isolation")
    
    # Record processing time for each dataset
    dataset_times = []
    
    for idx, ds_path in enumerate(dataset_files, 1):
        print("\n" + "=" * 60)
        print(f"🔄 Processing {idx}/{len(dataset_files)}: {ds_path.name}")
        print("=" * 60)
        
        # Record single dataset start time
        dataset_start_time = time.time()

        try:
            # Use subprocess isolation mode
            result = process_dataset_in_subprocess(
                dataset_path=ds_path,
                model_path=args.model_path,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
                probe_steps=args.probe_steps,
            )

            if not result.get("success"):
                print(f"❌ Processing failed: {ds_path.name} -> {result.get('error')}")
                continue

            features = result.get("features", {})
            
            # Initialize header (based on feature keys of first successful dataset, sorted alphabetically, fixed column order)
            if not header:
                # Fix dataset scale statistics columns at the front
                metrics_cols = ["dataset_num_items", "dataset_total_chars"]
                feature_cols = sorted(list(features.keys()))
                header = metrics_cols + feature_cols

            row_dict = {"dataset_name": ds_path.stem}
            # Write dataset scale statistics (from result dictionary returned by subprocess)
            row_dict["dataset_num_items"] = result.get("num_items", "")
            row_dict["dataset_total_chars"] = result.get("total_chars", "")
            for k in header:
                if k in ("dataset_num_items", "dataset_total_chars"):
                    continue
                row_dict[k] = features.get(k, "")

            write_row(csv_path, header, row_dict, is_first_write)
            is_first_write = False
            print(f"💾 Written to: {csv_path}")
            
            # Record single dataset completion time
            dataset_end_time = time.time()
            dataset_duration = dataset_end_time - dataset_start_time
            dataset_times.append((ds_path.name, dataset_duration))
            print(f"⏱️ Dataset processing completed, time taken: {format_time_duration(dataset_duration)}")

        except Exception as e:
            print(f"❌ Processing failed: {ds_path.name} -> {e}")
            continue

    # Calculate total time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    end_datetime = datetime.now()
    
    print("\n" + "=" * 60)
    print("📊 Processing Completion Statistics")
    print("=" * 60)
    print(f"⏰ Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total time taken: {format_time_duration(total_duration)}")
    print(f"📁 Number of datasets processed: {len(dataset_times)}")
    
    if dataset_times:
        print("\n📈 Processing time for each dataset:")
        for name, duration in dataset_times:
            print(f"   {name}: {format_time_duration(duration)}")
        
        avg_duration = sum(duration for _, duration in dataset_times) / len(dataset_times)
        print(f"\n📊 Average time per dataset: {format_time_duration(avg_duration)}")
    
    print("\n✅ All processing completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


