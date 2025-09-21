# MMLU LoRA Training & Feature Extraction System

A comprehensive machine learning experimental platform designed for **LoRA fine-tuning**, **MMLU evaluation**, and **dataset feature analysis**. The system features modular design, supports large-scale batch processing, and includes robust error handling and resource management mechanisms.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Core Features](#core-features)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Installation & Configuration](#installation--configuration)
- [Usage Guide](#usage-guide)
- [Feature Characteristics](#feature-characteristics)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## 🎯 System Overview

This is a comprehensive machine learning experimental platform designed for **LoRA fine-tuning**, **MMLU evaluation**, and **dataset feature analysis**. The system features modular design, supports large-scale batch processing, and includes robust error handling and resource management mechanisms.

### 🏗️ Core Features

#### 1. **Batch MMLU Evaluation** (`batch_mmlu_eval.py`)
- Automatic dataset folder scanning
- Batch processing of multiple dataset files
- Real-time result saving and progress tracking
- Error handling and recovery support

#### 2. **LoRA Fine-tuning Training** (`train_model.py`)
- PEFT library-based LoRA fine-tuning implementation
- 8bit quantization support for memory optimization
- Custom data collators and training loops
- Detailed training metrics recording

#### 3. **MMLU Model Evaluation** (`evaluate_model.py`)
- Standardized MMLU benchmark testing
- Support for base models and LoRA adapters
- Batch inference for improved efficiency
- Intelligent answer extraction and accuracy calculation

#### 4. **Feature Extraction System**
- **Static Features**: Text statistics, semantic features, perplexity features
- **Dynamic Features**: Loss analysis, gradient analysis, parameter changes
- **Isolated Processing**: Subprocess ensures complete isolation
- **CSV Output**: Structured result saving

### 🚀 Quick Start

#### Environment Setup
```bash
# 1. Set model path (required)
export MODEL_NAME="/path/to/your/model"
# Or create .env file
echo "MODEL_NAME=/path/to/your/model" > .env

# 2. Ensure dataset paths are correct
# Batch MMLU evaluation needs: /root/exp/train_data/merged_datasets/chunk_xxx/
# MMLU evaluation dataset: /root/exp/mmlu_dataset/test_all_subjects.json
```

#### Launch Commands
```bash
# 1. Batch MMLU evaluation (requires dataset folders)
python batch_mmlu_eval.py

# 2. Single experiment (requires dataset file)
python auto_mmlu_eval.py

# 3. Feature extraction (requires dataset folder and model path)
python -m feature_extraction.pipe_folder_to_csv_isolated \
    --data_folder /path/to/datasets \
    --model_path /path/to/model \
    --output_csv /path/to/output.csv
```

### 📊 System Architecture

```
Main Control Layer
├── batch_mmlu_eval.py      # Batch controller
├── auto_mmlu_eval.py      # Experiment orchestrator
├── evaluate_model.py      # Model evaluation
└── train_model.py         # LoRA training

Feature Extraction Layer
├── feature_pipeline.py       # Unified pipeline
├── static_features.py        # Static features
├── dynamic_probes.py         # Dynamic features
└── pipe_folder_to_csv_isolated.py  # Isolated processing
```

---

## 🔄 Workflow Overview

### Batch MMLU Evaluation Workflow
```
Dataset Folders → Batch Processing → LoRA Training → MMLU Evaluation → Result Summary
```

### Single Experiment Workflow
```
Dataset → LoRA Training → Model Evaluation → Result Collection → Memory Cleanup
```

### LoRA Training Workflow
```
Data Preprocessing → Model Preparation → LoRA Configuration → Training Execution → Model Saving
```

### MMLU Evaluation Workflow
```
MMLU Dataset → Model Loading → Batch Inference → Answer Extraction → Accuracy Calculation
```

### Feature Extraction Workflow
```
Static Feature Extraction → Dynamic Feature Extraction → Feature Merging → Result Output
```

## 📁 File Structure

```
/root/exp/exp_code/
├── 📁 Main Control Layer
│   ├── batch_mmlu_eval.py      # Batch MMLU evaluation controller
│   ├── auto_mmlu_eval.py      # Automated experiment orchestrator
│   ├── evaluate_model.py      # Model evaluation module
│   └── train_model.py         # LoRA training module
│
└── 📁 Feature Extraction Layer
    └── feature_extraction/
        ├── data_parsers.py           # Data parsing and hyperparameter definition
        ├── feature_pipeline.py       # Unified feature extraction pipeline
        ├── static_features.py        # Static feature extractor
        ├── dynamic_probes.py         # Dynamic feature analyzer
        └── pipe_folder_to_csv_isolated.py  # Isolated batch feature extraction
```

## Installation & Configuration

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU support)
- Recommended 20GB+ GPU memory

### Dependency Installation

```bash
pip install torch transformers peft bitsandbytes
pip install scikit-learn numpy pandas tqdm
pip install python-dotenv
```

### Environment Configuration

```bash
# Set model path in .env file
echo "MODEL_NAME=/path/to/your/model" > .env

# Or set environment variable
export MODEL_NAME="/path/to/your/model"
```

## Usage Guide

### Prerequisites

Before running any commands, ensure:

1. **Model Path**: Set `MODEL_NAME` environment variable
2. **Dataset Structure**: Organize datasets in required folder structure
3. **MMLU Dataset**: Place MMLU evaluation dataset in correct location

### Required Paths

```bash
# Model path (set via environment variable)
export MODEL_NAME="/path/to/your/model"

# Dataset structure for batch processing
/root/exp/train_data/merged_datasets/
├── chunk_001/
│   ├── dataset1.json
│   ├── dataset2.json
│   └── ...
├── chunk_002/
│   └── ...
└── ...

# MMLU evaluation dataset
/root/exp/mmlu_dataset/test_all_subjects.json
```

### 1. Batch MMLU Evaluation

Automatically process multiple dataset folders:

```bash
# Interactive mode
python batch_mmlu_eval.py

# Command line mode
python batch_mmlu_eval.py 1  # Process folder 1
```

**Requirements:**
- Model path set in `MODEL_NAME` environment variable
- Datasets organized in `/root/exp/train_data/merged_datasets/chunk_xxx/` folders
- MMLU dataset located at `/root/exp/mmlu_dataset/test_all_subjects.json`

**Features:**
- Scans `/root/exp/train_data/merged_datasets` for `chunk_xxx` folders
- Processes all JSON files in selected folder
- Real-time progress tracking and result saving
- Comprehensive error handling

### 2. Single Experiment

Run individual LoRA training and MMLU evaluation:

```bash
python auto_mmlu_eval.py
```

**Requirements:**
- Model path set in `MODEL_NAME` environment variable
- Dataset file specified in `auto_mmlu_eval.py` (currently hardcoded)
- MMLU dataset located at `/root/exp/mmlu_dataset/test_all_subjects.json`

**Features:**
- Coordinates LoRA training and MMLU evaluation
- Collects detailed training and evaluation metrics
- Manages GPU memory to prevent leaks
- Generates comprehensive result reports

### 3. Feature Extraction

Extract static and dynamic features from datasets:

```bash
# Batch processing
python -m feature_extraction.pipe_folder_to_csv_isolated \
    --data_folder /path/to/datasets \
    --model_path /path/to/model \
    --output_csv /path/to/output.csv \
    --sample_size 100 \
    --batch_size 1 \
    --probe_steps 20

# Single dataset processing
python -m feature_extraction.pipe_folder_to_csv_isolated \
    --single_dataset /path/to/dataset.json \
    --model_path /path/to/model
```

**Requirements:**
- `--model_path`: Model path (required)
- `--data_folder`: Dataset folder path (for batch processing)
- `--single_dataset`: Single dataset file path (for single processing)
- Dataset files in JSON or JSONL format

**Features:**
- Subprocess isolation for complete resource cleanup
- Incremental CSV writing with real-time saving
- Comprehensive feature extraction (static + dynamic)
- Detailed processing time statistics

## Feature Characteristics

### 🧠 Feature Extraction Types

#### Static Features (`static_features.py`)
- **Text Statistics**: Length, TTR, n-gram repetition rate, special character ratio
- **Semantic Features**: Answer groundedness, embedding outliers, semantic consistency
- **Perplexity Features**: Reference perplexity, base model perplexity, perplexity change rate

#### Dynamic Features (`dynamic_probes.py`)
- **Loss Analysis**: Initial loss, loss decay rate, loss stability
- **Gradient Analysis**: Average gradient norm, gradient consistency, gradient sparsity
- **Parameter Analysis**: Parameter change magnitude, activation sparsity
- **Training Analysis**: Landscape flatness, catastrophic forgetting proxy

### 🔧 Technical Features

#### Memory Optimization
- 8bit/4bit quantization support
- GPU memory limits with CPU offloading
- Automatic memory cleanup and garbage collection
- Streaming processing and batch optimization

#### Error Handling
- Multi-level fallback mechanisms
- Detailed error logging
- Graceful degradation
- Automatic resource cleanup

#### Data Format Support
- JSON/JSONL formats
- Multiple conversation formats (messages, conversations)
- Simple QA pair format
- Final format (context_text + qa_pairs)

## API Reference

### Core Classes

#### `FineTuner` (train_model.py)
```python
class FineTuner:
    def __init__(self):
        """Initialize fine-tuner with environment configuration"""
    
    def prepare_context_dataset(self, data_path: str):
        """Process context-QA JSON format data"""
    
    def prepare_model(self):
        """Prepare model with LoRA configuration"""
    
    def train(self, dataset, model, tokenizer, lora_config_info):
        """Train model with LoRA fine-tuning"""
```

#### `FeatureExtractionPipeline` (feature_pipeline.py)
```python
class FeatureExtractionPipeline:
    def __init__(self, model, tokenizer, device=None):
        """Initialize unified feature extraction pipeline"""
    
    def extract_all_features(self, dataset, hyperparams, sample_size=0, 
                           batch_size=8, probe_steps=100, enable_dynamic_probes=True):
        """Extract all features at once (static + dynamic)"""
    
    def run_quick_pipeline(self, dataset, hyperparams, sample_size=0, 
                          batch_size=4, probe_steps=10):
        """Quick feature extraction pipeline for testing"""
```

#### `StaticFeatureExtractor` (static_features.py)
```python
class StaticFeatureExtractor:
    def extract_all_static_features(self, dataset, sample_size=100, batch_size=8):
        """Extract all static features"""
    
    def calculate_special_char_ratio(self, text: str) -> float:
        """Calculate special character/punctuation ratio"""
    
    def calculate_answer_groundedness(self, context: str, answer: str) -> float:
        """Calculate answer groundedness"""
```

#### `DynamicProbeAnalyzer` (dynamic_probes.py)
```python
class DynamicProbeAnalyzer:
    def extract_all_dynamic_features(self, dataset, hyperparams, 
                                   probe_steps=100, sample_size=50, batch_size=8):
        """Extract all dynamic features"""
    
    def run_streaming_training(self, model, optimizer, train_batches, 
                             probe_steps, gradient_accumulation_steps, max_grad_norm):
        """Streaming training loop with true gradient accumulation"""
```

## Usage Examples

### Example 1: Batch Processing Multiple Datasets

```python
from batch_mmlu_eval import process_folder

# Process all datasets in folder
results = process_folder("chunk_001", max_samples=1000)
print(f"Processed {len(results)} datasets")
```

### Example 2: Single Experiment

```python
from auto_mmlu_eval import run_experiment

# Run single experiment
result = run_experiment(
    dataset_path="/path/to/dataset.json",
    max_samples=100
)
print(f"Accuracy: {result['results']['finetuned_model']['overall_accuracy']:.4f}")
```

### Example 3: Feature Extraction

```python
from feature_extraction.feature_pipeline import FeatureExtractionPipeline
from feature_extraction.data_parsers import HyperParams

# Initialize pipeline
pipeline = FeatureExtractionPipeline(model, tokenizer)

# Define hyperparameters
hyperparams = HyperParams(
    learning_rate=1e-4,
    lora_r=8,
    lora_alpha=16.0,
    lora_dropout=0.1
)

# Extract features
features = pipeline.extract_all_features(
    dataset=dataset,
    hyperparams=hyperparams,
    sample_size=100,
    batch_size=8,
    probe_steps=100
)

print(f"Extracted {len(features)} features")
```

### Example 4: Custom LoRA Training

```python
from train_model import FineTuner

# Initialize fine-tuner
fine_tuner = FineTuner()

# Prepare dataset
dataset = fine_tuner.prepare_context_dataset("/path/to/dataset.json")

# Prepare model
model, tokenizer, lora_config_info = fine_tuner.prepare_model()

# Train model
training_info = fine_tuner.train(dataset, model, tokenizer, lora_config_info)

print(f"Training completed in {training_info['training_metrics']['train_runtime']:.2f} seconds")
```

## Configuration

### Environment Variables

```bash
# Model path (required)
MODEL_NAME="/path/to/your/model"

# Optional configurations
CUDA_VISIBLE_DEVICES="0"
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### Required Directory Structure

```bash
# Project root structure
/root/exp/
├── exp_code/                    # This system
├── train_data/
│   └── merged_datasets/             # Training datasets
│       ├── chunk_001/               # Dataset folder 1
│       │   ├── dataset1.json
│       │   ├── dataset2.json
│       │   └── ...
│       ├── chunk_002/               # Dataset folder 2
│       └── ...
├── mmlu_dataset/                    # MMLU evaluation dataset
│   └── test_all_subjects.json
└── result/                          # Output results
    └── batch_mmlu_results_*.json
```

### Path Configuration Examples

```bash
# Example 1: Local model path
export MODEL_NAME="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# Example 2: Hugging Face model
export MODEL_NAME="meta-llama/Llama-2-7b-hf"

# Example 3: ModelScope model (if supported)
export MODEL_NAME="modelscope://your-org/your-model"
```

### Hyperparameter Configuration

```python
# LoRA Configuration
lora_r = 16                    # LoRA rank
lora_alpha = 32               # LoRA alpha
lora_dropout = 0.1           # LoRA dropout
target_modules = [            # LoRA target modules
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj'
]

# Training Configuration
num_train_epochs = 3
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
max_grad_norm = 0.3
warmup_ratio = 0.03
```

## Performance Optimization

### Memory Management
- Use 8bit quantization for large models
- Enable CPU offloading in memory-constrained environments
- Implement gradient accumulation for large effective batch sizes
- Regular memory cleanup and garbage collection

### Processing Optimization
- Batch processing for multiple datasets
- Subprocess isolation for feature extraction
- Streaming processing for large datasets
- Incremental result saving

### GPU Utilization
- Automatic device mapping
- Mixed precision training (bf16/fp16)
- Memory-efficient attention mechanisms
- Optimized data loading and preprocessing

## Troubleshooting

### Common Issues

#### 1. Path Configuration Issues
```bash
# Error: Model path not found
# Solution: Set correct MODEL_NAME
export MODEL_NAME="/correct/path/to/model"

# Error: Dataset folder not found
# Solution: Ensure correct directory structure
mkdir -p /root/exp/train_data/merged_datasets/chunk_001
# Place JSON dataset files in chunk_001/

# Error: MMLU dataset not found
# Solution: Place MMLU dataset in correct location
mkdir -p /root/exp/mmlu_dataset
# Place test_all_subjects.json in /root/exp/mmlu_dataset/
```

#### 2. CUDA Out of Memory
```bash
# Reduce batch size
batch_size = 1

# Enable CPU offloading
max_memory={0: "10GB", "cpu": "32GB"}

# Use gradient accumulation
gradient_accumulation_steps = 8
```

#### 3. Model Loading Issues
```bash
# Check model path
export MODEL_NAME="/correct/path/to/model"

# Verify model format
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$MODEL_NAME')"

# Check if model supports required format
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME'); print(tokenizer.pad_token)"
```

#### 4. Feature Extraction Failures
```bash
# Reduce sample size
sample_size = 10

# Use minimal batch size
batch_size = 1

# Reduce probe steps
probe_steps = 5
```

## Acknowledgments

- [Transformers](https://github.com/huggingface/transformers) - Model loading and training
- [PEFT](https://github.com/huggingface/peft) - LoRA implementation
- [MMLU](https://github.com/hendrycks/test) - Evaluation benchmark
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - Quantization support

---

**Happy experimenting! 🚀**
