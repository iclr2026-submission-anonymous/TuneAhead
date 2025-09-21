"""
LoRA-based model fine-tuning script
Main functions:
1. Dataset processing: load JSON format
2. Model training: use LoRA for model fine-tuning
3. Model saving: save trained model to local
Notice:
1. Hyperparameters are hardcoded, need to adjust according to actual situation
2. Dataset paths need to be adjusted according to actual situation
"""

import os
import json
import ast
import logging
import random
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import torch
import numpy as np

class FineTuner:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        # Load environment variables
        load_dotenv()
        # Prioritize reading MODEL_NAME environment variable, then use local default path (supports ModelScope path)
        self.model_name = os.environ.get(
            "MODEL_NAME",
            "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
        )
        self.logger.info(f"Using model path: {self.model_name}")

        # Set random seed
        self.seed = 42
        self.setup_seed(self.seed)
        self.logger.info(f"Set random seed: {self.seed}")

        # Create output directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("fine_tune_outputs", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize dataset_path
        self.dataset_path = None
        
        # Validate environment variables
        if not self.model_name:
            raise ValueError("Please set MODEL_NAME in .env file")

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger('fine_tune')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('fine_tune.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def setup_seed(self, seed):
        """Set all random seeds to ensure reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.logger.info(f"seed {seed}")
        # Some operations may be slower, but this is necessary for reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
    def prepare_context_dataset(self, data_path: str):
        """Process context-QA JSON format data"""
        self.logger.info("Starting to prepare context-QA JSON format dataset...")
        
        try:
            # Save dataset path (keep consistent with original function)
            self.dataset_path = data_path
    
            # Load JSON data
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data format
            if not isinstance(data, list):
                raise ValueError("Dataset should be in JSON array format")
    
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
    
            def process_message(item):
                """Process single message pair"""
                try:
                    messages = item.get("messages", [])
                    if len(messages) != 2:
                        return None
                        
                    # Use Llama format conversation template
                    if messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
                        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{messages[0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{messages[1]['content']}<|eot_id|>"
                    else:
                        # Fallback format
                        text = f"{messages[0]['content']}\n\nAnswer: {messages[1]['content']}"
                    
                    # Tokenization processing
                    result = tokenizer(
                        text,
                        truncation=True,
                        max_length=256,  # Reduce max length to avoid too much padding
                        padding="max_length",
                        return_tensors=None,
                    )
                    
                    # Debug: check data length
                    if len(processed_data) == 0:
                        non_pad_tokens = sum(1 for x in result['input_ids'] if x != tokenizer.pad_token_id)
                        self.logger.info(f"First sample total length: {len(result['input_ids'])}")
                        self.logger.info(f"First sample non-pad token count: {non_pad_tokens}")
                        self.logger.info(f"Pad token ratio: {(len(result['input_ids']) - non_pad_tokens) / len(result['input_ids']):.2%}")
                    
                    return result
                except Exception as e:
                    self.logger.warning(f"Error processing message: {str(e)}")
                    return None
    
            # Process all data entries
            processed_data = []
            for idx, item in enumerate(data, 1):
                result = process_message(item)
                if result is not None:
                    processed_data.append(result)
                else:
                    self.logger.warning(f"Skipping data item {idx}, format does not meet requirements")
            
            if not processed_data:
                raise ValueError("No data entries were successfully processed")
            
            self.logger.info(f"context-QA JSON dataset processing completed, total {len(processed_data)} samples")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error occurred while preparing context-QA JSON dataset: {str(e)}")
            raise

    def prepare_model(self):
        """Prepare model"""
        self.logger.info(f"Starting to load model: {self.model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'  # Fix padding warning
            
            # Configure 8bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory={0: "20GB", "cpu": "32GB"}  # Increase GPU memory limit, allow CPU offloading
            )
            
            # Record LoRA configuration (optimized for Llama model)
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.1
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']  # Llama model attention module names
            
            self.logger.info("LoRA Configuration:")
            self.logger.info(f"  Dataset: {os.path.basename(self.dataset_path)}")
            self.logger.info(f"  LoRA rank (r): {lora_r}")
            self.logger.info(f"  LoRA alpha: {lora_alpha}")
            self.logger.info(f"  LoRA dropout: {lora_dropout}")
            self.logger.info(f"  Target modules: {target_modules}")
            
            # LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules
            )
            
            # Prepare model
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            # Debug: check LoRA parameters
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            self.logger.info(f"Trainable parameters: {trainable_params:,} / {all_param:,} = {100 * trainable_params / all_param:.2f}%")
            
            # Save LoRA configuration information
            lora_config_info = {
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": target_modules
            }
            
            return model, tokenizer, lora_config_info
            
        except Exception as e:
            self.logger.error(f"Error occurred while preparing model: {str(e)}")
            raise

    def train(self, dataset, model, tokenizer, lora_config_info):
        """Train model"""
        self.logger.info("Starting training...")
        
        try:
            # Training hyperparameters h1 (improve memory utilization)
            num_train_epochs = 3
            batch_size = 4  # Increase batch size
            gradient_accumulation_steps = 4  # Reduce gradient accumulation steps
            learning_rate = 2e-4  # Increase learning rate
            max_grad_norm = 0.3
            warmup_ratio = 0.03
            
            # More aggressive configuration (if memory is sufficient)
            # batch_size = 8
            # gradient_accumulation_steps = 2
            # Effective batch size = 8 × 2 = 16
            
            # Training hyperparameters h2
            # num_train_epochs = 3
            # batch_size = 8
            # gradient_accumulation_steps = 2
            # learning_rate = 3e-5
            # max_grad_norm = 0.3
            # warmup_ratio = 0.05
            
            # Optimizer parameters
            optimizer_kwargs = {
                "betas": (0.9, 0.999),    # β=(0.9, 0.999)
                "weight_decay": 0.01       # weight_decay=0.01
            }
            
            # Record training hyperparameters
            self.logger.info("Training hyperparameters:")
            self.logger.info(f"  Training epochs: {num_train_epochs}")
            self.logger.info(f"  Batch size: {batch_size}")
            self.logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  Learning rate: {learning_rate}")
            self.logger.info(f"  Max gradient norm: {max_grad_norm}")
            self.logger.info(f"  Warmup ratio: {warmup_ratio}")
            self.logger.info("Optimizer parameters:")
            self.logger.info(f"  Beta: {optimizer_kwargs['betas']}")
            self.logger.info(f"  Weight Decay: {optimizer_kwargs['weight_decay']}")
           
            # Training parameters
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                bf16=True,
                report_to="none",
                optim="adamw_torch",
                adam_beta1=optimizer_kwargs["betas"][0],
                adam_beta2=optimizer_kwargs["betas"][1],
                weight_decay=optimizer_kwargs["weight_decay"],
                lr_scheduler_type="cosine",
                lr_scheduler_kwargs={"num_cycles": 0.5}
            )
            
            # Custom data collator to ensure labels are set correctly
            def qa_collator(features):
                input_ids = [f["input_ids"] for f in features]
                attention_mask = [f["attention_mask"] for f in features]
                labels = [f["input_ids"] for f in features]  # Key: labels = input_ids
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                }
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=qa_collator  # Use custom collator
            )
            
            # Record initial LoRA parameters
            initial_lora_params = {}
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    param_data = param.data.clone().cpu().numpy()
                    initial_lora_params[name] = {
                        'data': param_data.tolist(),
                        'shape': list(param_data.shape)
                    }
            
            # Start training
            self.logger.info("Starting training process...")
            
            # Record initial loss (after first batch)
            initial_loss = None
            
            # Custom training loop to record initial loss
            def compute_loss(model, inputs):
                outputs = model(**inputs)
                return outputs.loss
            
            # Get first batch to calculate initial loss
            first_batch = next(iter(trainer.get_train_dataloader()))
            if torch.cuda.is_available():
                first_batch = {k: v.cuda() for k, v in first_batch.items()}
            with torch.no_grad():
                initial_loss = compute_loss(model, first_batch).item()
            
            # Start formal training
            train_result = trainer.train()
            
            # Get final loss
            final_loss = train_result.metrics.get('train_loss', 0)
            
            # Record final LoRA parameters
            final_lora_params = {}
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    param_data = param.data.clone().cpu().numpy()
                    final_lora_params[name] = {
                        'data': param_data.tolist(),
                        'shape': list(param_data.shape)
                    }
            
            # Record training results
            self.logger.info("Training completed, training metrics:")
            self.logger.info(f"  Initial loss: {initial_loss:.4f}")
            self.logger.info(f"  Final loss: {final_loss:.4f}")
            self.logger.info(f"  Training duration: {train_result.metrics['train_runtime']:.2f} seconds")
            self.logger.info(f"  Training samples per second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
            self.logger.info(f"  Training steps per second: {train_result.metrics.get('train_steps_per_second', 0):.2f}")
            
            # Save model
            trainer.save_model()
            self.logger.info(f"Model saved to: {self.output_dir}")
            
            # Return detailed training information
            training_info = {
                "trainer": trainer,
                "train_result": train_result,
                "training_metrics": {
                    "train_runtime": train_result.metrics['train_runtime'],
                    "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
                    "train_steps_per_second": train_result.metrics.get('train_steps_per_second', 0),
                    "initial_loss": initial_loss,
                    "final_loss": final_loss
                },
                "training_config": {
                    "num_train_epochs": num_train_epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": learning_rate,
                    "max_grad_norm": max_grad_norm,
                    "warmup_ratio": warmup_ratio,
                    "optimizer_kwargs": optimizer_kwargs
                },
                "lora_config": lora_config_info,
                "lora_params": {
                    "initial": initial_lora_params,
                    "final": final_lora_params
                }
            }
            
            return training_info
            
        except Exception as e:
            self.logger.error(f"Error occurred during training: {str(e)}")
            raise 