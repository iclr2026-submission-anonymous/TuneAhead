"""
Dynamic model probe module

Responsible for dynamic feature analysis based on model fine-tuning, including:
- Loss decay rate
- Average gradient norm
- Gradient consistency

Supports final format data: context_text + qa_pairs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
import logging
import random # Added for random.sample

from .data_parsers import HyperParams

logger = logging.getLogger(__name__)

class DynamicProbeAnalyzer:
    """Dynamic probe analyzer (no longer inherits from DatasetAnalyzer)
    
    Specifically for dynamic feature analysis, supports final format data
    """
    def __init__(self, model, tokenizer, device="cuda"):
        """Initialize dynamic probe analyzer"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Apply memory optimization strategy (reference train_model.py)
        # Use more conservative configuration to avoid memory shortage
        self.batch_size = 2  # Reduce batch size
        self.gradient_accumulation_steps = 2  # Add gradient accumulation
        self.max_memory = {0: "20GB", "cpu": "32GB"}  # Limit GPU memory, allow CPU offloading
        self.memory_threshold = 0.95 # Memory usage threshold, force cleanup if exceeded
        
        # Time optimization configuration
        self.enable_fast_mode = True  # Enable fast mode
        self.grad_compute_interval = 3  # Gradient computation interval (steps)
        self.memory_cleanup_interval = 5  # Memory cleanup interval (steps)
        self.use_mixed_precision = True  # Use mixed precision training

    def sample_dataset(self, dataset: List[Dict], sample_size: int) -> List[Dict]:
        """Sample from dataset"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        # Ensure return list instead of numpy array
        sampled = np.random.choice(dataset, sample_size, replace=False)
        return list(sampled)

    def check_memory_usage(self) -> float:
        """Check current GPU memory usage rate"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = allocated / total
            return usage_ratio
        return 0.0

    def force_memory_cleanup(self):
        """Force memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()

    def prepare_training_batches(self, sampled_data: List[Dict], batch_size: int, is_final_format: bool) -> List[Dict[str, torch.Tensor]]:
        """Pre-divide training batches to avoid repeated computation (redesigned core method)"""
        logger.info("Starting to pre-divide training batches...")
        
        # Extract all QA pairs
        qa_list = []
        for item in sampled_data:
            if is_final_format:
                qa_list.extend(self.extract_qa_from_final_format(item))
            else:
                qa_list.append(item)
        
        # Pre-divide batches
        train_batches = []
        for i in range(0, len(qa_list), batch_size):
            batch_qa = qa_list[i:i+batch_size]
            
            # Format batches
            if is_final_format:
                batch_formatted = [self.format_qa_pair(q, a) for q, a in batch_qa]
            else:
                # Process standard format (messages format) data
                batch_formatted = []
                for item in batch_qa:
                    if "messages" in item and len(item["messages"]) >= 2:
                        question = item["messages"][0]["content"]
                        answer = item["messages"][1]["content"]
                        formatted_text = self.format_qa_pair(question, answer)
                        batch_formatted.append(formatted_text)
                    else:
                        # If unable to parse, skip this sample
                        logger.warning(f"Unable to parse data item: {item}")
                        continue
            
            # Use tokenizer to process
            inputs = self.tokenizer(
                batch_formatted,
                return_tensors="pt",
                truncation=True,
                max_length=128,  # Use shorter sequence length
                padding="max_length",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            train_batches.append(inputs)
        
        logger.info(f"Pre-division completed, {len(train_batches)} batches total, batch size {batch_size}")
        return train_batches

    def compute_initial_loss(self, model, train_batches: List[Dict[str, torch.Tensor]]) -> float:
        """Calculate initial loss (using pre-divided batches)"""
        model.eval()
        total_loss = 0.0
        valid_samples = 0
        
        with torch.no_grad():
            for batch_inputs in train_batches:
                try:
                    outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                    loss = outputs.loss
                    if torch.isfinite(loss):
                        total_loss += loss.item() * batch_inputs["input_ids"].size(0)
                        valid_samples += batch_inputs["input_ids"].size(0)
                except Exception as e:
                    logger.warning(f"Error occurred while calculating initial loss: {str(e)}")
                    continue
        
        model.train()
        return total_loss / valid_samples if valid_samples > 0 else 0.0

    def run_streaming_training(
        self, 
        model, 
        optimizer, 
        train_batches: List[Dict[str, torch.Tensor]], 
        probe_steps: int, 
        gradient_accumulation_steps: int, 
        max_grad_norm: float
    ) -> Dict[str, List[float]]:
        """Streaming training loop (true gradient accumulation + streaming processing)"""
        logger.info("Starting streaming training loop...")
        
        # Streaming storage (only store necessary metrics)
        step_losses = []
        step_grad_norms = []
        step_grad_sparsities = []
        
        # Save initial parameters for calculating parameter changes
        initial_params = {}
        for name, param in model.named_parameters():
            if 'lora' in name:
                initial_params[name] = param.data.clone()
        
        for step in tqdm(range(probe_steps), desc="Executing streaming training"):
            optimizer.zero_grad()
            
            # True gradient accumulation: use pre-divided batches
            accumulated_loss = 0
            valid_samples = 0
            
            # Calculate how many pre-divided batches are needed to complete gradient accumulation
            batches_per_step = min(gradient_accumulation_steps, len(train_batches))
            
            for i in range(batches_per_step):
                batch_idx = (step * batches_per_step + i) % len(train_batches)
                batch_inputs = train_batches[batch_idx]
                
                try:
                    # Forward pass
                    outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                    loss = outputs.loss / batches_per_step  # Scale loss
                    
                    if torch.isfinite(loss):
                        accumulated_loss += loss.item() * batch_inputs["input_ids"].size(0)
                        valid_samples += batch_inputs["input_ids"].size(0)
                        loss.backward()
                    
                    # Immediately clean up intermediate variables
                    del outputs, loss
                    
                except Exception as e:
                    logger.warning(f"Error occurred in training step {step} batch {i}: {str(e)}")
                    continue
            
            # After gradient accumulation is complete, update parameters
            if valid_samples > 0 and accumulated_loss > 0:
                try:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    
                    # Calculate current step metrics
                    avg_loss = accumulated_loss / valid_samples
                    step_losses.append(avg_loss)
                    
                    # Calculate gradient metrics at intervals (reduce computation frequency)
                    if step % self.grad_compute_interval == 0:
                        current_grad_norm = self.compute_current_grad_norm(model)
                        current_grad_sparsity = self.compute_current_grad_sparsity(model)
                    else:
                        # Use previous value or default value
                        current_grad_norm = step_grad_norms[-1] if step_grad_norms else 0.0
                        current_grad_sparsity = step_grad_sparsities[-1] if step_grad_sparsities else 0.0
                    
                    step_grad_norms.append(current_grad_norm)
                    step_grad_sparsities.append(current_grad_sparsity)
                    
                except Exception as e:
                    logger.warning(f"Error occurred while updating parameters: {str(e)}")
                    step_losses.append(0.0)
                    step_grad_norms.append(0.0)
                    step_grad_sparsities.append(0.0)
            else:
                step_losses.append(0.0)
                step_grad_norms.append(0.0)
                step_grad_sparsities.append(0.0)
            
            # Clean up memory at intervals (reduce cleanup frequency)
            if torch.cuda.is_available() and step % self.memory_cleanup_interval == 0:
                torch.cuda.empty_cache()
        
        return {
            "step_losses": step_losses,
            "step_grad_norms": step_grad_norms,
            "step_grad_sparsities": step_grad_sparsities,
            "initial_params": initial_params
        }

    def compute_current_grad_norm(self, model) -> float:
        """Calculate current gradient norm (streaming processing)"""
        grad_norm = 0.0
        grad_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
                grad_count += 1
        
        if grad_count > 0:
            grad_norm = grad_norm ** 0.5
            return grad_norm if torch.isfinite(torch.tensor(grad_norm)) else 0.0
        return 0.0

    def compute_current_grad_sparsity(self, model) -> float:
        """Calculate current gradient sparsity (streaming processing)"""
        total_grads = 0
        zero_grads = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                total_grads += grad_flat.numel()
                zero_grads += (grad_flat == 0).sum().item()
        
        return zero_grads / total_grads if total_grads > 0 else 0.0

    def compute_final_features(
        self, 
        initial_loss: float, 
        training_metrics: Dict[str, List[float]], 
        model, 
        initial_params: Dict
    ) -> Dict[str, float]:
        """Calculate final features (streaming processing, no intermediate results stored)"""
        step_losses = training_metrics["step_losses"]
        step_grad_norms = training_metrics["step_grad_norms"]
        step_grad_sparsities = training_metrics["step_grad_sparsities"]
        
        # Calculate final features
        if step_losses:
            final_loss = step_losses[-1]
            loss_decay_rate = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0.0
            
            # Loss stability: use standard deviation of losses
            loss_stability = 1.0 / (1.0 + np.std(step_losses)) if len(step_losses) > 1 else 0.0
            
            # Average gradient norm
            avg_grad_norm = np.mean(step_grad_norms) if step_grad_norms else 0.0
            
            # Average gradient sparsity
            avg_grad_sparsity = np.mean(step_grad_sparsities) if step_grad_sparsities else 0.0
            
            # Calculate parameter change magnitude (streaming processing)
            avg_param_change = self.compute_param_change(model, initial_params)
            
            # Calculate gradient consistency (streaming processing)
            gradient_consistency = self.compute_gradient_consistency(model)
            
            # Calculate activation sparsity (streaming processing)
            avg_activation_sparsity = avg_grad_sparsity  # Use gradient sparsity as proxy
            
            # Calculate other features
            landscape_flatness = 1.0 / (1.0 + gradient_consistency) if gradient_consistency > 0 else 0.0
            catastrophic_forgetting = 1.0 / (1.0 + loss_decay_rate) if loss_decay_rate > 0 else 0.0
        else:
            # Default values
            loss_decay_rate = 0.0
            loss_stability = 0.0
            avg_grad_norm = 0.0
            gradient_consistency = 0.0
            avg_grad_sparsity = 0.0
            avg_param_change = 0.0
            landscape_flatness = 0.0
            catastrophic_forgetting = 0.0
            avg_activation_sparsity = 0.0
        
        return {
            "initial_loss": initial_loss,
            "loss_decay_rate": loss_decay_rate,
            "loss_stability": loss_stability,
            "avg_grad_norm": avg_grad_norm,
            "gradient_consistency": gradient_consistency,
            "avg_grad_sparsity": avg_grad_sparsity,
            "avg_param_change": avg_param_change,
            "landscape_flatness": landscape_flatness,
            "catastrophic_forgetting": catastrophic_forgetting,
            "avg_activation_sparsity": avg_activation_sparsity
        }

    def compute_param_change(self, model, initial_params: Dict) -> float:
        """Calculate parameter change magnitude (streaming processing)"""
        if not initial_params:
            return 0.0
        
        param_change_norm = 0.0
        for name, param in model.named_parameters():
            if 'lora' in name and name in initial_params:
                change = (param.data - initial_params[name]).norm(2).item()
                param_change_norm += change ** 2
        
        return param_change_norm ** 0.5

    def compute_gradient_consistency(self, model) -> float:
        """Calculate gradient consistency (streaming processing)"""
        # Simplified version: calculate variance of current gradients as consistency metric
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                if torch.isfinite(torch.tensor(grad_norm)):
                    grad_norms.append(grad_norm)
        
        if len(grad_norms) > 1:
            # Use coefficient of variation of gradients as consistency metric
            mean_norm = np.mean(grad_norms)
            std_norm = np.std(grad_norms)
            return 1.0 / (1.0 + std_norm / mean_norm) if mean_norm > 0 else 0.0
        
        return 0.0

    def get_default_dynamic_features(self) -> Dict[str, float]:
        """Get default dynamic feature values"""
        return {
            "initial_loss": 0.0,
            "loss_decay_rate": 0.0,
            "loss_stability": 0.0,
            "avg_grad_norm": 0.0,
            "gradient_consistency": 0.0,
            "avg_grad_sparsity": 0.0,
            "avg_param_change": 0.0,
            "landscape_flatness": 0.0,
            "catastrophic_forgetting": 0.0,
            "avg_activation_sparsity": 0.0
        }

    def extract_all_dynamic_features(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        probe_steps: int = 100,
        sample_size: int = 50,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """Redesigned dynamic feature extraction (pre-divided batches + true gradient accumulation + streaming processing)
        
        Returns 10 dynamic features:
        1. initial_loss: Initial loss
        2. loss_decay_rate: Loss decay rate
        3. loss_stability: Loss curve stability
        4. avg_grad_norm: Average gradient norm
        5. gradient_consistency: Gradient consistency
        6. avg_grad_sparsity: Gradient sparsity
        7. avg_param_change: Parameter change magnitude
        8. landscape_flatness: Loss landscape flatness proxy
        9. catastrophic_forgetting: Catastrophic forgetting proxy
        10. avg_activation_sparsity: Activation layer sparsity
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        
        sampled_data = self.sample_dataset(dataset, sample_size)
        is_final_format = self.is_final_format(dataset)
        logger.info(f"Detected data format: {'Final format' if is_final_format else 'Standard format'}")

        # Apply optimization configuration from train_model.py
        lora_r = hyperparams.lora_r
        lora_alpha = hyperparams.lora_alpha
        lora_dropout = hyperparams.lora_dropout
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        
        # Redesigned training hyperparameters
        batch_size = min(batch_size, 2)  # Limit batch size
        gradient_accumulation_steps = 4   # Gradient accumulation steps
        learning_rate = 1e-4             # Learning rate
        max_grad_norm = 0.1              # Gradient clipping
        
        logger.info(f"Redesigned dynamic feature extraction configuration:")
        logger.info(f"  LoRA rank (r): {lora_r}")
        logger.info(f"  LoRA alpha: {lora_alpha}")
        logger.info(f"  LoRA dropout: {lora_dropout}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Max gradient norm: {max_grad_norm}")
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules
        )
        
        try:
            # Save original model state and reference
            original_model_state = self.model.training
            original_model = self.model  # Save original model reference
            logger.info("Saving original model state and reference...")
            
            # Prepare model
            model = get_peft_model(self.model, peft_config)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
            
            # ===== Redesign: Pre-divide training batches =====
            logger.info("Pre-dividing training batches to avoid repeated computation...")
            train_batches = self.prepare_training_batches(sampled_data, batch_size, is_final_format)
            logger.info(f"Pre-division completed, {len(train_batches)} batches total")
            
            # ===== Redesign: Calculate initial loss (using pre-divided batches) =====
            logger.info("Calculating initial loss...")
            initial_loss = self.compute_initial_loss(model, train_batches)
            logger.info(f"Initial loss: {initial_loss:.4f}")
            
            # ===== Redesign: Streaming training loop =====
            logger.info("Starting streaming training loop...")
            training_metrics = self.run_streaming_training(
                model, optimizer, train_batches, probe_steps, 
                gradient_accumulation_steps, max_grad_norm
            )
            
            # ===== Redesign: Calculate final features (streaming processing) =====
            final_features = self.compute_final_features(
                initial_loss, training_metrics, model, training_metrics.get("initial_params")
            )
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error occurred while calculating dynamic features: {str(e)}")
            return self.get_default_dynamic_features()
        finally:
            # Complete cleanup and recovery logic
            logger.info("Starting cleanup and model state recovery...")
            
            try:
                if 'model' in locals():
                    # 1. Remove LoRA adapter - fixed version
                    if hasattr(model, 'peft_config'):
                        logger.info("Removing LoRA adapter...")
                        # Directly restore to original model reference to avoid LoRA adapter residue
                        self.model = original_model
                        logger.info("Successfully restored to original model reference")
                    
                    # 2. Restore original training state
                    self.model.train(original_model_state)
                    logger.info(f"Restored model training state: {original_model_state}")
                    
                    # 3. Clear all gradients
                    self.model.zero_grad()
                    logger.info("Cleared model gradients")
                    
                    # 4. Force garbage collection
                    del model
                    if 'optimizer' in locals():
                        del optimizer
                    import gc
                    gc.collect()
                    logger.info("Executed garbage collection")
                    
            except Exception as cleanup_error:
                logger.warning(f"Error occurred during cleanup: {str(cleanup_error)}")
            
            # 5. Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared GPU memory cache")
            
            logger.info("Model state recovery completed")

    def is_final_format(self, dataset: List[Dict]) -> bool:
        """Determine if it's final format data"""
        if not dataset:
            return False
        sample = dataset[0]
        return "context_text" in sample and "qa_pairs" in sample
    
    def extract_qa_from_final_format(self, item: Dict) -> List[Tuple[str, str]]:
        """Extract all QA pairs from final format data"""
        context = item.get("context_text", "")
        qa_pairs = item.get("qa_pairs", [])
        result = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("output", "")
            if question and answer:
                full_question = f"{context}\n\n{question}" if context else question
                result.append((full_question, answer))
        return result

    def format_qa_pair(self, question: str, answer: str) -> str:
        """Format QA pair as model input format"""
        return f"Question: {question}\nAnswer: {answer}"

    def prepare_batch_inputs(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare training batch inputs (aggressive memory optimization version)"""
        if self.is_final_format(batch_data):
            # Final format data
            formatted_texts = []
            for item in batch_data:
                qa_pairs = self.extract_qa_from_final_format(item)
                for question, answer in qa_pairs:
                    formatted_text = self.format_qa_pair(question, answer)
                    formatted_texts.append(formatted_text)
        else:
            # Standard format data (messages format)
            formatted_texts = []
            for item in batch_data:
                if "messages" in item and len(item["messages"]) >= 2:
                    question = item["messages"][0]["content"]
                    answer = item["messages"][1]["content"]
                    formatted_text = self.format_qa_pair(question, answer)
                    formatted_texts.append(formatted_text)
                else:
                    # If unable to parse, skip this sample
                    logger.warning(f"Unable to parse data item: {item}")
                    continue
        
        # Use more aggressive tokenizer settings (reduce memory usage)
        inputs = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # Further reduce max length, from 256 to 128
            padding="max_length",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs

    def calculate_loss_decay_rate(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        probe_steps: int = 100,
        sample_size: int = 50,
        batch_size: int = 8
    ) -> Tuple[float, float, float]:
        """Calculate loss decay rate, average gradient norm and gradient consistency (supports final format, batch)"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        sampled_data = self.sample_dataset(dataset, sample_size)
        is_final_format = self.is_final_format(dataset)
        logger.info(f"Detected data format: {'Final format' if is_final_format else 'Standard format'}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hyperparams.lora_r,
            lora_alpha=hyperparams.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        try:
            # Save original model state and reference
            original_model_state = self.model.training
            original_model = self.model  # Save original model reference
            logger.info("Saving original model state and reference...")
            
            model = get_peft_model(self.model, peft_config)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)
            initial_loss = 0.0
            step_grad_norms = []
            batch_gradients = []
            qa_list = []
            for item in sampled_data:
                if is_final_format:
                    qa_list.extend(self.extract_qa_from_final_format(item))
                else:
                    qa_list.append(item)
            # Assemble training inputs
            train_batches = []
            for i in range(0, len(qa_list), batch_size):
                batch_qa = qa_list[i:i+batch_size]
                if is_final_format:
                    batch_formatted = [self.format_qa_pair(q, a) for q, a in batch_qa]
                else:
                    # Process standard format (messages format) data
                    batch_formatted = []
                    for item in batch_qa:
                        if "messages" in item and len(item["messages"]) >= 2:
                            question = item["messages"][0]["content"]
                            answer = item["messages"][1]["content"]
                            formatted_text = self.format_qa_pair(question, answer)
                            batch_formatted.append(formatted_text)
                        else:
                            # If unable to parse, skip this sample
                            logger.warning(f"Unable to parse data item: {item}")
                            continue
                inputs = self.tokenizer(
                    batch_formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                train_batches.append(inputs)
            num_batches = len(train_batches)
            for step in tqdm(range(probe_steps), desc="Executing gradient probe (batch)"):
                total_loss = 0.0
                valid_samples = 0
                optimizer.zero_grad()
                for batch_inputs in train_batches:
                    try:
                        outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                        loss = outputs.loss
                        if not torch.isfinite(loss):
                            logger.warning(f"Detected invalid loss value: {loss.item()}")
                            continue
                        total_loss += loss.item() * batch_inputs["input_ids"].size(0)
                        valid_samples += batch_inputs["input_ids"].size(0)
                        loss.backward()
                    except RuntimeError as e:
                        logger.warning(f"Error occurred in training step: {str(e)}")
                        continue
                if valid_samples > 0:
                    avg_loss = total_loss / valid_samples
                    step_grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            step_grad_norm += param.grad.norm(2).item() ** 2
                    step_grad_norm = step_grad_norm ** 0.5
                    if torch.isfinite(torch.tensor(step_grad_norm)):
                        step_grad_norms.append(step_grad_norm)
                        if step == 0:
                            initial_loss = avg_loss
                        if step % 10 == 0 and num_batches >= 2:
                            batch_grad_vectors = []
                            for batch_inputs in train_batches:
                                optimizer.zero_grad()
                                try:
                                    outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                                    loss = outputs.loss
                                    if torch.isfinite(loss):
                                        loss.backward()
                                        grad_vector = []
                                        for param in model.parameters():
                                            if param.grad is not None:
                                                grad_vector.extend(param.grad.flatten().cpu().numpy())
                                        if grad_vector:
                                            batch_grad_vectors.append(np.array(grad_vector))
                                except:
                                    continue
                            if len(batch_grad_vectors) >= 2:
                                batch_similarities = []
                                for i in range(len(batch_grad_vectors)):
                                    for j in range(i + 1, len(batch_grad_vectors)):
                                        min_len = min(len(batch_grad_vectors[i]), len(batch_grad_vectors[j]))
                                        if min_len > 0:
                                            vec1 = batch_grad_vectors[i][:min_len]
                                            vec2 = batch_grad_vectors[j][:min_len]
                                            dot_product = np.dot(vec1, vec2)
                                            norm1 = np.linalg.norm(vec1)
                                            norm2 = np.linalg.norm(vec2)
                                            if norm1 > 0 and norm2 > 0:
                                                similarity = dot_product / (norm1 * norm2)
                                                batch_similarities.append(similarity)
                                if batch_similarities:
                                    batch_gradients.append(np.mean(batch_similarities))
                        optimizer.step()
                    else:
                        logger.warning(f"Step {step} detected invalid gradient norm: {step_grad_norm}")
                        step_grad_norms.append(0.0)
                else:
                    step_grad_norms.append(0.0)
            final_loss = avg_loss if 'avg_loss' in locals() else initial_loss
            loss_decay_rate = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0.0
            avg_grad_norm = np.mean(step_grad_norms) if step_grad_norms else 0.0
            gradient_consistency = np.mean(batch_gradients) if batch_gradients else 0.0
        except Exception as e:
            logger.error(f"Error occurred while calculating loss decay rate: {str(e)}")
            loss_decay_rate = 0.0
            avg_grad_norm = 0.0
            gradient_consistency = 0.0
        finally:
            # Complete cleanup and recovery logic
            logger.info("Starting cleanup and model state recovery...")
            
            try:
                if 'model' in locals():
                    # 1. Remove LoRA adapter - fixed version
                    if hasattr(model, 'peft_config'):
                        logger.info("Removing LoRA adapter...")
                        # Directly restore to original model reference to avoid LoRA adapter residue
                        self.model = original_model
                        logger.info("Successfully restored to original model reference")
                    
                    # 2. Restore original training state
                    self.model.train(original_model_state)
                    logger.info(f"Restored model training state: {original_model_state}")
                    
                    # 3. Clear all gradients
                    self.model.zero_grad()
                    logger.info("Cleared model gradients")
                    
                    # 4. Force garbage collection
                    del model
                    if 'optimizer' in locals():
                        del optimizer
                    import gc
                    gc.collect()
                    logger.info("Executed garbage collection")
                    
            except Exception as cleanup_error:
                logger.warning(f"Error occurred during cleanup: {str(cleanup_error)}")
            
            # 5. Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared GPU memory cache")
            
            logger.info("Model state recovery completed")
        return loss_decay_rate, avg_grad_norm, gradient_consistency 

    def calculate_all_dynamic_features(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        probe_steps: int = 100,
        sample_size: int = 50,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """Redesigned dynamic feature extraction (pre-divided batches + true gradient accumulation + streaming processing)
        
        Returns 10 dynamic features:
        1. initial_loss: Initial loss
        2. loss_decay_rate: Loss decay rate
        3. loss_stability: Loss curve stability
        4. avg_grad_norm: Average gradient norm
        5. gradient_consistency: Gradient consistency
        6. avg_grad_sparsity: Gradient sparsity
        7. avg_param_change: Parameter change magnitude
        8. landscape_flatness: Loss landscape flatness proxy
        9. catastrophic_forgetting: Catastrophic forgetting proxy
        10. avg_activation_sparsity: Activation layer sparsity
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        
        sampled_data = self.sample_dataset(dataset, sample_size)
        is_final_format = self.is_final_format(dataset)
        logger.info(f"Detected data format: {'Final format' if is_final_format else 'Standard format'}")

        # Apply optimization configuration from train_model.py
        lora_r = hyperparams.lora_r
        lora_alpha = hyperparams.lora_alpha
        lora_dropout = hyperparams.lora_dropout
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        
        # Redesigned training hyperparameters
        batch_size = min(batch_size, 2)  # Limit batch size
        gradient_accumulation_steps = 4   # Gradient accumulation steps
        learning_rate = 1e-4             # Learning rate
        max_grad_norm = 0.1              # Gradient clipping
        
        logger.info(f"Redesigned dynamic feature extraction configuration:")
        logger.info(f"  LoRA rank (r): {lora_r}")
        logger.info(f"  LoRA alpha: {lora_alpha}")
        logger.info(f"  LoRA dropout: {lora_dropout}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Max gradient norm: {max_grad_norm}")
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules
        )
        
        try:
            # Prepare model
            model = get_peft_model(self.model, peft_config)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
            
            # ===== Redesign: Pre-divide training batches =====
            logger.info("Pre-dividing training batches to avoid repeated computation...")
            train_batches = self.prepare_training_batches(sampled_data, batch_size, is_final_format)
            logger.info(f"Pre-division completed, {len(train_batches)} batches total")
            
            # ===== Redesign: Calculate initial loss (using pre-divided batches) =====
            logger.info("Calculating initial loss...")
            initial_loss = self.compute_initial_loss(model, train_batches)
            logger.info(f"Initial loss: {initial_loss:.4f}")
            
            # ===== Redesign: Streaming training loop =====
            logger.info("Starting streaming training loop...")
            training_metrics = self.run_streaming_training(
                model, optimizer, train_batches, probe_steps, 
                gradient_accumulation_steps, max_grad_norm
            )
            
            # ===== Redesign: Calculate final features (streaming processing) =====
            final_features = self.compute_final_features(
                initial_loss, training_metrics, model, initial_params=None
            )
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error occurred while calculating dynamic features: {str(e)}")
            # Default values not allowed, must retry or use backup method
            logger.info("🔄 Trying to recalculate dynamic features with more conservative parameters...")
            try:
                # Retry with more conservative parameters
                batch_size = max(1, batch_size // 2)
                sample_size = max(1, sample_size // 2)
                probe_steps = max(1, probe_steps // 2)
                
                # Recreate LoRA configuration with smaller parameters
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=max(2, lora_r // 2),
                    lora_alpha=max(4.0, lora_alpha / 2),
                    lora_dropout=min(0.2, lora_dropout * 2),
                    bias="none",
                    target_modules=target_modules
                )
                
                # Retry
                model = get_peft_model(self.model, peft_config)
                model.train()
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate/2, betas=(0.9, 0.999), weight_decay=0.01)
                
                # Recalculate
                train_batches = self.prepare_training_batches(sampled_data, batch_size, is_final_format)
                initial_loss = self.compute_initial_loss(model, train_batches)
                training_metrics = self.run_streaming_training(
                    model, optimizer, train_batches, probe_steps, 
                    gradient_accumulation_steps, max_grad_norm
                )
                final_features = self.compute_final_features(
                    initial_loss, training_metrics, model, initial_params=None
                )
                
                logger.info(f"✅ Conservative parameter retry succeeded, extracted {len(final_features)} dynamic features")
                return final_features
                
            except Exception as e2:
                logger.error(f"Conservative parameter retry also failed: {str(e2)}")
                raise RuntimeError(f"Unable to calculate any dynamic features: {str(e2)}")
        finally:
            if 'model' in locals():
                model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 

    def extract_dynamic_features(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        sample_size: int = 50,
        batch_size: int = 8,
        probe_steps: int = 100
    ) -> Dict[str, float]:
        """Main method for extracting dynamic features
        
        Args:
            dataset: Dataset
            hyperparams: Hyperparameters
            sample_size: Sample size
            batch_size: Batch size
            probe_steps: Probe steps
        
        Returns:
            Dynamic feature dictionary
        """
        try:
            return self.calculate_all_dynamic_features(
                dataset=dataset,
                hyperparams=hyperparams,
                probe_steps=probe_steps,
                sample_size=sample_size,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Dynamic feature extraction failed: {str(e)}")
            raise RuntimeError(f"Unable to extract dynamic features: {str(e)}")

    def extract_basic_dynamic_features(
        self,
        dataset: List[Dict],
        sample_size: int = 1,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Extract basic dynamic features (minimum configuration, for backup)
        
        Args:
            dataset: Dataset
            sample_size: Sample size
            batch_size: Batch size
        
        Returns:
            Basic dynamic feature dictionary
        """
        try:
            # Use minimum configuration
            hyperparams = HyperParams(
                learning_rate=1e-5,  # Smaller learning rate
                lora_r=4,            # Smaller LoRA rank
                lora_alpha=8.0,      # Smaller alpha
                lora_dropout=0.1     # Larger dropout
            )
            
            return self.calculate_all_dynamic_features(
                dataset=dataset,
                hyperparams=hyperparams,
                probe_steps=1,       # Minimum probe steps
                sample_size=sample_size,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Basic dynamic feature extraction failed: {str(e)}")
            # Try simplest feature extraction
            try:
                # Only calculate model parameter statistics
                model_params = list(self.model.parameters())
                param_norms = [torch.norm(p).item() for p in model_params if p.requires_grad]
                
                if param_norms:
                    return {
                        "initial_loss": 0.0,
                        "loss_decay_rate": 0.0,
                        "loss_stability": 0.0,
                        "avg_grad_norm": np.mean(param_norms),
                        "gradient_consistency": np.std(param_norms),
                        "avg_grad_sparsity": 0.0,
                        "avg_param_change": 0.0,
                        "landscape_flatness": 0.0,
                        "catastrophic_forgetting": 0.0,
                        "avg_activation_sparsity": 0.0
                    }
                else:
                    raise RuntimeError("No trainable parameters")
            except Exception as e2:
                logger.error(f"Simplest dynamic feature extraction also failed: {str(e2)}")
                raise RuntimeError(f"Unable to extract any dynamic features: {str(e2)}")

    def get_default_dynamic_features(self) -> Dict[str, float]:
        """Get default dynamic features (deprecated, not allowed)"""
        raise RuntimeError("Default values not allowed, must calculate real results")
    
    def verify_model_state_cleanup(self) -> bool:
        """Verify if model state is completely cleaned"""
        try:
            # Check if model is in eval mode
            if self.model.training:
                logger.warning("Model is still in training mode")
                return False
            
            # Check if there are LoRA adapters
            if hasattr(self.model, 'peft_config'):
                logger.warning("Model still contains LoRA adapters")
                return False
            
            # Check if gradients are cleared
            for param in self.model.parameters():
                if param.grad is not None:
                    logger.warning("Model parameters still contain gradients")
                    return False
            
            # Check GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                if allocated > 1024 * 1024 * 1024:  # 1GB
                    logger.warning(f"GPU memory usage is high: {allocated / 1024**3:.2f}GB")
                    return False
            
            logger.info("✅ Model state verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Model state verification failed: {str(e)}")
            return False
    
    def enable_fast_mode(self, enabled: bool = True):
        """Enable/disable fast mode"""
        self.enable_fast_mode = enabled
        if enabled:
            self.grad_compute_interval = 3
            self.memory_cleanup_interval = 5
            self.use_mixed_precision = True
            logger.info("🚀 Fast mode enabled")
        else:
            self.grad_compute_interval = 1
            self.memory_cleanup_interval = 1
            self.use_mixed_precision = False
            logger.info("🐌 Standard mode enabled")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        stats = {
            "grad_compute_interval": self.grad_compute_interval,
            "memory_cleanup_interval": self.memory_cleanup_interval,
            "use_mixed_precision": self.use_mixed_precision,
            "enable_fast_mode": self.enable_fast_mode
        }
        
        # Add memory usage statistics
        if torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return stats 