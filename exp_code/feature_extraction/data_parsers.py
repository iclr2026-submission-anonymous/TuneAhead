"""
Data parsing module

Responsible for processing datasets in different formats, including:
- Hyperparameter definition
- Data formatting
- Basic data parsing functionality
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@dataclass
class HyperParams:
    """Hyperparameter configuration class"""
    learning_rate: float
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    # Can add other hyperparameters

class DatasetAnalyzer:
    """Dataset analyzer base class
    
    Provides basic data parsing and model loading functionality
    """
    
    def __init__(
        self,
        base_model_name_or_model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer=None
    ):
        """Initialize dataset analyzer
        
        Args:
            base_model_name_or_model: Pre-trained model name or loaded model object
            device: Computing device
            tokenizer: Loaded tokenizer object (optional)
        """
        self.device = device
        
        # Check if passed model is already loaded
        if hasattr(base_model_name_or_model, 'parameters'):
            # Passed model object
            self.model = base_model_name_or_model
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                raise ValueError("If passing model object, must also pass tokenizer")
        else:
            # Passed model name, need to load
            base_model_name = base_model_name_or_model
            
            # Check if bf16 is supported
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16  # Use bf16 to improve numerical stability
            else:
                torch_dtype = torch.float16   # Fallback to fp16
            
            # First load model on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=None,  # Don't use automatic device mapping
                trust_remote_code=True,
                torch_dtype=torch_dtype,  # Prioritize bf16, fallback to fp16
            )
            # Then manually move model to specified device
            self.model = self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
        
        self.model.eval()  # Set to evaluation mode
        
    def format_qa_pair(self, item: Dict) -> str:
        """Format QA pair as model input format
        
        Args:
            item: Data item, supports multiple formats
            
        Returns:
            Formatted text
        """
        if "messages" in item:
            # Process messages format (conversation format)
            messages = item["messages"]
            formatted_text = ""
            for msg in messages:
                if msg["role"] == "user":
                    formatted_text += f"Question: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_text += f"Answer: {msg['content']}\n"
            return formatted_text.strip()
        elif "conversations" in item:
            # Process Qwen2.5 format
            conversations = item["conversations"]
            formatted_text = ""
            for msg in conversations:
                if msg["role"] == "system":
                    formatted_text += f"{msg['content']}\n"
                elif msg["role"] == "user":
                    formatted_text += f"Question: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_text += f"Answer: {msg['content']}\n"
            return formatted_text.strip()
        else:
            # Process simple QA pair format
            return f"Question: {item['input']}\nAnswer: {item['output']}"
    
    def extract_qa_from_item(self, item: Dict) -> Tuple[str, str]:
        """Extract question and answer from data item
        
        Args:
            item: Data item
            
        Returns:
            (question, answer): Tuple of question and answer
        """
        if "conversations" in item:
            # Process Qwen2.5 format
            question = next(msg["content"] for msg in item["conversations"] 
                          if msg["role"] == "user")
            answer = next(msg["content"] for msg in item["conversations"] 
                        if msg["role"] == "assistant")
        else:
            # Process simple QA pair format
            question = item["input"]
            answer = item["output"]
        
        return question, answer
    
    def get_hyperparams_features(self, hyperparams: HyperParams) -> Dict[str, float]:
        """Extract hyperparameter features
        
        Args:
            hyperparams: Hyperparameter object
            
        Returns:
            Hyperparameter feature dictionary
        """
        return {
            "learning_rate": hyperparams.learning_rate,
            "lora_r": hyperparams.lora_r,
            "lora_alpha": hyperparams.lora_alpha
        }
    
    def get_dataset_size(self, dataset: List[Dict]) -> int:
        """Calculate dataset size
        
        Args:
            dataset: Dataset list
            
        Returns:
            Dataset size
        """
        return len(dataset)
    
    def sample_dataset(self, dataset: List[Dict], sample_size: int) -> List[Dict]:
        """Sample from dataset
        
        Args:
            dataset: Original dataset
            sample_size: Sample size
            
        Returns:
            Sampled dataset
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        # Ensure return list instead of numpy array
        sampled = np.random.choice(dataset, sample_size, replace=False)
        return list(sampled)
    
    def calculate_initial_loss(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """Calculate initial average loss
        
        Args:
            dataset: Dataset
            sample_size: Sample size
            
        Returns:
            Initial average loss
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        total_loss = 0.0
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="Calculate dataset initial loss"):
                # Format input
                formatted_text = self.format_qa_pair(item)
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Calculate loss
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                
        return total_loss / len(sampled_data)

    def extract_all_basic_features(self, dataset: List[Dict], hyperparams: HyperParams, sample_size: int = 100) -> Dict[str, float]:
        """Extract all basic features at once"""
        features = {}
        features.update(self.get_hyperparams_features(hyperparams))
        features["dataset_size"] = self.get_dataset_size(dataset)
        features["initial_loss"] = self.calculate_initial_loss(dataset, sample_size)
        return features 