"""
Feature extraction pipeline

Integrates static and dynamic feature extraction, fully using existing modules:
1. static_features.py - Static feature extraction
2. dynamic_probes.py - Dynamic feature extraction
3. data_parsers.py - Data parsing and hyperparameters

Execution order:
1. Static feature extraction (does not change model parameters)
2. Dynamic feature extraction (uses LoRA, does not affect original model)
3. Merge all features

Supports final format data: context_text + qa_pairs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import json
import os

logger = logging.getLogger(__name__)

# Import existing modules
from .static_features import StaticFeatureExtractor
from .dynamic_probes import DynamicProbeAnalyzer
from .data_parsers import HyperParams


class FeatureExtractionPipeline:
    """Unified feature extraction pipeline
    
    Fully uses existing modules, does not duplicate feature extraction logic
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None
    ):
        """Initialize unified feature extraction pipeline
        
        Args:
            model: Pre-trained model (loaded only once)
            tokenizer: Tokenizer
            device: Computing device, if None then auto-detect
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # For quantized models using device_map="auto", skip .to(device) operation
        # Because transformers has already automatically handled device allocation
        if not hasattr(self.model, 'hf_device_map') or self.model.hf_device_map is None:
            # Ensure model is on correct device
            if next(self.model.parameters()).device != torch.device(self.device):
                self.model = self.model.to(self.device)
        
        # Initialize existing modules
        self.static_extractor = StaticFeatureExtractor(self.model, self.tokenizer, self.device)
        self.dynamic_analyzer = DynamicProbeAnalyzer(self.model, self.tokenizer, self.device)
        
        logger.info(f"Feature extraction pipeline initialization completed, device: {self.device}")
    
    def extract_all_features(
        self, 
        dataset: List[Dict], 
        hyperparams: HyperParams,
        sample_size: int = 0, 
        batch_size: int = 8,
        probe_steps: int = 100,
        enable_dynamic_probes: bool = True
    ) -> Dict[str, float]:
        """Extract all features at once (static + dynamic)
        
        Args:
            dataset: Dataset
            hyperparams: Hyperparameters
            sample_size: Sample size
            batch_size: Batch size
            probe_steps: Probe steps
            enable_dynamic_probes: Whether to enable dynamic feature probes
            
        Returns:
            Dictionary containing all features
        """
        logger.info("🚀 Starting unified feature extraction (static + dynamic)...")
        if sample_size == 0:
            sample_size = len(dataset)
        if not dataset:
            logger.error("❌ Dataset is empty, cannot extract features")
            raise RuntimeError("Dataset is empty, cannot perform feature extraction")
        
        # ===== Phase 1: Static feature extraction =====
        logger.info("📊 Phase 1: Extracting static features...")
        try:
            static_features = self.static_extractor.extract_all_static_features(
                dataset=dataset,
                sample_size=sample_size,
                batch_size=batch_size
            )
            logger.info(f"✅ Static feature extraction completed, {len(static_features)} features total")
        except Exception as e:
            logger.error(f"Static feature extraction failed: {str(e)}")
            static_features = {}
        
        # ===== Phase 2: Dynamic feature extraction =====
        dynamic_features = {}
        if enable_dynamic_probes:
            logger.info("🤖 Phase 2: Extracting dynamic features...")
            try:
                dynamic_features = self.dynamic_analyzer.extract_all_dynamic_features(
                    dataset=dataset,
                    hyperparams=hyperparams,
                    probe_steps=probe_steps,
                    sample_size=sample_size,
                    batch_size=batch_size
                )
                logger.info(f"✅ Dynamic feature extraction completed, {len(dynamic_features)} features total")
            except Exception as e:
                logger.error(f"Dynamic feature extraction failed: {str(e)}")
                # Not allowed to use default values, must retry or use backup method
                logger.info("🔄 Trying backup dynamic feature extraction method...")
                try:
                    # Retry with more conservative parameters
                    dynamic_features = self.dynamic_analyzer.extract_dynamic_features(
                        dataset=dataset,
                        hyperparams=hyperparams,
                        sample_size=max(1, sample_size // 2),  # Reduce sample size
                        batch_size=1,  # Use minimum batch size
                        probe_steps=max(1, probe_steps // 2)   # Reduce probe steps
                    )
                    logger.info(f"✅ Backup dynamic feature extraction successful, {len(dynamic_features)} features total")
                except Exception as e2:
                    logger.error(f"Backup dynamic feature extraction also failed: {str(e2)}")
                    # Try most basic dynamic feature extraction
                    try:
                        dynamic_features = self.dynamic_analyzer.extract_basic_dynamic_features(
                            dataset=dataset,
                            sample_size=1,
                            batch_size=1
                        )
                        logger.info(f"✅ Basic dynamic feature extraction successful, {len(dynamic_features)} features total")
                    except Exception as e3:
                        logger.error(f"All dynamic feature extraction methods failed: {str(e3)}")
                        raise RuntimeError(f"Unable to extract any dynamic features: {str(e3)}")
        else:
            logger.info("⚠️ Dynamic feature probes disabled, but must extract dynamic features")
            try:
                # Try to extract even when disabled
                dynamic_features = self.dynamic_analyzer.extract_dynamic_features(
                    dataset=dataset,
                    hyperparams=hyperparams,
                    sample_size=sample_size,
                    batch_size=batch_size,
                    probe_steps=probe_steps
                )
                logger.info(f"✅ Forced dynamic feature extraction successful, {len(dynamic_features)} features total")
            except Exception as e:
                logger.error(f"Forced dynamic feature extraction failed: {str(e)}")
                raise RuntimeError(f"Unable to extract dynamic features: {str(e)}")
        
        # ===== Phase 3: Merge all features =====
        logger.info("📈 Phase 3: Merging all features...")
        all_features = {}
        
        # Merge static features
        if static_features:
            all_features.update(static_features)
        
        # Merge dynamic features
        all_features.update(dynamic_features)
        
        logger.info(f"✅ Unified feature extraction completed, {len(all_features)} features total")
        logger.info(f"📊 Feature keys: {list(all_features.keys())}")
        return all_features
    
    def run_quick_pipeline(
        self, 
        dataset: List[Dict], 
        hyperparams: HyperParams,
        sample_size: int = 0,
        batch_size: int = 4,
        probe_steps: int = 10
    ) -> Dict[str, float]:
        """Quick feature extraction pipeline (for testing and quick validation)
        
        Args:
            dataset: Dataset
            hyperparams: Hyperparameters
            sample_size: Sample size (used for both static and dynamic features)
            batch_size: Batch size
            probe_steps: Probe steps
        
        Returns:
            Dictionary containing all features
        """
        if sample_size == 0:
            sample_size = len(dataset)
        logger.info("🚀 Starting quick feature extraction pipeline...")
        logger.info(f"📊 Dataset size: {len(dataset)}, Sample size: {sample_size}")
        
        try:
            features = self.extract_all_features(
                dataset=dataset,
                hyperparams=hyperparams,
                sample_size=sample_size,
                batch_size=batch_size,
                probe_steps=probe_steps,
                enable_dynamic_probes=True  # Enable dynamic feature probes
            )
            
            logger.info(f"✅ Quick pipeline completed, extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Quick pipeline execution failed: {str(e)}")
            # Fallback not allowed, must retry
            logger.info("🔄 Quick pipeline failed, trying standard pipeline...")
            try:
                features = self.run_feature_extraction_pipeline(
                    dataset=dataset,
                    hyperparams=hyperparams,
                    static_sample_size=sample_size,
                    static_batch_size=batch_size,
                    dynamic_sample_size=sample_size,
                    dynamic_batch_size=batch_size,
                    probe_steps=probe_steps
                )
                logger.info(f"✅ Standard pipeline succeeded, extracted {len(features)} features")
                return features
            except Exception as e2:
                logger.error(f"Standard pipeline also failed: {str(e2)}")
                # Try most basic feature extraction
                try:
                    logger.info("🔄 Trying most basic feature extraction...")
                    features = self.static_extractor.extract_all_static_features(
                        dataset, 
                        sample_size=1, 
                        batch_size=1
                    )
                    logger.info(f"✅ Basic feature extraction succeeded, extracted {len(features)} features")
                    return features
                except Exception as e3:
                    logger.error(f"All feature extraction methods failed: {str(e3)}")
                    raise RuntimeError(f"Unable to extract any features: {str(e3)}")
    
    def run_comprehensive_pipeline(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        static_sample_size: int = 200,
        static_batch_size: int = 16,
        dynamic_sample_size: int = 100,
        dynamic_batch_size: int = 16,
        probe_steps: int = 200
    ) -> Dict[str, float]:
        """Comprehensive feature extraction pipeline (for production environment)
        
        Args:
            dataset: Dataset
            hyperparams: Hyperparameters
            static_sample_size: Sample size for static feature extraction
            static_batch_size: Batch size for static feature extraction
            dynamic_sample_size: Sample size for dynamic feature extraction
            dynamic_batch_size: Batch size for dynamic feature extraction
            probe_steps: Probe steps
        
        Returns:
            Dictionary containing all features
        """
        logger.info("🚀 Starting comprehensive feature extraction pipeline...")
        
        return self.extract_all_features(
            dataset=dataset,
            hyperparams=hyperparams,
            sample_size=max(static_sample_size, dynamic_sample_size),
            batch_size=max(static_batch_size, dynamic_batch_size),
            probe_steps=probe_steps,
            enable_dynamic_probes=True  # Enable dynamic feature probes
        )

    def run_feature_extraction_pipeline(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        static_sample_size: int = 100,
        static_batch_size: int = 8,
        dynamic_sample_size: int = 50,
        dynamic_batch_size: int = 8,
        probe_steps: int = 100
    ) -> Dict[str, float]:
        """Run complete feature extraction pipeline
        
        Execution order:
        1. Static feature extraction (does not change model parameters)
        2. Dynamic feature extraction (uses LoRA, does not affect original model)
        3. Merge all features
        
        Args:
            dataset: Dataset
            hyperparams: Hyperparameters
            static_sample_size: Sample size for static feature extraction
            static_batch_size: Batch size for static feature extraction
            dynamic_sample_size: Sample size for dynamic feature extraction
            dynamic_batch_size: Batch size for dynamic feature extraction
            probe_steps: Probe steps
            
        Returns:
            Dictionary containing all features
        """
        logger.info("🚀 Starting feature extraction pipeline...")
        
        # Directly call unified feature extraction method
        return self.extract_all_features(
            dataset=dataset,
            hyperparams=hyperparams,
            sample_size=max(static_sample_size, dynamic_sample_size),
            batch_size=max(static_batch_size, dynamic_batch_size),
            probe_steps=probe_steps,
            enable_dynamic_probes=True  # Enable dynamic feature probes
        )
    
    def get_feature_summary(self, features: Dict[str, float]) -> Dict[str, any]:
        """Get feature summary information
        
        Args:
            features: Feature dictionary
        
        Returns:
            Feature summary dictionary
        """
        if not features:
            return {"error": "Feature dictionary is empty"}
        
        # Group by feature type
        feature_categories = {
            "Text Statistical Features": [
                "avg_input_length", "avg_output_length", "io_length_ratio",
                "input_length_std", "output_length_std", "input_ttr", "output_ttr",
                "output_ngram_repetition", "vocab_complexity", "special_char_ratio"
            ],
            "Semantic Features": [
                "answer_groundedness", "embedding_outlier_ratio", "approximate_duplicates",
                "semantic_diversity", "io_similarity", "semantic_consistency"
            ],
            "Perplexity Features": [
                "reference_perplexity", "base_model_perplexity", "perplexity_change_rate",
                "reference_perplexity_std", "base_perplexity_std"
            ],
            "Dynamic Features": [
                "initial_loss", "loss_decay_rate", "loss_stability", "avg_grad_norm",
                "gradient_consistency", "avg_grad_sparsity", "avg_param_change",
                "landscape_flatness", "catastrophic_forgetting", "avg_activation_sparsity"
            ]
        }
        
        summary = {
            "total_features": len(features),
            "feature_categories": {}
        }
        
        # Count features by category
        for category, feature_names in feature_categories.items():
            category_features = {k: v for k, v in features.items() if k in feature_names}
            summary["feature_categories"][category] = {
                "count": len(category_features),
                "features": list(category_features.keys())
            }
        
        return summary
    
    def save_features_to_file(self, features: Dict[str, float], filepath: str) -> bool:
        """Save features to file
        
        Args:
            features: Feature dictionary
            filepath: File path
        
        Returns:
            Whether save was successful
        """
        try:
            # Convert to serializable format
            serializable_features = self._convert_to_serializable(features)
            
            # Debug: check converted features
            logger.info(f"Converted feature type check:")
            for key, value in serializable_features.items():
                logger.info(f"  {key}: {type(value)} = {value}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_features, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Features saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features to file: {str(e)}")
            return False
    
    def _convert_to_serializable(self, obj):
        """Convert object to serializable format"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(v) for v in obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item') and str(type(obj)).startswith("<class 'torch."):
            # Handle torch numeric types
            try:
                return obj.item()
            except:
                return str(obj)
        elif str(type(obj)).startswith("<class 'torch."):
            # More lenient torch type detection
            try:
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'detach'):
                    return obj.detach().cpu().item()
                else:
                    return str(obj)
            except:
                return str(obj)
        elif isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # For complex objects, convert to string
        else:
            return obj 