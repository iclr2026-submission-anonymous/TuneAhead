"""
Static feature extraction module

Responsible for extracting static features that do not require model training, including:
- Text statistical features (length, TTR, n-gram, etc.)
- Semantic features (based on pre-trained model embeddings)
- Perplexity features
- New additions: special character ratio, instruction complexity, vocabulary alignment, answer traceability, embedding space outlier ratio

Supports final format data: context_text + qa_pairs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.ensemble import IsolationForest
import re
import logging
import string
import math
import gc

logger = logging.getLogger(__name__)

class StaticFeatureExtractor:
    """Static feature extractor
    
    Directly receives model and tokenizer, supports final format data
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None
    ):
        """Initialize static feature extractor
        
        Args:
            model: Pre-trained model
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
            try:
                if hasattr(self.model, 'parameters') and next(self.model.parameters()).device != torch.device(self.device):
                    self.model = self.model.to(self.device)
            except (AttributeError, StopIteration):
                # If model has no parameters or cannot determine device, skip device check
                pass
    
    def calculate_special_char_ratio(self, text: str) -> float:
        """Calculate special character/punctuation ratio
        
        Adapted for English plain text data, mainly detects punctuation and special characters
        
        Args:
            text: Input text
            
        Returns:
            Special character ratio
        """
        if not text:
            return 0.0
        
        # English punctuation and special characters
        punctuation_chars = set('.,!?;:()[]{}""\'\'-–—…')
        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?/~`"\'\\')
        
        # Number detection (may indicate data quality issues)
        digit_pattern = r'\d+'
        digits = len(re.findall(digit_pattern, text))
        
        # Repeated character detection (like "aaa", "...")
        repeat_pattern = r'(.)\1{2,}'  # 3 or more consecutive identical characters
        repeats = len(re.findall(repeat_pattern, text))
        
        # All uppercase word detection (may indicate titles or emphasis)
        uppercase_pattern = r'\b[A-Z]{2,}\b'
        uppercase_words = len(re.findall(uppercase_pattern, text))
        
        # Abnormal space detection (multiple consecutive spaces)
        space_pattern = r' {2,}'
        extra_spaces = len(re.findall(space_pattern, text))
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Calculate various special characters
        punct_count = sum(1 for char in text if char in punctuation_chars)
        special_count = sum(1 for char in text if char in special_chars)
        
        # Comprehensive special character ratio (suitable for ordinary English text)
        total_special = (punct_count + 
                        special_count + 
                        digits * 2 +  # Number weight
                        repeats * 3 +  # Repeated character weight
                        uppercase_words * 1 +  # All uppercase word weight
                        extra_spaces * 2)  # Abnormal space weight
        
        return total_special / total_chars
    
    def calculate_answer_groundedness(self, context: str, answer: str) -> float:
        """Calculate answer groundedness
        
        Calculate the proportion of n-grams in the answer that appear in the context
        Adapted for English text characteristics, including stemming and synonym detection
        
        Args:
            context: Context
            answer: Answer
            
        Returns:
            Groundedness score
        """
        if not context or not answer:
            return 0.0
        
        # English text preprocessing
        def preprocess_text(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation (keep spaces)
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra spaces
            text = ' '.join(text.split())
            return text
        
        context_processed = preprocess_text(context)
        answer_processed = preprocess_text(answer)
        
        # Extract words from answer
        answer_words = answer_processed.split()
        context_words = context_processed.split()
        
        if len(answer_words) < 2:
            return 0.0
        
        # Calculate 2-gram and 3-gram
        answer_bigrams = [' '.join(answer_words[i:i+2]) for i in range(len(answer_words)-1)]
        answer_trigrams = [' '.join(answer_words[i:i+3]) for i in range(len(answer_words)-2)]
        
        context_bigrams = [' '.join(context_words[i:i+2]) for i in range(len(context_words)-1)]
        context_trigrams = [' '.join(context_words[i:i+3]) for i in range(len(context_words)-2)]
        
        # Calculate matching ratio
        bigram_matches = sum(1 for bg in answer_bigrams if bg in context_bigrams)
        trigram_matches = sum(1 for tg in answer_trigrams if tg in context_trigrams)
        
        # English word matching (considering stemming)
        def get_word_stem(word):
            # Simple stemming (remove common suffixes)
            suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break
            return word
        
        # Calculate stem matching
        answer_stems = [get_word_stem(word) for word in answer_words]
        context_stems = [get_word_stem(word) for word in context_words]
        
        stem_matches = sum(1 for stem in answer_stems if stem in context_stems)
        
        # Calculate comprehensive groundedness score
        total_ngrams = len(answer_bigrams) + len(answer_trigrams)
        if total_ngrams == 0:
            return 0.0
        
        # Weighted calculation: n-gram matching + stem matching
        ngram_groundedness = (bigram_matches + trigram_matches) / total_ngrams
        stem_groundedness = stem_matches / len(answer_words) if answer_words else 0
        
        # Comprehensive score (n-gram weight 0.7, stem weight 0.3)
        groundedness = ngram_groundedness * 0.7 + stem_groundedness * 0.3
        
        return groundedness
    
    def calculate_embedding_outliers(self, embeddings: List[torch.Tensor]) -> float:
        """Calculate embedding space outlier ratio
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Outlier ratio
        """
        if not embeddings or len(embeddings) < 2:
            return 0.0
        
        try:
            # Convert to numpy array, handle BF16 type
            if isinstance(embeddings, list):
                # First convert to float32, then to numpy
                emb_tensor = torch.stack(embeddings)
                if emb_tensor.dtype == torch.bfloat16:
                    emb_tensor = emb_tensor.float()  # Convert to float32
                emb_array = emb_tensor.cpu().numpy()
            else:
                if embeddings.dtype == torch.bfloat16:
                    embeddings = embeddings.float()  # Convert to float32
                emb_array = embeddings.cpu().numpy()
            
            # Ensure data is 2D array
            if emb_array.ndim == 1:
                emb_array = emb_array.reshape(-1, 1)
            
            # Use Isolation Forest to detect outliers
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(emb_array)
            
            # Calculate outlier ratio (-1 indicates outlier)
            outlier_ratio = np.sum(predictions == -1) / len(predictions)
            
            return outlier_ratio
            
        except Exception as e:
            logger.error(f"Error occurred while calculating embedding space outliers: {str(e)}")
            # Try simpler method: distance-based outlier detection
            try:
                if isinstance(embeddings, list):
                    emb_tensor = torch.stack(embeddings)
                else:
                    emb_tensor = embeddings
                
                # Calculate mean of all embedding vectors
                mean_embedding = torch.mean(emb_tensor, dim=0)
                
                # Calculate distance from each embedding vector to mean
                distances = torch.norm(emb_tensor - mean_embedding, dim=1)
                
                # Use 3 times standard deviation as outlier threshold
                threshold = torch.mean(distances) + 3 * torch.std(distances)
                outlier_count = torch.sum(distances > threshold)
                
                return outlier_count.item() / len(distances)
                
            except Exception as e2:
                logger.error(f"Backup outlier detection method also failed: {str(e2)}")
                # Last resort: simple detection based on embedding vector norms
                try:
                    norms = torch.norm(emb_tensor, dim=1)
                    mean_norm = torch.mean(norms)
                    std_norm = torch.std(norms)
                    threshold = mean_norm + 2 * std_norm
                    outlier_count = torch.sum(norms > threshold)
                    return outlier_count.item() / len(norms)
                except Exception as e3:
                    logger.error(f"All outlier detection methods failed: {str(e3)}")
                    raise RuntimeError(f"Unable to calculate embedding space outliers: {str(e3)}")

    def is_final_format(self, dataset: List[Dict]) -> bool:
        """Determine if it's final format data"""
        if not dataset:
            return False
        sample = dataset[0]
        
        # Check if it's final format (contains context_text and qa_pairs)
        if "context_text" in sample and "qa_pairs" in sample:
            return True
        
        # Check if it's standard conversation format (contains messages)
        if "messages" in sample:
            return False  # This is standard format, not final format
        
        # Check if it's simple QA pair format
        if "input" in sample and "output" in sample:
            return False  # This is standard format, not final format
        
        # Check if it's Qwen2.5 format
        if "conversations" in sample:
            return False  # This is standard format, not final format
        
        # Default to standard format
        return False
    
    def extract_qa_from_final_format(self, item: Dict) -> List[Tuple[str, str]]:
        """Extract all QA pairs from final format data
        
        Args:
            item: Final format data item {"context_text": str, "qa_pairs": List[Dict]}
            
        Returns:
            List[Tuple[str, str]]: QA pair list, each element is (question, answer)
        """
        context = item.get("context_text", "")
        qa_pairs = item.get("qa_pairs", [])
        
        result = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("output", "")
            
            if question and answer:
                # Combine context and question
                full_question = f"{context}\n\n{question}" if context else question
                result.append((full_question, answer))
        
        return result
    
    def format_qa_pair(self, question: str, answer: str) -> str:
        """Format QA pair as model input format
        
        Args:
            question: Question
            answer: Answer
            
        Returns:
            Formatted text
        """
        return f"Question: {question}\nAnswer: {answer}"
    
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
    
    def calculate_ttr(self, text: str) -> float:
        """Calculate Type-Token Ratio (TTR) of text
        
        Adapted for English text characteristics, including stemming and stop word filtering
        
        Args:
            text: Input text
            
        Returns:
            TTR value
        """
        if not text:
            return 0.0
        
        # English stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # English text preprocessing
        def preprocess_text(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation (keep letters and numbers)
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra spaces
            text = ' '.join(text.split())
            return text
        
        # Simple stemming
        def get_word_stem(word):
            # Remove common suffixes
            suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es', 'tion', 'sion']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break
            return word
        
        # Preprocess text
        processed_text = preprocess_text(text)
        words = processed_text.split()
        
        if not words:
            return 0.0
        
        # Filter stop words and extract stems
        filtered_words = []
        for word in words:
            if word not in stop_words and len(word) > 1:  # Filter stop words and single character words
                stem = get_word_stem(word)
                filtered_words.append(stem)
        
        if not filtered_words:
            return 0.0
        
        # Calculate TTR
        unique_words = len(set(filtered_words))
        total_words = len(filtered_words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def calculate_ngram_repetition(self, text: str, n: int = 3) -> float:
        """Calculate n-gram repetition rate
        
        Adapted for English text characteristics, including stemming and stop word filtering
        
        Args:
            text: Input text
            n: n value for n-gram
            
        Returns:
            n-gram repetition rate
        """
        if not text:
            return 0.0
        
        # English stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # English text preprocessing
        def preprocess_text(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation (keep letters and numbers)
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra spaces
            text = ' '.join(text.split())
            return text
        
        # Simple stemming
        def get_word_stem(word):
            # Remove common suffixes
            suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es', 'tion', 'sion']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break
            return word
        
        # Preprocess text
        processed_text = preprocess_text(text)
        words = processed_text.split()
        
        if len(words) < n:
            return 0.0
        
        # Filter stop words and extract stems
        filtered_words = []
        for word in words:
            if word not in stop_words and len(word) > 1:  # Filter stop words and single character words
                stem = get_word_stem(word)
                filtered_words.append(stem)
        
        if len(filtered_words) < n:
            return 0.0
        
        # Generate n-grams
        ngrams = [tuple(filtered_words[i:i+n]) for i in range(len(filtered_words)-n+1)]
        if not ngrams:
            return 0.0
        
        # Calculate repeated n-gram ratio
        ngram_counter = Counter(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counter.values() if count > 1)
        
        return repeated_ngrams / len(ngram_counter) if ngram_counter else 0.0

    def extract_all_static_features(self, dataset: List[Dict], sample_size: int = 100, batch_size: int = 8, invalid_sample_policy: str = "skip") -> Dict[str, float]:
        """Extract all static features at once (optimized version: single inference)"""
        logger.info("Starting to extract all static features (optimized version: single inference)...")
        is_final_format = self.is_final_format(dataset)
        logger.info(f"Detected data format: {'Final format' if is_final_format else 'Standard format'}")
        sampled_data = self.sample_dataset(dataset, sample_size)
        
        # Collect all QA pairs
        qa_list = []
        contexts = []  # For answer groundedness calculation
        skipped_pairs = 0
        imputed_pairs = 0
        use_impute = invalid_sample_policy.lower() in ("impute_low", "impute_low_with_ppl_cap")
        if is_final_format:
            for item in sampled_data:
                try:
                    context = item.get("context_text", "")
                    pairs = self.extract_qa_from_final_format(item)
                    if pairs:
                        # Append corresponding context placeholder for each QA pair to maintain index alignment
                        contexts.extend([context] * len(pairs))
                        qa_list.extend(pairs)
                    else:
                        skipped_pairs += 1
                        if use_impute:
                            imputed_pairs += 1
                except Exception:
                    skipped_pairs += 1
                    if use_impute:
                        imputed_pairs += 1
        else:
            for item in sampled_data:
                try:
                    q, a = self._extract_qa_from_standard_format(item)
                    if not q or not a or not str(a).strip():
                        skipped_pairs += 1
                        if use_impute:
                            imputed_pairs += 1
                        continue
                    contexts.append("")  # Standard format has no context
                    qa_list.append((q, a))
                except Exception:
                    skipped_pairs += 1
                    if use_impute:
                        imputed_pairs += 1
        
        logger.info(f"Processed {len(qa_list)} QA pairs, skipped {skipped_pairs}; Low-score imputation strategy: {'Enabled' if use_impute else 'Disabled'}, will impute {imputed_pairs} pairs")

        # If no valid QA pairs: only return diagnostic fields, do not interrupt flow
        if len(qa_list) == 0:
            logger.warning("⚠️ Valid samples is 0, this static feature extraction will return NaN/empty values and not interrupt flow")
            return {
                "static_valid_count": 0,
                "static_skipped_count": skipped_pairs,
                "static_imputed_count": imputed_pairs,
                "static_coverage": 0.0,
            }
        
        # Text statistical features (fast calculation)
        input_texts = [q for q, a in qa_list]
        output_texts = [a for q, a in qa_list]
        combined_texts = [f"{q} {a}" for q, a in qa_list]
        
        # Fast calculation of text features
        logger.info("📊 Calculating text statistical features...")
        input_lengths = [len(self.tokenizer.encode(q, add_special_tokens=False)) for q in tqdm(input_texts, desc="Calculate input length")]
        output_lengths = [len(self.tokenizer.encode(a, add_special_tokens=False)) for a in tqdm(output_texts, desc="Calculate output length")]
        input_ttrs = [self.calculate_ttr(q) for q in tqdm(input_texts, desc="Calculate input TTR")]
        output_ttrs = [self.calculate_ttr(a) for a in tqdm(output_texts, desc="Calculate output TTR")]
        output_ngram_reps = [self.calculate_ngram_repetition(a, n=3) for a in tqdm(output_texts, desc="Calculate n-gram repetition rate")]
        special_char_ratios = [self.calculate_special_char_ratio(f"{q} {a}") for q, a in tqdm(qa_list, desc="Calculate special character ratio")]
        
        # Fast calculation of vocabulary complexity
        logger.info("📈 Calculating vocabulary complexity...")
        vocab_complexity = []
        for combined_text in tqdm(combined_texts, desc="Calculate vocabulary complexity"):
            words = re.findall(r'\b\w+\b', combined_text.lower())
            long_words = sum(1 for word in words if len(word) > 8)
            complexity = long_words / len(words) if words else 0
            vocab_complexity.append(complexity)
        
        # Fast calculation of answer groundedness
        logger.info("🔍 Calculating answer groundedness...")
        groundedness_scores = []
        for i, (q, a) in enumerate(tqdm(qa_list, desc="Calculate answer groundedness")):
            context = contexts[i] if i < len(contexts) else ""
            groundedness_scores.append(self.calculate_answer_groundedness(context, a))
        
        # 🚀 Optimization: two-stage independent execution to reduce memory peak
        # Stage 1: Only calculate perplexity (no hidden states, low memory usage)
        logger.info("🤖 Stage 1: Calculating perplexity features...")
        all_perplexities = []

        def _build_chat_inputs_and_labels(pairs: List[Tuple[str, str]], max_len: int = 256):
            input_ids_list = []
            attn_list = []
            labels_list = []
            for q, a in pairs:
                try:
                    messages_full = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                    messages_prompt = [{"role": "user", "content": q}]
                    full_ids = self.tokenizer.apply_chat_template(
                        messages_full, tokenize=True, truncation=True, max_length=max_len
                    )
                    prompt_ids = self.tokenizer.apply_chat_template(
                        messages_prompt, tokenize=True, add_generation_prompt=True,
                        truncation=True, max_length=max_len
                    )
                    if isinstance(full_ids, list):
                        full = torch.tensor(full_ids, dtype=torch.long)
                        prompt_len = len(prompt_ids) if isinstance(prompt_ids, list) else int(prompt_ids.shape[-1])
                    else:
                        full = full_ids.to(torch.long).view(-1)
                        prompt_len = full.shape[0] // 2  # Fallback
                    labels = full.clone()
                    if prompt_len > 0:
                        labels[:prompt_len] = -100
                    input_ids_list.append(full)
                    attn_list.append(torch.ones_like(full))
                    labels_list.append(labels)
                except Exception:
                    merged = self.format_qa_pair(q, a)
                    enc = self.tokenizer(merged, return_tensors=None, truncation=True, max_length=max_len)
                    q_enc = self.tokenizer(q, return_tensors=None, truncation=True, max_length=max_len)
                    if isinstance(enc, dict):
                        full = torch.tensor(enc["input_ids"], dtype=torch.long)[0]
                    else:
                        full = torch.tensor(enc, dtype=torch.long)
                    q_len = len(q_enc["input_ids"][0]) if isinstance(q_enc, dict) else len(q_enc)
                    labels = full.clone()
                    labels[:min(q_len, labels.numel())] = -100
                    input_ids_list.append(full)
                    attn_list.append(torch.ones_like(full))
                    labels_list.append(labels)

            if not input_ids_list:
                return None, None, None

            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            max_len_batch = max(t.numel() for t in input_ids_list)
            def pad_to(t: torch.Tensor, L: int, val: int):
                if t.numel() >= L:
                    return t[:L]
                return torch.cat([t, torch.full((L - t.numel(),), val, dtype=t.dtype)], dim=0)

            input_ids = torch.stack([pad_to(t, max_len_batch, pad_id) for t in input_ids_list], dim=0)
            attention = torch.stack([pad_to(t, max_len_batch, 0) for t in attn_list], dim=0)
            labels = torch.stack([pad_to(t, max_len_batch, -100) for t in labels_list], dim=0)
            return input_ids, attention, labels

        with torch.no_grad():
            for i in tqdm(range(0, len(qa_list), batch_size), desc="Stage 1: Perplexity inference"):
                batch_qa = qa_list[i:i+batch_size]
                input_ids, attention_mask, labels = _build_chat_inputs_and_labels(batch_qa, max_len=256)
                if input_ids is None:
                    all_perplexities.extend([float('nan')] * len(batch_qa))
                    continue
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Calculate perplexity (ignore masked tokens)
                try:
                    logits = outputs.logits.float()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                    token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    token_loss = token_loss.view(shift_labels.size())
                    valid = (shift_labels != -100).float()
                    seq_loss = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                    seq_loss = torch.clamp(seq_loss, min=-5.0, max=20.0)
                    batch_ppl = torch.exp(seq_loss).cpu().tolist()
                    all_perplexities.extend(batch_ppl)
                except Exception as e:
                    logger.warning(f"Perplexity calculation failed (batch): {str(e)}")
                    all_perplexities.extend([float('nan')] * input_ids.size(0))
                
                # Clean up intermediate variables for current batch
                del outputs, input_ids, attention_mask, labels
            
            # Stage 1 completed, clean up memory
            logger.info("🧹 Stage 1 completed, cleaning up memory...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Stage 2: Only calculate embedding-related features (with hidden states, no labels, medium memory usage)
        logger.info("🤖 Stage 2: Calculating embedding-related features...")
        all_embeddings = []
        
        def _build_chat_inputs_only(pairs: List[Tuple[str, str]], max_len: int = 256):
            input_ids_list = []
            attn_list = []
            for q, a in pairs:
                try:
                    messages_full = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                    full_ids = self.tokenizer.apply_chat_template(
                        messages_full, tokenize=True, truncation=True, max_length=max_len
                    )
                    if isinstance(full_ids, list):
                        full = torch.tensor(full_ids, dtype=torch.long)
                    else:
                        full = full_ids.to(torch.long).view(-1)
                    input_ids_list.append(full)
                    attn_list.append(torch.ones_like(full))
                except Exception:
                    merged = self.format_qa_pair(q, a)
                    enc = self.tokenizer(merged, return_tensors=None, truncation=True, max_length=max_len)
                    if isinstance(enc, dict):
                        full = torch.tensor(enc["input_ids"], dtype=torch.long)[0]
                    else:
                        full = torch.tensor(enc, dtype=torch.long)
                    input_ids_list.append(full)
                    attn_list.append(torch.ones_like(full))

            if not input_ids_list:
                return None, None

            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            max_len_batch = max(t.numel() for t in input_ids_list)
            def pad_to(t: torch.Tensor, L: int, val: int):
                if t.numel() >= L:
                    return t[:L]
                return torch.cat([t, torch.full((L - t.numel(),), val, dtype=t.dtype)], dim=0)

            input_ids = torch.stack([pad_to(t, max_len_batch, pad_id) for t in input_ids_list], dim=0)
            attention = torch.stack([pad_to(t, max_len_batch, 0) for t in attn_list], dim=0)
            return input_ids, attention

        with torch.no_grad():
            for i in tqdm(range(0, len(qa_list), batch_size), desc="Stage 2: Embedding inference"):
                batch_qa = qa_list[i:i+batch_size]
                input_ids, attention_mask = _build_chat_inputs_only(batch_qa, max_len=256)
                if input_ids is None:
                    continue
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Extract embedding (from hidden_states)
                hidden = outputs.hidden_states[-1]
                # Take first token embedding and immediately move to CPU to avoid GPU accumulation
                batch_embedding = hidden[:, 0, :].detach().float().cpu()  # [batch_size, hidden_size]
                # Split into single samples and append to avoid large tensors retaining views in list
                for j in range(batch_embedding.size(0)):
                    all_embeddings.append(batch_embedding[j].clone())
                
                # Clean up intermediate variables for current batch
                del outputs, input_ids, attention_mask, hidden
            
            # Stage 2 completed, clean up memory
            logger.info("🧹 Stage 2 completed, cleaning up memory...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Calculate semantic similarity
        logger.info("🔍 Calculating semantic similarity...")
        similarities = []
        for i in tqdm(range(len(all_embeddings)), desc="Calculate similarity"):
            if i < len(all_embeddings):
                embedding = all_embeddings[i]
                # Ensure embedding is 2D tensor [1, hidden_size]
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
                
                # Calculate similarity with itself (should be 1.0)
                try:
                    # Ensure embedding is 2D tensor with correct dimensions
                    if embedding.dim() == 1:
                        embedding = embedding.unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
                    elif embedding.dim() > 2:
                        embedding = embedding.squeeze()
                        if embedding.dim() == 1:
                            embedding = embedding.unsqueeze(0)
                    
                    # Ensure correct dimensions before calculating similarity
                    if embedding.dim() == 2 and embedding.size(0) == 1:
                        # Calculate cosine similarity with itself
                        sim = F.cosine_similarity(embedding, embedding, dim=1)
                        similarities.append(sim.cpu().item())
                    else:
                        # If dimensions are still incorrect, calculate vector norm as similarity metric
                        norm = torch.norm(embedding)
                        similarities.append(norm.cpu().item())
                        
                except Exception as e:
                    logger.error(f"Error occurred while calculating similarity: {str(e)}")
                    # Final fallback: calculate vector norm
                    try:
                        norm = torch.norm(embedding)
                        similarities.append(norm.cpu().item())
                    except Exception as e2:
                        logger.error(f"Vector norm calculation also failed: {str(e2)}")
                        # Use fixed value 1.0 as final fallback
                        similarities.append(1.0)
            else:
                # Should not reach here, if reached indicates logic problem
                raise RuntimeError(f"Index {i} out of range, embeddings length is {len(all_embeddings)}")
        
        # Calculate embedding space outlier ratio
        logger.info("📊 Calculating embedding space outliers...")
        embedding_outlier_ratio = self.calculate_embedding_outliers(all_embeddings)
        
        # Calculate approximate duplicate samples (simplified version)
        logger.info("🔄 Calculating approximate duplicate samples...")
        vectorizer = TfidfVectorizer(max_features=1000)  # Limit features to improve speed
        try:
            tfidf_matrix = vectorizer.fit_transform(combined_texts)
            similarity_matrix = sklearn_cosine_similarity(tfidf_matrix)
            np.fill_diagonal(similarity_matrix, 0)
            duplicate_pairs = np.sum(similarity_matrix > 0.8)
            approximate_duplicates = duplicate_pairs / (len(combined_texts) * (len(combined_texts) - 1)) if len(combined_texts) > 1 else 0
        except Exception as e:
            logger.error(f"Error occurred while calculating approximate duplicate samples: {str(e)}")
            # Try simpler method to calculate duplicates
            try:
                # Use simple string similarity
                logger.info("🔄 Trying string similarity to calculate duplicates...")
                duplicate_count = 0
                total_pairs = 0
                
                for i in range(len(combined_texts)):
                    for j in range(i + 1, len(combined_texts)):
                        text1 = combined_texts[i]
                        text2 = combined_texts[j]
                        
                        # Calculate simple character overlap rate
                        if len(text1) > 0 and len(text2) > 0:
                            common_chars = set(text1.lower()) & set(text2.lower())
                            total_chars = set(text1.lower()) | set(text2.lower())
                            similarity = len(common_chars) / len(total_chars) if total_chars else 0
                            
                            if similarity > 0.8:  # 80% similarity threshold
                                duplicate_count += 1
                            total_pairs += 1
                
                approximate_duplicates = duplicate_count / total_pairs if total_pairs > 0 else 0
                logger.info(f"✅ String similarity method succeeded, duplicate rate: {approximate_duplicates:.4f}")
                
            except Exception as e2:
                logger.error(f"String similarity method also failed: {str(e2)}")
                # Last resort: simple duplicate detection based on text length
                try:
                    logger.info("🔄 Trying duplicate detection based on text length...")
                    text_lengths = [len(text) for text in combined_texts]
                    length_counts = {}
                    
                    for length in text_lengths:
                        length_counts[length] = length_counts.get(length, 0) + 1
                    
                    # Calculate length duplicate ratio
                    duplicate_lengths = sum(1 for count in length_counts.values() if count > 1)
                    total_unique_lengths = len(length_counts)
                    
                    approximate_duplicates = duplicate_lengths / total_unique_lengths if total_unique_lengths > 0 else 0
                    logger.info(f"✅ Length duplicate detection succeeded, duplicate rate: {approximate_duplicates:.4f}")
                    
                except Exception as e3:
                    logger.error(f"All duplicate detection methods failed: {str(e3)}")
                    raise RuntimeError(f"Unable to calculate approximate duplicate samples: {str(e3)}")
        
        # ===== Calculate all statistical features =====
        logger.info("📈 Calculating final statistical features...")
        features = {}
        
        # 1. Basic statistical features
        features.update({
            "avg_input_length": np.mean(input_lengths),
            "avg_output_length": np.mean(output_lengths),
            "io_length_ratio": np.mean(output_lengths) / np.mean(input_lengths) if np.mean(input_lengths) > 0 else 0,
            "input_length_std": np.std(input_lengths),
            "output_length_std": np.std(output_lengths),
            "input_ttr": np.mean(input_ttrs),
            "output_ttr": np.mean(output_ttrs),
            "output_ngram_repetition": np.mean(output_ngram_reps),
            "approximate_duplicates": approximate_duplicates,
            "vocab_complexity": np.mean(vocab_complexity),
            "special_char_ratio": np.mean(special_char_ratios),
            "answer_groundedness": np.mean(groundedness_scores),
            "embedding_outlier_ratio": embedding_outlier_ratio,
            "static_valid_count": len(qa_list),
            "static_skipped_count": skipped_pairs,
            "static_imputed_count": imputed_pairs,
            "static_coverage": (len(qa_list) / (len(qa_list) + skipped_pairs)) if (len(qa_list) + skipped_pairs) > 0 else 0.0,
        })
        
        # 2. Perplexity features (failure is not fatal, skip directly)
        valid_perplexities = [p for p in all_perplexities if not math.isnan(p)]
        if valid_perplexities:
            avg_perplexity = np.mean(valid_perplexities)
            features.update({
                "reference_perplexity": avg_perplexity,
                "base_model_perplexity": avg_perplexity,  # Use real value
                "perplexity_change_rate": 0.0,  # Static features cannot calculate change rate
                "reference_perplexity_std": np.std(valid_perplexities),
                "base_perplexity_std": np.std(valid_perplexities)
            })
        else:
            logger.warning("⚠️ No valid perplexity values, skipping perplexity-related features this time")
        
        # 3. Semantic features (must calculate real values)
        if similarities:
            features.update({
                "semantic_diversity": np.mean(similarities),
                "io_similarity": np.mean(similarities),
                "semantic_consistency": np.std(similarities)  # Use standard deviation as consistency metric
            })
        else:
            raise RuntimeError("Unable to calculate semantic features: similarity list is empty")
        
        logger.info(f"✅ Optimized feature extraction completed, {len(features)} features total")
        return features
    
    def _extract_qa_from_standard_format(self, item: Dict) -> Tuple[str, str]:
        """Extract QA pairs from standard format data (supports multiple formats)"""
        if "messages" in item:
            # Process standard conversation format (like the dataset you provided)
            messages = item["messages"]
            question = ""
            answer = ""
            
            for msg in messages:
                if msg["role"] == "user":
                    question = msg["content"]
                elif msg["role"] == "assistant":
                    answer = msg["content"]
            
            if not question or not answer:
                raise ValueError(f"Unable to extract complete QA pair from messages: {item}")
            
            return question, answer
            
        elif "conversations" in item:
            # Process Qwen2.5 format
            question = next(msg["content"] for msg in item["conversations"] 
                          if msg["role"] == "user")
            answer = next(msg["content"] for msg in item["conversations"] 
                        if msg["role"] == "assistant")
        elif "input" in item and "output" in item:
            # Process simple QA pair format
            question = item["input"]
            answer = item["output"]
        else:
            # Try other possible formats
            raise ValueError(f"Unrecognized data format: {list(item.keys())}")
        
        return question, answer
    
    def _process_single_qa(self, question: str, answer: str, input_lengths, output_lengths,
                          input_ttrs, output_ttrs, output_ngram_reps, texts, vocab_complexity,
                          reference_perplexities, base_perplexities, embeddings, similarities,
                          semantic_consistencies):
        """Process single QA pair feature extraction"""
        # ===== 1. Text statistical features =====
        # Calculate length features
        input_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        input_lengths.append(len(input_tokens))
        output_lengths.append(len(output_tokens))
        
        # Calculate TTR
        input_ttrs.append(self.calculate_ttr(question))
        output_ttrs.append(self.calculate_ttr(answer))
        
        # Calculate n-gram repetition rate
        ngram_rep_3 = self.calculate_ngram_repetition(answer, n=3)
        ngram_rep_4 = self.calculate_ngram_repetition(answer, n=4)
        output_ngram_reps.append((ngram_rep_3 + ngram_rep_4) / 2)
        
        # Collect text for calculating approximate duplicates
        texts.append(f"{question} {answer}")
        
        # Calculate vocabulary complexity (long word ratio)
        combined_text = f"{question} {answer}".lower()
        words = re.findall(r'\b\w+\b', combined_text)
        long_words = sum(1 for word in words if len(word) > 8)
        complexity = long_words / len(words) if words else 0
        vocab_complexity.append(complexity)
        
        # ===== 2. Perplexity features =====
        # Calculate reference model perplexity (for entire QA pair)
        formatted_text = self.format_qa_pair(question, answer)
        ref_inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        ref_loss = self.model(**ref_inputs, labels=ref_inputs["input_ids"]).loss
        ref_perplexity = torch.exp(ref_loss).item()
        reference_perplexities.append(ref_perplexity)
        
        # Calculate base model perplexity (only for answer)
        base_inputs = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        base_loss = self.model(**base_inputs, labels=base_inputs["input_ids"]).loss
        base_perplexity = torch.exp(base_loss).item()
        base_perplexities.append(base_perplexity)
        
        # ===== 3. Semantic features =====
        # Calculate semantic diversity (based on answer embedding)
        answer_outputs = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        answer_hidden = self.model(**answer_outputs, output_hidden_states=True).hidden_states[-1]
        answer_embedding = answer_hidden[:, 0, :].detach().float().cpu().mean(dim=0)
        embeddings.append(answer_embedding)
        
        # Calculate input-output similarity
        question_outputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        question_hidden = self.model(**question_outputs, output_hidden_states=True).hidden_states[-1]
        question_embedding = question_hidden[:, 0, :].detach().float().cpu().mean(dim=0)
        
        similarity = F.cosine_similarity(
            question_embedding.unsqueeze(0),
            answer_embedding.unsqueeze(0)
        ).item()
        similarities.append(similarity)
        
        # Calculate semantic consistency (based on answer internal consistency)
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) > 1:
            sentence_embeddings = []
            for sentence in sentences:
                if sentence.strip():
                    sent_outputs = self.tokenizer(
                        sentence.strip(),
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(self.device)
                    sent_hidden = self.model(**sent_outputs, output_hidden_states=True).hidden_states[-1]
                    sent_embedding = sent_hidden[:, 0, :].detach().float().cpu().mean(dim=0)
                    sentence_embeddings.append(sent_embedding)
            
            if len(sentence_embeddings) > 1:
                # Calculate average similarity between sentences
                sent_similarities = []
                for i in range(len(sentence_embeddings)):
                    for j in range(i + 1, len(sentence_embeddings)):
                        sim = F.cosine_similarity(
                            sentence_embeddings[i].unsqueeze(0),
                            sentence_embeddings[j].unsqueeze(0)
                        ).item()
                        sent_similarities.append(sim)
                semantic_consistencies.append(np.mean(sent_similarities))
            else:
                semantic_consistencies.append(1.0)  # Single sentence case
        else:
            semantic_consistencies.append(1.0)  # Single sentence case 