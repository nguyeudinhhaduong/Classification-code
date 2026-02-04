"""
Language Detector Module - Stage 1 of Pipeline
Uses LLM (Qwen) to detect programming language of code snippets.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects programming language using Qwen LLM.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Language Detector.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model_name = config['models']['language_detector']['model_name']
        self.max_length = config['models']['language_detector']['max_length']
        self.temperature = config['models']['language_detector']['temperature']
        self.max_new_tokens = config['models']['language_detector']['max_new_tokens']
        self.supported_languages = config['models']['language_detector']['supported_languages']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Language Detector initialized with model: {self.model_name}")
    
    def load_model(self):
        """Load the Qwen model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Model loaded successfully")
    
    def _create_prompt(self, code_snippet: str) -> List[Dict]:
        """
        Create chat prompt for language detection.
        
        Args:
            code_snippet: The code to classify
            
        Returns:
            List of message dictionaries for chat template
        """
        target_labels = str(self.supported_languages)
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert code classification engine. 
Strict Rules:
1. Classify the code into EXACTLY ONE of these categories: {target_labels}.
2. Distinguish carefully between 'C' and 'C++'. Look for '<iostream>', 'class', 'namespace', or 'template' (indicates C++) versus '<stdio.h>', 'malloc', 'struct' without methods (indicates C).
3. Distinguish 'Java' from 'C#'. Look for 'System.out.println' (Java) vs 'Console.WriteLine' (C#).
4. Output ONLY the language name from the list. No explanations."""
            },
            {
                "role": "user",
                "content": f"Code snippet:\n\n{code_snippet[:self.max_length]}\n\nIdentify the language:"
            }
        ]
        
        return messages
    
    def detect_single(self, code_snippet: str) -> str:
        """
        Detect language of a single code snippet.
        
        Args:
            code_snippet: The code to classify
            
        Returns:
            Detected language name
        """
        messages = self._create_prompt(code_snippet)
        
        # Create input for model
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature
            )
        
        # Extract only the generated text
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean the response
        clean_lang = response.strip().replace("'", "").replace('"', '')
        
        return clean_lang
    
    def detect_batch(self, code_snippets: List[str], show_progress: bool = True) -> List[str]:
        """
        Detect languages for a batch of code snippets.
        
        Args:
            code_snippets: List of code snippets to classify
            show_progress: Whether to show progress bar
            
        Returns:
            List of detected language names
        """
        if self.model is None:
            self.load_model()
        
        results = []
        iterator = tqdm(code_snippets, desc="Detecting Languages") if show_progress else code_snippets
        
        for code in iterator:
            try:
                lang = self.detect_single(code)
                results.append(lang)
            except Exception as e:
                logger.error(f"Error detecting language: {e}")
                results.append("Unknown")
        
        return results
    
    def get_language_distribution(self, predictions: List[str]) -> Dict[str, int]:
        """
        Get distribution of predicted languages.
        
        Args:
            predictions: List of predicted languages
            
        Returns:
            Dictionary with language counts
        """
        from collections import Counter
        return dict(Counter(predictions))
