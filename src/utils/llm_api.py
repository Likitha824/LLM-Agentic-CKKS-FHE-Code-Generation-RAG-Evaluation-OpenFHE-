import os
import json
import requests
import time
from typing import Dict, List, Any, Optional

import openai

class LLMInterface:
    """Base class for LLM interfaces."""
    
    def __init__(self):
        self.model_name = "base"
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text from prompt."""
        raise NotImplementedError("Subclasses must implement this method")


class DeepSeekInterface(LLMInterface):
    """Interface for DeepSeek Coder."""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            # Ensure a pad token is set.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype=dtype
            )
            
            if device == "cuda":
                self.model = self.model.cuda()
            
            print(f"Model loaded on {device}")
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 5000) -> str:
        """Generate code using DeepSeek Coder."""
        try:
            import torch
            self._load_model()
            
            # Format as a chat message with appropriate role.
            print(f"Generating with {self.model_name}, temp={temperature}, max_tokens={max_tokens}")
            
            # Prepare input tokens - use the direct approach instead
            text = f"<|user|>\n{prompt}\n<|assistant|>"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated tokens while excluding the prompt tokens.
            prompt_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            import traceback; traceback.print_exc()
            return f"Error: {str(e)}"


class LlamaInterface(LLMInterface):
    """Interface for Llama models using the pipeline API."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.pipe = None
        
    def _load_model(self):
        """Load the model if not already loaded."""
        if self.pipe is None:
            import torch
            from transformers import pipeline
            
            print(f"Loading {self.model_name}...")
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("Model loaded")
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate code using Llama."""
        try:
            self._load_model()
            
            # Format the input as a chat with a system message.
            messages = [
                {"role": "system", "content": "You are an expert C++ programmer specializing in OpenFHE and CKKS encryption."},
                {"role": "user", "content": prompt}
            ]
            
            print(f"Generating with {self.model_name}, temp={temperature}, max_tokens={max_tokens}")
            outputs = self.pipe(
                messages,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_k=50,
                top_p=0.95,
            )
            
            # Attempt to extract the assistant's response.
            try:
                response = outputs[0]["generated_text"][-1]["content"]
            except (KeyError, IndexError, TypeError):
                full_text = outputs[0]["generated_text"]
                import re
                patterns = [
                    r'(?:assistant|Assistant):\s*(.*)',
                    r'\[/INST\]\s*(.*)',
                    r'<\/s>\s*(.*)'
                ]
                response = None
                for pattern in patterns:
                    match = re.search(pattern, full_text, re.DOTALL)
                    if match:
                        response = match.group(1).strip()
                        break
                if response is None:
                    response = full_text
            
            return response
            
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            import traceback; traceback.print_exc()
            return f"Error: {str(e)}"


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI GPT-3.5 Turbo using the ChatCompletion API."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
        # Set API key from environment
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using OpenAI's GPT-3.5 Turbo."""
        try:
            # Prepare the chat messages
            messages = [{"role": "user", "content": prompt}]
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Extract the assistant's reply
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating with OpenAI {self.model_name}: {e}")
            import traceback; traceback.print_exc()
            return f"Error: {str(e)}"


def get_llm_interface(model_name: str) -> LLMInterface:
    """Factory function to get the appropriate LLM interface."""
    name = model_name.lower()
    if name == "deepseek":
        return DeepSeekInterface("deepseek-ai/deepseek-coder-6.7b-instruct")
    elif name == "llama":
        return LlamaInterface("meta-llama/Llama-3.2-3B-Instruct")
    elif name in ["gpt-3.5-turbo", "gpt3.5-turbo", "openai"]:
        return OpenAIInterface("gpt-3.5-turbo")
    elif "deepseek" in name:
        return DeepSeekInterface(model_name)
    elif "llama" in name or "meta-llama" in name:
        return LlamaInterface(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
