from abc import ABC, abstractmethod
from typing import Any, List
from captum._utils.typing import TokenizerLike
from openai import OpenAI
import os

class RemoteLLMProvider(ABC):
    """All remote LLM providers like vLLM extends this class that offer logprob APIs."""
    
    tokenizer: TokenizerLike
    api_url: str
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        **gen_args: Any
    ) -> str:
        """
        Generates text and returns the generated text.
        
        Args:
            prompt: The input prompt to generate from
            gen_args: Additional generation arguments
            
        Returns:
            The generated text.
        """
        ...
    
    @abstractmethod
    def get_logprobs(
        self, 
        prompt: str
    ) -> List[float]:
        """
        Gets log probabilities for all tokens in a given prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            A list of log probabilities corresponding to each token in the prompt
        """
        ...

class VLLMProvider(RemoteLLMProvider):
    def __init__(self, api_url: str, tokenizer: TokenizerLike):
        assert api_url.strip() != "", "API URL is required"
        
        self.api_url = api_url
        self.tokenizer = tokenizer
        self.client = OpenAI(base_url=self.api_url,
                             api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
                            )
        

    def generate(self, prompt: str, **gen_args: Any) -> str:
        if not 'max_tokens' in gen_args:
            gen_args['max_tokens'] = gen_args.pop('max_new_tokens', 25)

        response = self.client.completions.create(
            model=self.tokenizer.name_or_path,
            prompt=prompt,
            **gen_args
        )
        
        return response.choices[0].text
    
    def get_logprobs(self, prompt: str) -> List[float]:
        response = self.client.completions.create(
            model=self.tokenizer.name_or_path,
            prompt=prompt,
            temperature=0.0,
            max_tokens=1,
            extra_body={"prompt_logprobs": 0}
        )
        prompt_logprobs = []
        for probs in response.choices[0]['prompt_logprobs']:
            prompt_logprobs.append(list(probs.values())[0]['logprob'])
        
        return prompt_logprobs
        