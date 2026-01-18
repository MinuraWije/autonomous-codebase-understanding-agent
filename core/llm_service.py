"""LLM service factory and interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from agent.llm_wrapper import HuggingFaceChatLLM
from app.config import settings
from core.constants import (
    PLANNER_TEMPERATURE,
    SYNTHESIZER_TEMPERATURE,
    VERIFIER_TEMPERATURE,
    SUMMARY_TEMPERATURE
)
from core.exceptions import LLMError


class LLMServiceInterface(ABC):
    """Interface for LLM services."""
    
    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke the LLM with messages."""
        pass
    
    @abstractmethod
    def invoke_text(self, text: str) -> str:
        """Invoke the LLM with a text prompt."""
        pass


class HuggingFaceLLMService(LLMServiceInterface):
    """Hugging Face LLM service implementation."""
    
    def __init__(self, model: str, temperature: float = 0.0):
        """
        Initialize the LLM service.
        
        Args:
            model: Model name
            temperature: Temperature for generation
        """
        try:
            self.llm = HuggingFaceChatLLM(
                model=model,
                huggingface_api_key=settings.huggingface_api_key,
                temperature=temperature
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM: {e}")
    
    def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke the LLM with messages."""
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise LLMError(f"LLM invocation failed: {e}")
    
    def invoke_text(self, text: str) -> str:
        """Invoke the LLM with a text prompt."""
        from langchain_core.messages import HumanMessage
        return self.invoke([HumanMessage(content=text)])


class LLMServiceFactory:
    """Factory for creating LLM services."""
    
    @staticmethod
    def create_planner_service() -> LLMServiceInterface:
        """Create a planner LLM service."""
        return HuggingFaceLLMService(
            model=settings.planner_model,
            temperature=PLANNER_TEMPERATURE
        )
    
    @staticmethod
    def create_synthesizer_service() -> LLMServiceInterface:
        """Create a synthesizer LLM service."""
        return HuggingFaceLLMService(
            model=settings.synthesizer_model,
            temperature=SYNTHESIZER_TEMPERATURE
        )
    
    @staticmethod
    def create_verifier_service() -> LLMServiceInterface:
        """Create a verifier LLM service."""
        return HuggingFaceLLMService(
            model=settings.verifier_model,
            temperature=VERIFIER_TEMPERATURE
        )
    
    @staticmethod
    def create_summary_service() -> LLMServiceInterface:
        """Create a summary LLM service."""
        return HuggingFaceLLMService(
            model=settings.planner_model,
            temperature=SUMMARY_TEMPERATURE
        )
    
    @staticmethod
    def create_service(model: str, temperature: float = 0.0) -> LLMServiceInterface:
        """
        Create a custom LLM service.
        
        Args:
            model: Model name
            temperature: Temperature for generation
        
        Returns:
            LLM service instance
        """
        return HuggingFaceLLMService(model=model, temperature=temperature)
