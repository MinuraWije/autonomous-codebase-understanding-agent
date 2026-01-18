"""LLM wrapper for Hugging Face models."""
from typing import Optional, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import requests
import json


class HuggingFaceChatLLM(BaseChatModel):
    """Wrapper for Hugging Face Inference API chat models."""
    
    model_name: str
    api_key: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    base_url: str = "https://router.huggingface.co/v1/chat/completions"
    
    def __init__(self, model: str = None, huggingface_api_key: str = None, model_name: str = None, api_key: str = None, temperature: float = 0.0, **kwargs):
        # Support both naming conventions - map to Pydantic field names
        final_model = model or model_name
        final_api_key = huggingface_api_key or api_key
        
        if not final_model:
            raise ValueError("model or model_name is required")
        if not final_api_key:
            raise ValueError("huggingface_api_key or api_key is required")
        
        # Pass to Pydantic with correct field names
        super().__init__(model_name=final_model, api_key=final_api_key, temperature=temperature, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from Hugging Face API."""
        # Convert LangChain messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content from response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
            else:
                # Fallback for different response formats
                content = result.get("generated_text", str(result))
            
            # Create ChatGeneration
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Hugging Face API error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {error_detail}"
                    
                    # Check for model not supported error
                    if isinstance(error_detail, dict):
                        error_info = error_detail.get('error', {})
                        if isinstance(error_info, dict) and 'model_not_supported' in str(error_info):
                            error_msg += "\n\n💡 Tip: The model may not be available on the router API. "
                            error_msg += "Try updating the model in your .env file to a supported model like: "
                            error_msg += "meta-llama/Llama-3.2-3B-Instruct or google/gemma-2-2b-it"
                except:
                    error_msg += f" - Status: {e.response.status_code}"
            
            # Return error as message
            message = AIMessage(content=f"Error: {error_msg}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Hugging Face API error: {str(e)}"
            message = AIMessage(content=f"Error: {error_msg}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Streaming not implemented yet."""
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        yield result.generations[0]
