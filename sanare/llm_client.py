from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class LLMUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "auto").lower()
    model: str | None = os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL")
    base_url: str | None = os.getenv("LLM_BASE_URL") or os.getenv("VLLM_BASE_URL")
    temperature: float = 0.0


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        provider = self._resolve_provider()
        if provider in {"openai", "vllm", "nvidia"}:
            return self._openai_compatible(system_prompt, user_prompt, provider)
        if provider == "gemini":
            return self._gemini(system_prompt, user_prompt)
        raise LLMUnavailableError("No LLM provider configured")

    def _resolve_provider(self) -> str:
        if self.config.provider in {"openai", "gemini", "vllm", "nvidia"}:
            return self.config.provider
        if os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_BASE_URL"):
            return "nvidia"
        if self.config.base_url:
            return "vllm"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        raise LLMUnavailableError("Set OPENAI_API_KEY, GEMINI_API_KEY, or VLLM_BASE_URL")

    def _openai_compatible(self, system_prompt: str, user_prompt: str, provider: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailableError("Install openai package") from exc

        client_kwargs: dict[str, Any] = {}
        if provider == "nvidia":
            client_kwargs["base_url"] = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
            client_kwargs["api_key"] = os.getenv("NVIDIA_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
        elif self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
            client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "EMPTY")

        client = OpenAI(**client_kwargs)
        request_kwargs: dict[str, Any] = {
            "model": self.config.model or self._default_model(provider),
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if provider in {"vllm", "nvidia"}:
            request_kwargs["extra_body"] = {
                "structured_outputs": {
                    "json": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["patient_summary", "conditions", "medications", "risk_level", "next_step"],
                        "properties": {
                            "patient_summary": {"type": "string"},
                            "conditions": {"type": "array", "items": {"type": "string"}},
                            "medications": {"type": "array", "items": {"type": "string"}},
                            "risk_level": {"type": "string", "enum": ["low", "moderate", "high"]},
                            "next_step": {"type": "string"},
                        },
                    }
                }
            }

        response = client.chat.completions.create(
            **request_kwargs,
        )
        return response.choices[0].message.content or "{}"

    def _default_model(self, provider: str) -> str:
        if provider == "vllm":
            return "Qwen/Qwen2.5-7B-Instruct"
        if provider == "nvidia":
            return os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
        return "gpt-4o-mini"

    def _gemini(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMUnavailableError("Install google-generativeai package") from exc

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            self.config.model or "gemini-1.5-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_prompt,
            generation_config={
                "temperature": self.config.temperature,
                "response_mime_type": "application/json",
            },
        )
        return response.text or "{}"

