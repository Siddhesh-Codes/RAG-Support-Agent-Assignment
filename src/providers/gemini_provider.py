"""Google Gemini LLM provider implementation using google-genai SDK."""

import os
from typing import Optional, Any
from google import genai
from google.genai import types

from src.llm_provider import LLMProvider, LLMResponse, ToolCall


class GeminiProvider(LLMProvider):
    """Gemini LLM provider using the modern google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please provide it in .env or via environment."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def get_model_name(self) -> str:
        return self.model

    def _convert_schema(self, schema_dict: dict) -> dict:
        """Sanitize standard JSON Schema for Gemini format."""
        # Convert type names to uppercase (e.g. 'object' -> 'OBJECT', 'string' -> 'STRING')
        res = {}
        for k, v in schema_dict.items():
            if k == "type" and isinstance(v, str):
                res["type"] = v.upper()
            elif k == "properties" and isinstance(v, dict):
                res["properties"] = {
                    prop_name: self._convert_schema(prop_def)
                    for prop_name, prop_def in v.items()
                }
            elif k == "items" and isinstance(v, dict):
                res["items"] = self._convert_schema(v)
            else:
                res[k] = v
        return res

    def _convert_tools(self, tools: list[dict]) -> list[types.Tool]:
        """Convert list of tool schemas to Gemini types.Tool objects."""
        function_declarations = []
        for tool in tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = self._convert_schema(tool.get("parameters", {}))

            fn_decl = types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters,
            )
            function_declarations.append(fn_decl)

        return [types.Tool(function_declarations=function_declarations)]

    def _build_contents(
        self,
        messages: list[dict],
        tool_results: Optional[list[dict]] = None,
    ) -> tuple[Optional[str], list[types.Content]]:
        """Extract system instruction and format messages as Gemini Content objects."""
        system_instruction = None
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)],
                    )
                )
            elif role in ("assistant", "model"):
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        # Append tool results if provided
        if tool_results:
            parts = []
            for tr in tool_results:
                tool_name = tr.get("tool_name", "tool")
                result_data = tr.get("result", {})
                parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response=result_data,
                    )
                )
            if parts:
                contents.append(types.Content(role="user", parts=parts))

        return system_instruction, contents

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Execute chat completion with tool calling support."""
        system_instruction, contents = self._build_contents(messages, tool_results)

        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            # Disable automatic tool execution so our agent retains full control
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = types.GenerateContentConfig(**config_kwargs)

        # Robust exponential backoff for transient 503 / 429 API spikes
        max_retries = 8
        base_delay = 2.0
        last_exception = None

        import time
        import re
        import sys

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                last_exception = e
                err_msg = str(e)
                # Check for transient server overload or rate limits
                if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    delay = base_delay * (1.8 ** attempt)
                    # Parse suggested retry delay if present in error message (e.g. "Please retry in 8.8s" or "retryDelay: '8s'")
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                    if match:
                        suggested = float(match.group(1)) + 1.0
                        delay = max(delay, suggested)
                    else:
                        match2 = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", err_msg, re.IGNORECASE)
                        if match2:
                            suggested = float(match2.group(1)) + 1.0
                            delay = max(delay, suggested)
                        elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                            delay = max(delay, 9.0)

                    print(f"[{self.model}] Rate limit / server busy (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s...", file=sys.stderr, flush=True)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"Gemini API call failed: {str(e)}") from e
        else:
            raise RuntimeError(f"Gemini API call failed after {max_retries} retries: {str(last_exception)}") from last_exception

        tool_calls: list[ToolCall] = []

        # Check for function calls from the model
        if response.function_calls:
            for fc in response.function_calls:
                args = fc.args if isinstance(fc.args, dict) else {}
                tool_calls.append(ToolCall(name=fc.name, arguments=args))

        text_content = response.text or ""

        return LLMResponse(
            text=text_content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )
