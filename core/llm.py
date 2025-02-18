import requests
import json
import time
import os
import logging
from openai import OpenAI, AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from typing import Dict, List, Union, AsyncIterator, AsyncGenerator, Literal, Callable
from contextlib import asynccontextmanager
import requests
from types import SimpleNamespace
import re
import asyncio
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

def _format_messages(system_prompt: str = None, user_prompt: str = None, messages: List[Dict] = None) -> List[Dict]:
    """Convert between different message formats while maintaining backward compatibility"""
    if messages is not None:
        return messages
    if system_prompt is not None and user_prompt is not None:
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    raise ValueError("Either (system_prompt, user_prompt) or messages must be provided")

def call_llm(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
    initial_retry_delay: int = 1
) -> str:
    """
    Call LLM with retry mechanism.

    Parameters:
        model_id (str): The model identifier for the LLM.
        system_prompt (str): The system prompt.
        user_prompt (str): The user input prompt.
        temperature (float): The temperature setting for response generation.
        max_tokens (int): Maximum number of tokens to generate.
        max_retries (int): Number of retry attempts on failure.
        initial_retry_delay (int): Initial delay between retries, with exponential backoff.

    Returns:
        str: Generated text from LLM.
    
    Raises:
        LLMError: If all retry attempts fail.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(
                model=model_id,
                messages=formatted_messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return result.choices[0].message.content
            
        except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"{type(e).__name__} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
    
    raise LLMError("All retry attempts failed")

def call_llm_with_tools(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = None,
    max_retries: int = 3,
    tools: List[Dict] = None,
    tool_choice: str = "auto"
) -> Union[str, Dict]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice if tools else None,
            max_tokens=max_tokens
        )
        return _handle_tool_response(response.choices[0].message)

    except Exception as e:
        raise LLMError(f"LLM API call failed: {str(e)}")

async def call_llm_async(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
    initial_retry_delay: int = 1
) -> str:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=model_id,
                messages=formatted_messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return result.choices[0].message.content
            
        except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"{type(e).__name__} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
    
    raise LLMError("All retry attempts failed")

async def call_llm_with_tools_async(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
    tools: List[Dict] = None,
    tool_choice: str = "auto"
) -> Union[str, Dict]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice if tools else None,
            max_tokens=max_tokens
        )
        return _handle_tool_response(response.choices[0].message)

    except Exception as e:
        raise LLMError(f"LLM API call failed: {str(e)}")

def extract_function_calls_to_tool_calls(llm_text: str) -> SimpleNamespace:
    
    """
    Scan the LLM's text output for a <function=NAME>{...}</function> pattern,
    and convert to appropriate format for tool calls
    """
    pattern = r"<function=([^>]+)>(.*?)(?:</function>|<function>|<function/>|></function>)"
    matches = re.findall(pattern, llm_text)
    
    # If we find at least one match
    if matches:
        function_name, args_json_str = matches[0]  # Just take the first match
        # Parse the JSON to ensure it's valid
        parsed_args = json.loads(args_json_str.strip())
        

        function_obj = SimpleNamespace(
            name=function_name,
            arguments=json.dumps(parsed_args)
        )
        # Build the structure that your existing code expects
        return SimpleNamespace(function=function_obj)
    
    # If no matches, return an empty dict or whatever fallback you need
    return None

def _handle_tool_response(message):
    if hasattr(message, 'tool_calls') and message.tool_calls:
        return {
            'tool_calls': message.tool_calls[0],
            'content': message.content
        }
    if hasattr(message, 'content') and message.content:
        text_response = message.content
        tool_calls = extract_function_calls_to_tool_calls(text_response)
        if tool_calls:
            logger.info("found tool calls in response")
            return {
                'tool_calls': tool_calls,
                'content': ""
            }
        else:
            return {
                'content': text_response
            }
    return message

@asynccontextmanager
async def call_llm_with_tools_stream(
    base_url: str,
    api_key: str,
    model_id: str,
    client: AsyncOpenAI = None,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
    tools: List[Callable] = None,
    tool_choice: Literal['none','auto','required'] = "auto",
    sniff_chunks: int = 5,
    model_settings: Dict = {},
) -> AsyncGenerator[AsyncIterator, None]:
    """
    Returns an async context manager that yields an async iterator of text chunks.
    If a tool is sniffed in the first n chunks, that tool is invoked and tool 
    outputs are streamed; otherwise, the model's streaming output is returned.
    """
    assert client or (base_url and api_key)
    if not client:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries) 
    # to-do: pass client as arg
    model_settings = model_settings or {}
    assert user_prompt or messages, 'Missing user_prompt or list of messages.'
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
    # tools = ChatCompletionToolParam(*tools)
    async def generator() -> AsyncIterator[str]:
        attempt = 0
        while attempt < max_retries:
            try:
                response_text = ""
                i = 0
                final_tool_calls: dict[int, ChoiceDeltaToolCall] = {}
                stream = await client.chat.completions.create(
                    model=model_id,
                    messages=formatted_messages,
                    n=1,
                    parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                    tools=(None if tool_choice == 'none' else tools) or NOT_GIVEN,
                    tool_choice=tool_choice or NOT_GIVEN,
                    stream=True,
                    stream_options={'include_usage': True},
                    max_tokens=max_tokens or NOT_GIVEN,
                    temperature=temperature or NOT_GIVEN,
                    top_p=model_settings.get('top_p', NOT_GIVEN),
                    timeout=model_settings.get('timeout', NOT_GIVEN),
                    seed=model_settings.get('seed', NOT_GIVEN),
                    presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                    frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                    logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                    reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
                )
                
                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta: ChoiceDelta = chunk.choices[0].delta
                    
                    for tool_call in delta.tool_calls or []:
                        index = tool_call.index
                        if index not in final_tool_calls:
                            final_tool_calls[index] = tool_call

                    if i < sniff_chunks or final_tool_calls or '<function' in response_text:
                        if delta.content:
                            response_text += delta.content
                        if not final_tool_calls:
                            function_tool_call = extract_function_calls_to_tool_calls(response_text)
                            if function_tool_call:
                                final_tool_calls[0] = function_tool_call
                    else:
                        if response_text:
                            yield response_text
                            response_text = ''
                        yield delta.content
                    
                    i += 1
                
                if final_tool_calls:
                    for i, tool_call in final_tool_calls.items():
                        yield tool_call
                        # implement tool execution here and retry with feedback in prompt?

                if response_text:
                    yield response_text
                    response_text = ''

                if attempt == 0:
                    break
            except Exception as e:
                attempt += 1
                logging.exception(f"Attempt {attempt} of {max_retries} -- {e}")
                yield f"Attempt {attempt} of {max_retries} -- {e}\n"

    try:
        yield generator()
    finally:
        # cleanup?
        pass

