import logging
import json
import uuid
import httpx
from fastapi import Request, Request, HTTPException
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Union
from time import time
from core import llm

from agents.tool_box import get_tool_schemas
from examples.test_streaming_tool import roll_dice

async def handle_chat_completion(chat_request: dict, client: AsyncOpenAI) -> StreamingResponse:
    """Handle inference using custom models and stream results in OpenAI format as SSE."""

    async def stream_generator():
        try:
            model = chat_request.get("model")
            messages = chat_request.get("messages")
            id = uuid.uuid4()
            print(f'tool schemas: {get_tool_schemas([roll_dice])}')
            async with llm.call_llm_with_tools_stream(
                base_url=None,
                api_key=None,
                model_id=model,
                client=client,
                messages=messages,
                # tools=get_tool_schemas([roll_dice]),
                # tool_choice='auto',
            ) as result:
                print(f'result type is {type(result)}') # result type is <class 'async_generator'>
                async for chunk in result:
                    openai_response = {
                        "id": f"chatcmpl-{id}",
                        "object": "chat.completion.chunk",
                        "created": int(time()),
                        "model": model,
                        "choices": [
                            {
                                "delta": {"content": chunk},
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(openai_response)}\n\n"
                    print(f"data: {json.dumps(openai_response)}\n\n")

            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.exception(f"Exception: {e}")
            error_response = {"error": str(e)}
            yield f"data: {json.dumps(error_response)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


async def forward(request: Request, api_url: str, api_key: str) -> Union[StreamingResponse, JSONResponse]:
    """Proxy forward to OpenAI API with simplified error handling."""
    try:
        headers = {**request.headers, "authorization": f"Bearer {api_key}"}
        request_content = await request.body()
        request_content_str = request_content.decode("utf-8")
        url = f"{api_url}{request.url.path}"

        # Handle streaming requests
        if request_content_str and json.loads(request_content_str).get("stream"):
            upstream_media_type = None

            async def stream_generator():
                nonlocal upstream_media_type
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream(
                        method=request.method,
                        url=url,
                        headers=headers,
                        content=request_content,
                    ) as upstream_response:
                        if upstream_response.status_code >= 400:
                            error_content = await upstream_response.read()
                            raise HTTPException(
                                status_code=upstream_response.status_code,
                                detail=error_content.decode('utf-8')
                            )

                        upstream_media_type = upstream_response.headers.get(
                            "content-type", "application/json"
                        )
                        async for chunk in upstream_response.aiter_bytes():
                            yield chunk

            # Prime stream_generator to capture the media type
            gen = stream_generator().__aiter__()
            try:
                first_chunk = await gen.__anext__()
            except StopAsyncIteration:
                first_chunk = b""
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": e.detail}
                )

            async def final_generator():
                yield first_chunk
                async for chunk in gen:
                    yield chunk

            return StreamingResponse(
                content=final_generator(),
                media_type=upstream_media_type or "application/json",
            )
        
        # Handle non-streaming requests
        else:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=request_content,
                )

                if response.status_code >= 400:
                    return JSONResponse(
                        status_code=response.status_code,
                        content={"error": response.text}
                    )

                return StreamingResponse(
                    content=response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=response.headers,
                )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )