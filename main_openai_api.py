import os
import logging
import json
import dotenv
dotenv.load_dotenv()
from pathlib import Path
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from interfaces.openai_interface import handle_chat_completion, forward

HEURIST_BASE_URL = os.getenv("HEURIST_BASE_URL")
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")

app = FastAPI()
db = None
client = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, client
    logging.info("Starting application...")
    client = AsyncOpenAI(
        base_url=HEURIST_BASE_URL,
        api_key=HEURIST_API_KEY,
        )
    yield
    logging.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def route(request: Request, call_next):
    """Middleware to proxy requests to OpenAI API or handle them manually based on the model."""
    path = request.url.path
    if path.endswith("/chat/completions"):
        try:
            body = await request.body()

            if not body:
                raise HTTPException(status_code=400, detail="Request body missing")
            chat_request = json.loads(body)
            
            return await handle_chat_completion(chat_request, client)

        except HTTPException as he:
            logging.exception(f'HTTPException: {he}')
            return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
        except Exception as e:
            logging.exception(f'Exception: {e}')
            return JSONResponse(
                content={"error": f"Failed to process request: {str(e)}"}, 
                status_code=500
            )
    
    return await forward(request, HEURIST_BASE_URL, HEURIST_API_KEY)



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'main_openai_api:app', reload=True, host="0.0.0.0", port=5050, reload_dirs=[str(Path(__file__).parent)]
    )