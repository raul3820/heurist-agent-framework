from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from mesh_manager import AgentLoader, Config
import os
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MeshAPI")

app = FastAPI()

config = Config()
agents_dict = AgentLoader(config).load_agents()

class MeshRequest(BaseModel):
    agent_id: str
    input: Dict[str, Any]
    api_key: str | None = None
    heurist_api_key: str | None = None

@app.post("/mesh_request")
async def process_mesh_request(request: MeshRequest):
    if request.agent_id not in agents_dict:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
    
    agent_cls = agents_dict[request.agent_id]
    agent = agent_cls()
    
    if request.heurist_api_key:
        agent.set_heurist_api_key(request.heurist_api_key)
    
    # Handle API credit deduction if enabled
    credits_api_url = os.getenv('HEURIST_CREDITS_DEDUCTION_API')
    credits_api_auth = os.getenv('HEURIST_CREDITS_DEDUCTION_AUTH')
    if credits_api_url:
        if not request.api_key:
            raise HTTPException(status_code=401, detail="API key is required")
        if not credits_api_auth:
            raise HTTPException(status_code=500, detail="Credits API auth not configured")
        try:
            # Parse user_id and api_key, split by first occurrence only
            if '#' in request.api_key:
                user_id, api_key = request.api_key.split('#', 1)
            else:
                user_id, api_key = request.api_key.split('-', 1)
            
            logger.info(f"Deducting credits for agent {request.agent_id} with user_id {user_id} and api_key {api_key}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    credits_api_url,
                    headers={"Authorization": credits_api_auth},
                    json={
                        'user_id': user_id,
                        'api_key': api_key,
                        'model_type': 'AGENT',
                        'model_id': request.agent_id
                    }
                ) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=403, detail="API credit validation failed")
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid API key format")
        except Exception as e:
            logger.error(f"Error validating API credits: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error validating API credits")
    
    try:
        result = await agent.call_agent(request.input)
        await agent.cleanup()
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """
    Return a list of available agents and their metadata,
    including any tools that each agent supports.
    """
    agents_info = {}
    for agent_id, agent_cls in agents_dict.items():
        agent = agent_cls()
        # We can optionally re-generate or just rely on the stored metadata from S3.
        # For a quick approach, re-attach the tools right now.
        tools = None
        if hasattr(agent, 'get_tool_schemas') and callable(agent.get_tool_schemas):
            tools = agent.get_tool_schemas()
        
        agents_info[agent_id] = {
            "metadata": agent.metadata,
            "module": agent_cls.__module__.split('.')[-1],
            "tools": tools
        }

    return agents_info
