import os
from typing import Dict, Any, List, Optional
import requests
import time
from datetime import datetime, timedelta
from .mesh_agent import MeshAgent, with_cache, with_retry, monitor_execution
from core.llm import call_llm_async, call_llm_with_tools_async
import json
from dotenv import load_dotenv
import random

load_dotenv()

class ElfaTwitterIntelligenceAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        # Load and parse API keys
        api_keys_str = os.getenv('ELFA_API_KEY')
        if not api_keys_str:
            raise ValueError("ELFA_API_KEY environment variable is required")
        
        # Split and clean the API keys
        self.api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
        if not self.api_keys:
            raise ValueError("No valid API keys found in ELFA_API_KEY")
        
        self.current_api_key = random.choice(self.api_keys)
        self.last_rotation_time = time.time()
        self.rotation_interval = 300  # Rotate every 5 minutes
        
        self.base_url = "https://api.elfa.ai/v1"
        self._update_headers()
        
        self.metadata.update({
            'name': 'ELFA Twitter Intelligence Agent',
            'version': '1.0.0',
            'author': 'Heurist team',
            'author_address': '0x7d9d1821d15B9e0b8Ab98A058361233E255E405D',
            'description': 'This agent analyzes Twitter mentions and trends for crypto tokens using ELFA API',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Natural language query about token mentions, trends, or account analysis',
                    'type': 'str',
                    'required': True
                },
                {
                    'name': 'raw_data_only',
                    'description': 'If true, return only raw data without natural language response',
                    'type': 'bool',
                    'required': False,
                    'default': False
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Natural language explanation of the Twitter analysis',
                    'type': 'str'
                },
                {
                    'name': 'data',
                    'description': 'Structured data from ELFA API',
                    'type': 'dict'
                }
            ],
            'external_apis': ['ELFA'],
            'tags': ['Social', 'Twitter', 'Analysis']
        })

    def _update_headers(self):
        """Update headers with current API key"""
        self.headers = {
            "x-elfa-api-key": self.current_api_key,
            "Accept": "application/json"
        }

    def _rotate_api_key(self):
        """Rotate API key if enough time has passed"""
        current_time = time.time()
        if current_time - self.last_rotation_time >= self.rotation_interval:
            self.current_api_key = random.choice(self.api_keys)
            self._update_headers()
            self.last_rotation_time = current_time

    def get_system_prompt(self) -> str:
        return (
            "You are a specialized assistant that analyzes Twitter data for crypto tokens using ELFA API. "
            "Your responses should be clear, concise, and data-driven.\n\n"
            "When analyzing mentions:\n"
            "1. Summarize the overall sentiment and engagement\n"
            "2. Highlight key influencers and their impact\n"
            "3. Identify notable trends or patterns\n\n"
            "When providing trending tokens:\n"
            "1. List top tokens by mention volume\n"
            "2. Include engagement metrics\n"
            "3. Note any significant changes or patterns\n\n"
            "When analyzing accounts:\n"
            "1. Summarize engagement metrics\n"
            "2. Highlight influential connections\n"
            "3. Note posting patterns and impact"
        )

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'search_mentions',
                    'description': 'Search for mentions of specific keywords',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'keywords': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'List of keywords to search for'
                            },
                            'days_ago': {
                                'type': 'integer',
                                'description': 'Number of days to look back',
                                'default': 30
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Maximum number of results',
                                'default': 20
                            }
                        },
                        'required': ['keywords'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_trending_tokens',
                    'description': 'Get current trending tokens',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'time_window': {
                                'type': 'string',
                                'description': 'Time window to analyze',
                                'default': '24h'
                            }
                        }
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_account_stats',
                    'description': 'Get account analytics and statistics',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'username': {
                                'type': 'string',
                                'description': 'Twitter username to analyze'
                            }
                        },
                        'required': ['username'],
                    },
                }
            }
        ]

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def _make_request(self, endpoint: str, method: str = "GET", params: Dict = None) -> Dict:
        self._rotate_api_key()  # Check and rotate API key if needed
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"ELFA API request failed: {str(e)}")

    @with_cache(ttl_seconds=300)
    async def search_mentions(self, keywords: List[str], days_ago: int = 30, limit: int = 20) -> Dict:
        end_time = int(time.time())
        start_time = int((datetime.now() - timedelta(days=days_ago)).timestamp())
        
        params = {
            'keywords': ','.join(keywords),
            'from': start_time,
            'to': end_time,
            'limit': limit
        }
        
        return await self._make_request("mentions/search", params=params)

    @with_cache(ttl_seconds=300)
    async def get_trending_tokens(self, time_window: str = "24h") -> Dict:
        params = {
            'timeWindow': time_window,
            'page': 1,
            'pageSize': 50,
            'minMentions': 5
        }
        return await self._make_request("trending-tokens", params=params)

    @with_cache(ttl_seconds=300)
    async def get_account_stats(self, username: str) -> Dict:
        params = {'username': username}
        return await self._make_request("account/smart-stats", params=params)

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get('query')
        if not query:
            raise ValueError("Query parameter is required")

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            temperature=0.1,
            tools=self.get_tool_schemas()
        )

        if not response:
            return {"error": "Failed to process query"}

        if not response.get('tool_calls'):
            return {"response": response['content'], "data": {}}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)
        raw_data_only = params.get('raw_data_only', False)

        try:
            if tool_call.function.name == 'search_mentions':
                data = await self.search_mentions(**function_args)
            elif tool_call.function.name == 'get_trending_tokens':
                data = await self.get_trending_tokens(**function_args)
            elif tool_call.function.name == 'get_account_stats':
                data = await self.get_account_stats(**function_args)
            else:
                return {"error": "Unsupported operation"}

            if raw_data_only:
                return {"response": "", "data": data}

            explanation = await call_llm_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata['large_model_id'],
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": query},
                    {"role": "tool", "content": str(data), "tool_call_id": tool_call.id}
                ],
                temperature=0.6
            )

            return {
                "response": explanation,
                "data": data
            }

        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}