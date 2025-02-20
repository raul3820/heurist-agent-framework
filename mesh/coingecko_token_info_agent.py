import requests
import os
from .mesh_agent import MeshAgent, with_cache, with_retry, monitor_execution
from core.llm import call_llm_async, call_llm_with_tools_async
from typing import List, Dict, Any
import json

class CoinGeckoTokenInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            'Authorization': f'Bearer {os.getenv("COINGECKO_API_KEY")}'
        }
        
        # Add required metadata
        self.metadata.update({
            'name': 'CoinGecko Token Info Agent',
            'version': '1.0.2',
            'author': 'Heurist team',
            'author_address': '0x7d9d1821d15B9e0b8Ab98A058361233E255E405D',
            'description': 'This agent can fetch token information, market data, and trending coins from CoinGecko. ',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Natural language query about a token, or a request for trending coins. ',
                    'type': 'str',
                    'required': False
                },
                {
                    'name': 'raw_data_only',
                    'description': 'If true, the agent will only return the raw or base structured data without additional LLM explanation.',
                    'type': 'bool',
                    'required': False,
                    'default': False
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Natural language explanation of the token information (empty if a direct tool call).',
                    'type': 'str'
                },
                {
                    'name': 'data',
                    'description': 'Structured token information or trending coins data.',
                    'type': 'dict'
                }
            ],
            'external_apis': ['Coingecko'],
            'tags': ['Trading', 'Data']
        })

    def get_system_prompt(self) -> str:
        return """You are a helpful assistant that can fetch token information from CoinGecko.
        You can:
        1. Search for specific tokens and get their details
        2. Get current trending coins in the market
        
        For specific token queries, identify whether the user provided a CoinGecko ID directly or needs to search by token name.
        For trending coins requests, use the get_trending_coins tool to fetch the current top trending cryptocurrencies.
        
        When selecting tokens from search results, apply these criteria in order:
        1. First priority: Select the token where name or symbol perfectly matches the query
        2. If multiple matches exist, select the token with the highest market cap rank (lower number = higher rank)
        3. If market cap ranks are not available, prefer the token with the most complete information"""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'get_coingecko_id',
                    'description': 'Search for a token by name to get its CoinGecko ID',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_name': {
                                'type': 'string',
                                'description': 'The token name to search for'
                            }
                        },
                        'required': ['token_name'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_token_info',
                    'description': 'Get detailed token information and market data using CoinGecko ID',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'coingecko_id': {
                                'type': 'string',
                                'description': 'The CoinGecko ID of the token'
                            }
                        },
                        'required': ['coingecko_id'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_trending_coins',
                    'description': 'Get the current top trending cryptocurrencies on CoinGecko',
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': [],
                    },
                }
            }
        ]



    async def select_best_token_match(self, search_results: Dict, query: str) -> str:
        """
        Select best matching token using the following criteria:
        1. Ignore tokens with None market_cap_rank
        2. Find closest name/symbol match
        3. Use market cap rank as tiebreaker
        """
        if not search_results.get('coins'):
            return None
            
        # Filter out tokens with None market_cap_rank
        valid_tokens = [
            token for token in search_results['coins'] 
            if token.get('market_cap_rank') is not None
        ]
        
        if not valid_tokens:
            return None
        
        # Create prompt for token selection
        token_selection_prompt = f"""Given the search query "{query}" and these token results:
        {json.dumps(valid_tokens, indent=2)}
        
        Select the most appropriate token based on these criteria in order:
        1. Find the token where name or symbol most closely matches the query
        - Exact matches are preferred
        - For partial matches, consider string similarity and common variations
        2. If multiple tokens have similar name matches, select the one with the highest market cap rank (lower number = higher rank)
        
        Return only the CoinGecko ID of the selected token, nothing else."""
        
        selected_token = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['small_model_id'],
            messages=[
                {"role": "system", "content": "You are a token selection assistant. You only return the CoinGecko ID of the best matching token based on the given criteria."},
                {"role": "user", "content": token_selection_prompt}
            ],
            temperature=0.1
        )
        
        # Clean up response to get just the ID
        selected_token = selected_token.strip().strip('"').strip("'")
        
        # Verify the selected ID exists in filtered results
        if any(token['id'] == selected_token for token in valid_tokens):
            return selected_token
        return None

    # ------------------------------------------------------------------------
    #                       SHARED / UTILITY METHODS
    # ------------------------------------------------------------------------
    async def _respond_with_llm(
        self, query: str, tool_call_id: str, data: dict, temperature: float
    ) -> str:
        """
        Reusable helper to ask the LLM to generate a user-friendly explanation
        given a piece of data from a tool call.
        """
        return await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(data), "tool_call_id": tool_call_id}
            ],
            temperature=temperature
        )

    def _handle_error(self, maybe_error: dict) -> dict:
        """
        Small helper to return the error if present in
        a dictionary with the 'error' key.
        """
        if 'error' in maybe_error:
            return {"error": maybe_error['error']}
        return {}

    # ------------------------------------------------------------------------
    #                      COINGECKO API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    async def get_trending_coins(self) -> dict:
        try:
            response = requests.get(
                f"{self.api_url}/search/trending",
                headers=self.headers
            )
            response.raise_for_status()
            trending_data = response.json()
            
            # Format the trending coins data
            formatted_trending = []
            for coin in trending_data.get('coins', [])[:10]:
                coin_info = coin['item']
                formatted_trending.append({
                    'name': coin_info['name'],
                    'symbol': coin_info['symbol'],
                    'market_cap_rank': coin_info.get('market_cap_rank', 'N/A'),
                    'price_usd': coin_info["data"].get('price', 'N/A')
                })
            return {'trending_coins': formatted_trending}
            
        except requests.RequestException as e:
            print(f"error: {e}")
            return {"error": f"Failed to fetch trending coins: {str(e)}"}

    @with_cache(ttl_seconds=3600)
    async def get_coingecko_id(self, token_name: str) -> dict | str:
        try:
            response = requests.get(
                f"{self.api_url}/search?query={token_name}",
                headers=self.headers
            )
            response.raise_for_status()
            search_results = response.json()
            print(f"search_results: {search_results}")
            # Return the first coin id if found
            if search_results.get('coins'):
                first_coin = search_results['coins'][0]
                return first_coin['id']
            
            selected_token_id = await self.select_best_token_match(search_results, token_name)
            return selected_token_id or None
        
        except requests.RequestException as e:
            print(f"error: {e}")
            return {"error": f"Failed to search for token: {str(e)}"}

    @with_cache(ttl_seconds=3600)
    async def get_token_info(self, coingecko_id: str) -> dict:
        try:
            response = requests.get(
                f"{self.api_url}/coins/{coingecko_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"error: {e}")
            return {"error": f"Failed to fetch token info: {str(e)}"}

    def format_token_info(self, data: Dict) -> Dict:
        """Format token information in a structured way"""
        market_data = data.get('market_data', {})
        return {
            "token_info": {
                "name": data.get('name', 'N/A'),
                "symbol": data.get('symbol', 'N/A').upper(),
                "market_cap_rank": data.get('market_cap_rank', 'N/A'),
                "categories": data.get('categories', []),
                "description": data.get('description', {}).get('en', 'N/A')
            },
            "market_metrics": {
                "current_price_usd": market_data.get('current_price', {}).get('usd', 'N/A'),
                "market_cap_usd": market_data.get('market_cap', {}).get('usd', 'N/A'),
                "fully_diluted_valuation_usd": market_data.get('fully_diluted_valuation', {}).get('usd', 'N/A'),
                "total_volume_usd": market_data.get('total_volume', {}).get('usd', 'N/A'),
            },
            "price_metrics": {
                "ath_usd": market_data.get('ath', {}).get('usd', 'N/A'),
                "ath_change_percentage": market_data.get('ath_change_percentage', {}).get('usd', 'N/A'),
                "ath_date": market_data.get('ath_date', {}).get('usd', 'N/A'),
                "high_24h_usd": market_data.get('high_24h', {}).get('usd', 'N/A'),
                "low_24h_usd": market_data.get('low_24h', {}).get('usd', 'N/A'),
                "price_change_24h": market_data.get('price_change_24h', 'N/A'),
                "price_change_percentage_24h": market_data.get('price_change_percentage_24h', 'N/A'),
            },
            "supply_info": {
                "total_supply": market_data.get('total_supply', 'N/A'),
                "max_supply": market_data.get('max_supply', 'N/A'),
                "circulating_supply": market_data.get('circulating_supply', 'N/A')
            }
        }

    # ------------------------------------------------------------------------
    #                      COMMON HANDLER LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, query: str, tool_call_id: str, raw_data_only: bool
    ) -> Dict[str, Any]:
        """
        A single method that calls the appropriate function, handles
        errors/formatting, and optionally calls the LLM to explain the result.
        """
        # Decide on temperature defaults per tool:
        # (You could also store these in a dictionary if you want more variation)
        temp_for_explanation = 0.7 if tool_name != 'get_trending_coins' else 0.3

        if tool_name == 'get_trending_coins':
            result = await self.get_trending_coins()
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query,
                tool_call_id=tool_call_id,
                data=result,
                temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        elif tool_name == 'get_coingecko_id':
            token_name = function_args.get('token_name')
            if not token_name:
                return {"error": "Missing 'token_name' in tool_arguments"}
            
            raw_id_result = await self.get_coingecko_id(token_name)
            if isinstance(raw_id_result, dict) and 'error' in raw_id_result:
                return raw_id_result

            # raw_id_result is presumably just the string id
            coingecko_id = raw_id_result
            # Optionally, also fetch token info or just return the ID:
            # We'll just return the ID here:
            final_data = {"coingecko_id": coingecko_id}
            
            if raw_data_only:
                return {"response": "", "data": final_data}

            explanation = await self._respond_with_llm(
                query=query,
                tool_call_id=tool_call_id,
                data=final_data,
                temperature=temp_for_explanation
            )
            return {"response": explanation, "data": final_data}

        elif tool_name == 'get_token_info':
            coingecko_id = function_args.get('coingecko_id')
            if not coingecko_id:
                return {"error": "Missing 'coingecko_id' in tool_arguments"}
            
            token_info = await self.get_token_info(coingecko_id)
            errors = self._handle_error(token_info)
            if errors:
                return errors

            formatted_data = self.format_token_info(token_info)
            
            if raw_data_only:
                return {"response": "", "data": formatted_data}

            explanation = await self._respond_with_llm(
                query=query,
                tool_call_id=tool_call_id,
                data=formatted_data,
                temperature=temp_for_explanation
            )
            return {"response": explanation, "data": formatted_data}

        return {"error": f"Unsupported tool '{tool_name}'"}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Either 'query' or 'tool' is required in params.
          - If 'tool' is provided, call that tool directly with 'tool_arguments' (bypassing the LLM).
          - If 'query' is provided, route via LLM for dynamic tool selection.
        """
        query = params.get('query')
        tool_name = params.get('tool')
        tool_args = params.get('tool_arguments', {})
        raw_data_only = params.get('raw_data_only', False)

        # ---------------------
        # 1) DIRECT TOOL CALL
        # ---------------------
        if tool_name:
            # For a direct tool call, we do NOT use call_llm_with_tools_async
            # We simply call our helper. We'll pass an empty `query` or something placeholder
            # if user hasn't given a `query`.
            # We'll also pass tool_call_id = "direct_tool" for consistency.
            return await self._handle_tool_logic(
                tool_name=tool_name,
                function_args=tool_args,
                query=query or "Direct tool call without LLM.",
                tool_call_id="direct_tool",
                raw_data_only=raw_data_only
            )

        # ---------------------
        # 2) NATURAL LANGUAGE QUERY (LLM decides the tool)
        # ---------------------
        if query:
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
                # No tool calls => the LLM just answered
                return {"response": response['content'], "data": {}}

            # LLM provided a single tool call (or the first if multiple).
            tool_call = response['tool_calls']
            tool_call_name = tool_call.function.name
            tool_call_args = json.loads(tool_call.function.arguments)

            # Use the same _handle_tool_logic
            return await self._handle_tool_logic(
                tool_name=tool_call_name,
                function_args=tool_call_args,
                query=query,
                tool_call_id=tool_call.id,
                raw_data_only=raw_data_only
            )

        # ---------------------
        # 3) NEITHER query NOR tool
        # ---------------------
        return {"error": "Either 'query' or 'tool' must be provided in the parameters."}