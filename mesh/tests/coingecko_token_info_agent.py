import sys
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.coingecko_token_info_agent import CoinGeckoTokenInfoAgent
import asyncio

load_dotenv()

async def run_agent():
    agent = CoinGeckoTokenInfoAgent()
    try:
        # Test with a query mentioning CoinGecko ID
        agent_input = {
            'query': 'Get information about MONA'
        }
        
        agent_output = await agent.handle_message(agent_input)
        print(f"result of handle_message: {agent_output}")

        # Test with query mentioned the token name 
        agent_input_name = {
            'query': 'analyze HEU'
        }
        
        agent_output_name = await agent.handle_message(agent_input_name)
        print(f"result of handle_message when token name is provided: {agent_output_name}")

        # test with trending coins
        agent_input_trending = {
            'query': 'Get information about trending coins'
        }
        agent_output_trending = await agent.handle_message(agent_input_trending)
        print(f"result of handle_message when trending coins is provided: {agent_output_trending}")
        
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            'input_by_id': agent_input,
            'output_by_id': agent_output,
            'input_by_name': agent_input_name,
            'output_by_name': agent_output_name,
            'input_by_trending': agent_input_trending,
            'output_by_trending': agent_output_trending
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
            
        print(f"Results saved to {output_file}")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(run_agent())