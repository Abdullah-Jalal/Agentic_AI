import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_server():
    url = "http://localhost:8002/mcp"
    print(f"Connecting to {url}...")
    
    try:
        async with sse_client(url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                print("✅ Successfully connected to MCP server!\n")
                
                # List tools
                print("🛠️  Available Tools:")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                    
                # Call a tool
                print("\n🚀 Testing 'add' tool...")
                result = await session.call_tool("add", arguments={"a": 5, "b": 10})
                print(f"Result: {result.content[0].text}")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_server())
