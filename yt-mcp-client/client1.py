import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVERS = {
    "yt-mcp-client": {
        "transport": "stdio",
        "command": r"C:\Users\M.LAPTOP\.local\bin\uv.exe",
        "args": [
            "--directory",
            r"C:\LangGraph\mcp",
            "run",
            "python",
            "main.py",
        ]
    }
}

async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())
