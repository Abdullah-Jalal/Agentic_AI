from fastmcp import FastMCP
import random
import json

mcp = FastMCP("simple calculater Server")

@mcp.tool()
def add(a:int , b:int) ->int:
    """add two numbers, the function will return the sum of a and b"""
    return a+b

@mcp.tool()
def random_number(min:int , max:int) -> int:
    """return a random number between min and max"""
    return random.randint(min,max)

@mcp.tool()
def server_info()->str:
    '''server is run on local host'''
    info={
        "host":"0.0.0.0",
        "port":8080
    }
    return json.dumps(info,indent=2)

@mcp.resource("server:///info", mime_type="application/json")
def get_server_info_resource():
    info={
        "host":"0.0.0.0",
        "port":8080
    }
    return json.dumps(info,indent=2)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8080)