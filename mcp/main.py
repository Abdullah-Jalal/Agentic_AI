import random
from fastmcp import FastMCP

mcp = FastMCP(name="Demo-Server")

@mcp.tool
def roll_dice(n_dice: int = 1) -> list[int]:
    """Rolls a given number of 6-sided dice"""
    return [random.randint(1, 6)]

@mcp.tool 
def add_number(a:float , b: float) ->float:
    """Adds two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()