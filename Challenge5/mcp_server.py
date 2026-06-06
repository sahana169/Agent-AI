# Challenge 5 - Local MCP Server
#
# This file defines a simple MCP (Model Context Protocol) server with 3 tools.
# The server runs as a subprocess and communicates via stdio (standard input/output).
# The Strands agent connects to it using MCPClient.
#
# ─── SETUP (run once) ─────────────────────────────────────────────────────────
#   pip install "strands-agents[ollama]" "mcp"
# ──────────────────────────────────────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP
from datetime import date, datetime
import math

# FastMCP creates an MCP-compliant server with minimal boilerplate.
# "LocalToolsServer" is the server name shown in logs.
mcp = FastMCP("LocalToolsServer")


# ─── TOOL 1: Calculator ───────────────────────────────────────────────────────
# @mcp.tool() registers this function as an MCP tool.
# The docstring becomes the tool description the agent reads.
@mcp.tool()
def calculator(expression: str) -> str:
    """
    Evaluate a math expression and return the result.
    Use for arithmetic like addition, subtraction, multiplication, division, powers.
    Example: '10 * 5 + 3' or '2 ** 10'
    """
    try:
        # Allow safe math functions like sqrt, pi, etc.
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_globals)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# ─── TOOL 2: Get Current Date & Time ─────────────────────────────────────────
@mcp.tool()
def get_datetime(timezone: str = "local") -> str:
    """
    Get the current date and time.
    Returns today's date, current time, and day of the week.
    """
    now = datetime.now()
    return (
        f"Date: {now.strftime('%B %d, %Y')}\n"
        f"Time: {now.strftime('%I:%M %p')}\n"
        f"Day:  {now.strftime('%A')}"
    )


# ─── TOOL 3: Age Calculator ───────────────────────────────────────────────────
@mcp.tool()
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> str:
    """
    Calculate a person's age from their date of birth.
    Provide birth_year, birth_month (1-12), and birth_day (1-31).
    Example: birth_year=1995, birth_month=6, birth_day=15
    """
    try:
        today    = date.today()
        birthday = date(birth_year, birth_month, birth_day)
        age = today.year - birthday.year - (
            (today.month, today.day) < (birthday.month, birthday.day)
        )
        return f"Age: {age} years old (born {birthday.strftime('%B %d, %Y')})"
    except ValueError as e:
        return f"Invalid date: {e}"


# ─── TOOL 4: Unit Converter ───────────────────────────────────────────────────
@mcp.tool()
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    Supported conversions:
      Temperature: celsius↔fahrenheit, celsius↔kelvin
      Distance:    km↔miles, meters↔feet
      Weight:      kg↔pounds
    Example: value=100, from_unit='celsius', to_unit='fahrenheit'
    """
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    conversions = {
        ("celsius",    "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"):    lambda v: (v - 32) * 5/9,
        ("celsius",    "kelvin"):     lambda v: v + 273.15,
        ("kelvin",     "celsius"):    lambda v: v - 273.15,
        ("km",         "miles"):      lambda v: v * 0.621371,
        ("miles",      "km"):         lambda v: v / 0.621371,
        ("meters",     "feet"):       lambda v: v * 3.28084,
        ("feet",       "meters"):     lambda v: v / 3.28084,
        ("kg",         "pounds"):     lambda v: v * 2.20462,
        ("pounds",     "kg"):         lambda v: v / 2.20462,
    }

    fn = conversions.get((f, t))
    if fn:
        result = round(fn(value), 4)
        return f"{value} {from_unit} = {result} {to_unit}"
    return f"Conversion from '{from_unit}' to '{to_unit}' not supported."


# Entry point — run the MCP server via stdio transport
# stdio means the server reads/writes JSON messages through stdin/stdout
# This is how Strands MCPClient communicates with it as a subprocess
if __name__ == "__main__":
    mcp.run(transport="stdio")
