# Challenge 2 - Tools Agent using Strands SDK + Ollama (llama3.2:3b)
# The agent can use tools to calculate, check weather, and compute age.

from datetime import date
from strands import Agent, tool
from strands.models.ollama import OllamaModel

# ─────────────────────────────────────────────
# TOOL 1: Calculator
# The @tool decorator turns a regular Python function into an agent tool.
# The docstring is important — the agent reads it to know when to use the tool.
# ─────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression and return the result.
    Use this for any arithmetic like addition, subtraction, multiplication, division.
    Example: '10 * 5 + 3' or '100 / 4'
    """
    try:
        result = eval(expression)  # Safe here since the agent controls the input
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ─────────────────────────────────────────────
# TOOL 2: Weather (simulated)
# A real weather tool would call an API like OpenWeatherMap.
# For this challenge we simulate it so no API key is needed.
# ─────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature and conditions for the requested location.
    """
    # Simulated weather data — replace with a real API call if you want
    weather_data = {
        "new york":   "22°C, Partly Cloudy",
        "london":     "15°C, Rainy",
        "tokyo":      "28°C, Sunny",
        "sydney":     "18°C, Windy",
        "paris":      "20°C, Clear skies",
        "dubai":      "38°C, Sunny and Hot",
        "mumbai":     "32°C, Humid",
    }
    key = city.lower().strip()
    if key in weather_data:
        return f"Weather in {city.title()}: {weather_data[key]}"
    return f"Weather data not available for '{city}'. Try: New York, London, Tokyo, Sydney, Paris, Dubai, Mumbai."


# ─────────────────────────────────────────────
# TOOL 3: Age Calculator
# Calculates exact age in years given a birth year, month, and day.
# ─────────────────────────────────────────────
@tool
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> str:
    """
    Calculate a person's current age based on their date of birth.
    Provide birth_year, birth_month (1-12), and birth_day (1-31).
    Example: birth_year=1995, birth_month=6, birth_day=15
    """
    try:
        today = date.today()
        birthday = date(birth_year, birth_month, birth_day)
        age = today.year - birthday.year - (
            (today.month, today.day) < (birthday.month, birthday.day)
        )
        return f"Age: {age} years old (born {birthday.strftime('%B %d, %Y')})"
    except ValueError as e:
        return f"Invalid date: {e}"


# ─────────────────────────────────────────────
# MODEL + AGENT SETUP
# ─────────────────────────────────────────────
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# Pass all three tools to the agent — it will automatically decide which to call
agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use the calculator tool for math, get_weather for weather queries, "
        "and calculate_age for age-related questions. "
        "Always use a tool when the question calls for one."
    ),
    tools=[calculator, get_weather, calculate_age],
)

# ─────────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────────
print("Tools Agent ready! Type 'exit' to quit.")
print("Try: 'What is 25 * 4?', 'Weather in Tokyo', 'How old is someone born on 1995-06-15?'\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    print("Assistant: ", end="", flush=True)
    agent(user_input)
    print()
