"""
agno_agent.py

Author: Margalordyzd
Date: 2025-05-22

Description:
    This script demonstrates how to create an agent using the Agno framework with a custom calculator tool.
    The agent is configured to use the LMStudio language model, display data in tables, and output only the report.
    The calculator tool multiplies two input parameters and prints a message when used.
    The agent processes a sample prompt and streams the response with full reasoning and intermediate steps.

Requirements:
    For this code to work, an LMStudio LLM need to be running in the default port: http://127.0.0.1:1234/v1

    The model used in this project is: llama-3.2-1b-instruct

"""
from agno.agent import Agent
from agno.tools.reasoning import ReasoningTools
from agno.models.huggingface import HuggingFace
from agno.models.lmstudio import LMStudio
from agno.tools import tool

@tool(show_result=True, stop_after_tool_call=True)
def calculator(x:int, y:int):
    """Multiply the 2 input parameters.

    Args:
        x (int): First value.
        y (int): Second Value.
    """
    print("Tool has been used")
    return x * y


agent = Agent(
    model=LMStudio(id="llama-3.2-1b-instruct"),
    instructions=[
        "Use tables to display data",
        "Only output the report, no other text",
    ],
    tools=[calculator],
    markdown=True,
)

agent.print_response("use calculator to multiply for 2 and 4", stream=True, show_full_reasoning=True, stream_intermediate_steps=True)