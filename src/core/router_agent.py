"""
Router Agent using ADK LlmAgent.
Classifies user intent with structured output.
"""

from google.adk.agents import LlmAgent
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal
from src.core.model_factory import ModelFactory


class RouterOutput(BaseModel):
    """Structured output schema for router agent."""
    
    intent: Literal["question", "code_edit", "refactor", "explain"] = Field(
        description="Classified user intent"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )
    suggested_focus: str = Field(
        default="",
        description="Optional suggested focus path based on query content"
    )


def create_router_agent(model: str = "gemini-1.5-flash-8b") -> LlmAgent:
    """
    Create the router agent for intent classification.
    
    Args:
        model: Model name to use for routing (should be fast and cheap)
        
    Returns:
        Configured LlmAgent for routing
    """
    
    instruction = """You are an expert intent classifier for a code assistant named Exe.

Your job is to analyze user queries and classify them into one of these intents:

1. **question**: User is asking about code, architecture, how something works, or wants information
   Examples:
   - "What does this function do?"
   - "How does the authentication flow work?"
   - "Why is this variable named X?"

2. **code_edit**: User wants to modify, fix, add, or change code
   Examples:
   - "Fix the bug in main.py"
   - "Add error handling to the login function"
   - "Update the API endpoint to use POST"

3. **refactor**: User wants to improve code structure without changing behavior
   Examples:
   - "Refactor this to use dependency injection"
   - "Extract this into a separate function"
   - "Clean up this code"

4. **explain**: User wants detailed, educational explanation of specific code
   Examples:
   - "Explain how this algorithm works"
   - "Walk me through this code step by step"
   - "What's the purpose of each parameter?"

IMPORTANT:
- Analyze the query carefully
- Consider the user's wording and intent
- Provide high confidence (0.8+) only when intent is clear
- Use medium confidence (0.5-0.8) for ambiguous queries
- Low confidence (<0.5) means you're uncertain
- Suggest a focus path if the query mentions specific files/directories

Your output will be used to route the query to the appropriate handler."""

    return LlmAgent(
        name="router_agent",
        model=ModelFactory.create_model(model),
        description="Classifies user intent into: question, code_edit, refactor, or explain",
        instruction=instruction,
        output_schema=RouterOutput,  # Pass the Pydantic class directly
        output_key="router_decision",  # Store decision in session state
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,  # Low temperature for consistent classification
            max_output_tokens=500,  # Keep responses short and focused
            top_p=0.9,
            top_k=40
        )
    )


def parse_router_output(event_content: str) -> RouterOutput:
    """
    Parse router agent output into structured RouterOutput.
    
    Args:
        event_content: Raw text output from router agent
        
    Returns:
        Parsed RouterOutput object
    """
    import json
    
    try:
        # Try to parse as JSON
        if event_content.strip().startswith('{'):
            data = json.loads(event_content)
            return RouterOutput(**data)
        else:
            # Fallback: extract intent from text
            content_lower = event_content.lower()
            
            if "code_edit" in content_lower or "modify" in content_lower:
                intent = "code_edit"
                confidence = 0.7
            elif "refactor" in content_lower:
                intent = "refactor"
                confidence = 0.7
            elif "explain" in content_lower:
                intent = "explain"
                confidence = 0.7
            else:
                intent = "question"
                confidence = 0.6
            
            return RouterOutput(
                intent=intent,
                confidence=confidence,
                reasoning="Parsed from unstructured output"
            )
    
    except Exception as e:
        # Default fallback
        return RouterOutput(
            intent="question",
            confidence=0.5,
            reasoning=f"Error parsing output: {str(e)}"
        )
