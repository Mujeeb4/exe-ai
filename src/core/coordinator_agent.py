"""
Coordinator Agent - Main orchestrator for Exe.
Delegates tasks to specialized sub-agents based on intent.
"""

from google.adk.agents import LlmAgent
from google.genai import types
from typing import Optional
from .router_agent import create_router_agent
from .coder_agent import create_coder_agent, create_qa_agent
from ..config import ExeConfig


def create_coordinator_agent(config: ExeConfig, repo_context: Optional[str] = None) -> LlmAgent:
    """
    Create the main coordinator agent.
    
    The coordinator orchestrates the workflow:
    1. First delegates to router_agent for intent classification
    2. Based on intent, delegates to appropriate sub-agent
    3. Manages conversation flow and user interaction
    
    Args:
        config: Exe configuration with model settings
        repo_context: Optional repository context/overview
        
    Returns:
        Configured LlmAgent coordinator
    """
    
    # Create specialized sub-agents
    router = create_router_agent(model=config.router_model)
    coder = create_coder_agent(
        model=config.coder_model,
        repo_context=repo_context
    )
    qa = create_qa_agent(
        model=config.coder_model,  # Use same advanced model for Q&A
        repo_context=repo_context
    )
    
    instruction = """You are Exe, an intelligent code assistant that helps developers understand and modify their codebase.

You coordinate between specialized agents to handle different types of requests:

1. **router_agent**: Classifies user intent (question, code_edit, refactor, explain)
2. **coder_agent**: Handles code edits and refactoring
3. **qa_agent**: Answers questions and provides explanations

WORKFLOW:
1. For the first message in a conversation, delegate to router_agent to classify intent
2. Based on the router's classification:
   - For "code_edit" or "refactor" -> delegate to coder_agent
   - For "question" or "explain" -> delegate to qa_agent
3. Maintain conversation context across turns
4. Provide friendly, helpful responses

BEST PRACTICES:
- Always explain what you're doing to the user
- If a task is ambiguous, ask for clarification
- Summarize actions taken by sub-agents
- Maintain conversational tone
- Use the router on each new request to adapt to intent changes

CONVERSATION GUIDELINES:
- Be concise but complete
- Use markdown for code formatting
- Highlight important information
- Provide actionable next steps when relevant
- Ask follow-up questions when needed

EXAMPLE INTERACTIONS:

User: "What does the Router class do?"
You: "Let me search the codebase for you..."
[Delegate to qa_agent]

User: "Fix the bug in main.py line 42"
You: "I'll help you fix that. Let me examine the code first..."
[Delegate to coder_agent]

User: "Refactor the authentication logic"
You: "I'll refactor the authentication code for better structure..."
[Delegate to coder_agent]"""

    if repo_context:
        instruction += f"\n\nREPOSITORY OVERVIEW:\n{repo_context}\n\nUse this to provide context-aware assistance."
    
    return LlmAgent(
        name="exe_coordinator",
        model=config.router_model,  # Use fast model for coordination
        description="Main Exe coordinator that orchestrates code assistance tasks",
        instruction=instruction,
        sub_agents=[router, coder, qa],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.5,  # Balanced for coordination decisions
            max_output_tokens=2048,
            top_p=0.9
        )
    )


def create_simple_agent(config: ExeConfig, repo_context: Optional[str] = None) -> LlmAgent:
    """
    Create a simplified single-agent version (no delegation).
    
    Useful for:
    - Simpler use cases
    - Debugging
    - Lower latency requirements
    
    Args:
        config: Exe configuration
        repo_context: Optional repository context
        
    Returns:
        Single LlmAgent that handles all tasks
    """
    
    from .tools import ALL_TOOLS
    
    instruction = """You are Exe, an intelligent code assistant.

Your capabilities:
1. Answer questions about code
2. Generate code edits using unified diff format
3. Search the codebase
4. Read and list files
5. Apply patches to files

TOOLS AVAILABLE:
- search_code: Find relevant code by semantic search
- read_file: Read file contents
- list_files: List files in a directory
- edit_file: Apply unified diff patches

WORKFLOW:
1. For questions: Use search_code to find context, then answer
2. For edits:
   a. Use search_code to understand context
   b. Use read_file if needed for complete view
   c. Generate unified diff patch
   d. Use edit_file to apply changes
   
UNIFIED DIFF FORMAT:
Always use this format for patches:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
 context line
-removed line
+added line
 context line
```

GUIDELINES:
- Always search before editing
- Be precise and minimal in changes
- Explain your reasoning
- Ask for clarification when needed"""

    if repo_context:
        instruction += f"\n\nREPOSITORY CONTEXT:\n{repo_context}"
    
    return LlmAgent(
        name="exe_simple",
        model=config.coder_model,  # Use advanced model for everything
        description="All-in-one Exe agent for code assistance",
        instruction=instruction,
        tools=ALL_TOOLS,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096
        )
    )
