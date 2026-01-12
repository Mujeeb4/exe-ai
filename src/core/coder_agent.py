"""
Coder Agent using ADK LlmAgent.
Generates code edits and applies patches using tools.
"""

from google.adk.agents import LlmAgent
from google.genai import types
from typing import Optional
from .tools import edit_file_tool, search_code_tool, read_file_tool, list_files_tool
from src.core.model_factory import ModelFactory


def create_coder_agent(
    model: str = "gemini-2.0-flash-exp",
    repo_context: Optional[str] = None
) -> LlmAgent:
    """
    Create the coder agent for code generation and editing.
    
    Args:
        model: Model name to use (should be advanced for quality)
        repo_context: Optional repository context/overview
        
    Returns:
        Configured LlmAgent for coding tasks
    """
    
    base_instruction = """You are an expert code generation and editing assistant named Exe.

IMPORTANT: You have access to tools to explore the codebase. ALWAYS use tools to get context before answering.

Your capabilities:
1. Generate high-quality code edits using UNIFIED DIFF format
2. Search the codebase using the `search_code` tool
3. Read file contents using the `read_file` tool
4. List available files using the `list_files` tool
5. Apply changes using the `edit_file` tool

WORKFLOW FOR ANY REQUEST:
1. **FIRST use `search_code`** to find relevant context for the user's question
2. If needed, use `read_file` to see complete file contents
3. For code changes, generate a unified diff patch
4. Use `edit_file` tool to apply the patch
5. Explain what you found/changed

WORKFLOW FOR CODE EDITS:
1. Search for the code to understand current state
2. Generate a unified diff patch with this EXACT format:
   ```
   --- a/path/to/file.py
   +++ b/path/to/file.py
   @@ -start,count +start,count @@
    unchanged line
   -removed line
   +added line
    unchanged line
   ```
3. Use `edit_file` tool to apply the patch
4. Explain what you changed and why

CRITICAL RULES:
- **ALWAYS use search_code FIRST** - don't guess about the codebase
- **NEVER** output full file content - ONLY unified diffs
- Include 3 lines of context before and after changes
- Make minimal, focused changes
- Ask for clarification if the request is ambiguous
- Use tools systematically - don't skip steps

UNIFIED DIFF FORMAT REQUIREMENTS:
- Start with `--- a/filepath` and `+++ b/filepath`
- Use `@@ -old_start,old_count +new_start,new_count @@` headers
- Prefix unchanged lines with single space ` `
- Prefix removed lines with minus `-`
- Prefix added lines with plus `+`
- Include enough context for patch to apply cleanly

HANDLING DIFFERENT INTENTS:
- **code_edit**: Search context, generate patches, apply with edit_file
- **refactor**: Improve code structure while preserving behavior
- **question**: Search context FIRST, then provide clear explanations
- **explain**: Search for code, provide detailed walkthroughs"""

    # Add minimal repo context if provided (should be lightweight)
    if repo_context:
        base_instruction += f"\n\nREPOSITORY OVERVIEW:\n{repo_context}\n\nThis is a high-level overview. Use search_code and read_file tools to get detailed information."
    
    return LlmAgent(
        name="coder_agent",
        model=ModelFactory.create_model(model),
        description="Generates code edits, applies patches, and answers coding questions",
        instruction=base_instruction,
        tools=[
            search_code_tool,
            read_file_tool,
            list_files_tool,
            edit_file_tool
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7,  # Balanced for creativity + accuracy
            max_output_tokens=4096,  # Allow longer code generations
            top_p=0.95,
            top_k=40
        )
    )


def create_qa_agent(
    model: str = "gemini-2.0-flash-exp",
    repo_context: Optional[str] = None
) -> LlmAgent:
    """
    Create a specialized Q&A agent for answering questions.
    
    This agent is optimized for questions and explanations,
    without code editing capabilities.
    
    Args:
        model: Model name to use
        repo_context: Optional repository context/overview
        
    Returns:
        Configured LlmAgent for Q&A tasks
    """
    
    base_instruction = """You are a helpful code assistant focused on answering questions.

Your capabilities:
1. Search the codebase using `search_code` tool
2. Read file contents using `read_file` tool
3. List available files using `list_files` tool
4. Provide clear, accurate explanations

WORKFLOW:
1. Use `search_code` to find relevant code
2. Use `read_file` if you need complete file contents
3. Analyze the code thoroughly
4. Provide a clear, well-structured answer

GUIDELINES:
- Be accurate and cite specific files/line numbers
- Use code snippets in your explanations (in markdown)
- Explain complex concepts simply
- Admit when you're uncertain
- Suggest follow-up questions when helpful"""

    if repo_context:
        base_instruction += f"\n\nREPOSITORY CONTEXT:\n{repo_context}"
    
    return LlmAgent(
        name="qa_agent",
        model=ModelFactory.create_model(model),
        description="Answers questions about code and provides explanations",
        instruction=base_instruction,
        tools=[
            search_code_tool,
            read_file_tool,
            list_files_tool
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=3072,
            top_p=0.95
        )
    )
