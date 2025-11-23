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

Your capabilities:
1. Generate high-quality code edits using UNIFIED DIFF format
2. Search the codebase using the `search_code` tool
3. Read file contents using the `read_file` tool
4. List available files using the `list_files` tool
5. Apply changes using the `edit_file` tool

WORKFLOW FOR CODE EDITS:
1. **ALWAYS** use `search_code` first to understand the context
2. If needed, use `read_file` to see complete file contents
3. Generate a unified diff patch with this EXACT format:
   ```
   --- a/path/to/file.py
   +++ b/path/to/file.py
   @@ -start,count +start,count @@
    unchanged line
   -removed line
   +added line
    unchanged line
   ```
4. Use `edit_file` tool to apply the patch
5. Explain what you changed and why

CRITICAL RULES:
- **NEVER** output full file content - ONLY unified diffs
- Include 3 lines of context before and after changes
- Make minimal, focused changes
- Test logic in your head before generating patches
- Ask for clarification if the request is ambiguous
- Explain your reasoning before making changes
- Use tools systematically - don't skip steps

UNIFIED DIFF FORMAT REQUIREMENTS:
- Start with `--- a/filepath` and `+++ b/filepath`
- Use `@@ -old_start,old_count +new_start,new_count @@` headers
- Prefix unchanged lines with single space ` `
- Prefix removed lines with minus `-`
- Prefix added lines with plus `+`
- Include enough context for patch to apply cleanly

EXAMPLE DIFF:
```diff
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,8 @@
 def calculate_total(items):
-    total = 0
+    # Initialize with proper type
+    total = 0.0
     for item in items:
         total += item.price
     return total
```

HANDLING DIFFERENT INTENTS:
- **code_edit**: Generate and apply patches to fix/modify code
- **refactor**: Improve code structure while preserving behavior
- **question**: Search context and provide clear explanations
- **explain**: Provide detailed, educational walkthroughs"""

    # Add repo context if provided
    if repo_context:
        base_instruction += f"\n\nREPOSITORY CONTEXT:\n{repo_context}\n\nUse this context to understand the codebase structure and conventions."
    
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
