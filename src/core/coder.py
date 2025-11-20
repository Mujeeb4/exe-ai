"""
Coder Agent: High-level intelligence.
Generates answers or unified diffs for code modifications.
"""

import openai
from typing import List, Optional, Dict
from .models import CoderOutput, RouterOutput, CodeChunk


class Coder:
    """Generates intelligent responses and code patches."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def process(self, query: str, router_output: RouterOutput, chunks: List[CodeChunk], 
                conversation_history: Optional[List[Dict[str, str]]] = None) -> CoderOutput:
        """
        Generate a response based on router output and retrieved context.
        
        Args:
            query: User's input query
            router_output: Output from Router agent
            chunks: Relevant code chunks from vector DB
            conversation_history: Optional recent conversation messages for context
            
        Returns:
            CoderOutput with answer or patch
        """
        # Build context from chunks
        context = "\n\n".join([
            f"# {chunk.file_path}:{chunk.start_line}-{chunk.end_line}\n{chunk.content}"
            for chunk in chunks
        ])
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nRecent conversation:\n" + "\n".join([
                f"{msg['role'].upper()}: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history[-6:]  # Last 3 turns (6 messages)
            ])
        
        if router_output.intent in ["code_edit", "refactor"]:
            # Generate unified diff
            system_prompt = """You are a code modification expert. Generate ONLY a unified diff patch.
Never output full file content. Use this format:
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
 unchanged line
-removed line
+added line
 unchanged line"""
            
            from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
            
            messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}{conversation_context}\n\nTask: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            
            patch_content = response.choices[0].message.content or ""
            
            # Strip markdown code blocks if present
            if patch_content.startswith("```diff"):
                patch_content = patch_content.replace("```diff", "").replace("```", "").strip()
            elif patch_content.startswith("```"):
                patch_content = patch_content.replace("```", "").strip()
            
            return CoderOutput(
                type="patch",
                content=patch_content,
                files_to_modify=router_output.relevant_files
            )
        else:
            # Generate natural language answer
            from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
            
            messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
                {"role": "system", "content": "You are a helpful coding assistant. Provide clear, concise answers."},
                {"role": "user", "content": f"Context:\n{context}{conversation_context}\n\nQuestion: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            
            return CoderOutput(
                type="answer",
                content=response.choices[0].message.content or "No response generated."
            )
