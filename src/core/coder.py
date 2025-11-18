"""
Coder Agent: High-level intelligence.
Generates answers or unified diffs for code modifications.
"""

import openai
from typing import List
from .models import CoderOutput, RouterOutput, CodeChunk


class Coder:
    """Generates intelligent responses and code patches."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def process(self, query: str, router_output: RouterOutput, chunks: List[CodeChunk]) -> CoderOutput:
        """
        Generate a response based on router output and retrieved context.
        
        Args:
            query: User's input query
            router_output: Output from Router agent
            chunks: Relevant code chunks from vector DB
            
        Returns:
            CoderOutput with answer or patch
        """
        # Build context from chunks
        context = "\n\n".join([
            f"# {chunk.file_path}:{chunk.start_line}-{chunk.end_line}\n{chunk.content}"
            for chunk in chunks
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
                {"role": "user", "content": f"Context:\n{context}\n\nTask: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            patch_content = response.choices[0].message.content or ""
            
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
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            return CoderOutput(
                type="answer",
                content=response.choices[0].message.content or "No response generated."
            )
