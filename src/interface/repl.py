"""
Interactive REPL (Read-Eval-Print Loop) for Exo.
"""

from pathlib import Path
from typing import Optional
from rich.markdown import Markdown
from rich.syntax import Syntax
from .console import console
from ..config import ExeConfig
from ..memory.db import VectorDB
from ..memory.history import ChatHistory
from ..memory.embedder import Embedder
from ..core.router import Router
from ..core.coder import Coder
from ..ingestion.editor import Editor
from ..ingestion.watcher import FileWatcher


def start_repl(config: ExeConfig, db: VectorDB, repo_context: Optional[str] = None):
    """
    Start the interactive REPL.
    
    Args:
        config: Exe configuration
        db: Vector database instance
        repo_context: Complete repository context string
    """
    console.print("[bold green]Exe is ready![/bold green] Type 'exit' to quit.\n")
    
    # Initialize components with repository context
    embedder = Embedder(config.api_key)
    router = Router(config.api_key, config.focus_path, repo_context, model=config.router_model)
    coder = Coder(config.api_key, repo_context, model=config.coder_model)
    editor = Editor()
    history = ChatHistory()
    
    # Create session
    session_id = history.create_session(str(Path.cwd()))
    
    # Start file watcher
    watcher = FileWatcher(Path.cwd(), db, embedder, editor)
    watcher.start()
    
    try:
        while True:
            # Get user input
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                break
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue
            
            # Save user message
            history.add_message(session_id, "user", user_input)
            
            # Route query
            console.print("[dim]Thinking...[/dim]")
            router_output = router.route(user_input, db, embedder)
            
            # Get relevant chunks
            chunks = router_output.relevant_chunks or []
            
            # Retrieve recent conversation context
            conversation_history = history.get_session_messages(session_id, limit=6)
            
            # Generate response
            coder_output = coder.process(user_input, router_output, chunks, conversation_history)
            
            # Display response
            console.print("\n[bold green]Exe:[/bold green]")
            
            if coder_output.type == "answer":
                # Display markdown answer
                console.print(Markdown(coder_output.content))
            
            elif coder_output.type == "patch":
                # Display diff with syntax highlighting
                syntax = Syntax(coder_output.content, "diff", theme="monokai")
                console.print(syntax)
                
                # Ask for confirmation (unless auto_apply is on)
                if config.auto_apply:
                    apply = True
                else:
                    apply = console.input("\n[yellow]Apply this patch?[/yellow] [y/N]: ").lower() == 'y'
                
                if apply:
                    # Apply patch to files
                    for file_path in coder_output.files_to_modify or []:
                        success = editor.apply_patch(Path(file_path), coder_output.content)
                        
                        if success:
                            console.print(f"[green]✓[/green] Patched {file_path}")
                        else:
                            console.print(f"[red]✗[/red] Failed to patch {file_path}")
            
            # Save assistant message
            history.add_message(session_id, "assistant", coder_output.content)
            
            console.print()  # Blank line for readability
    
    finally:
        # Cleanup
        watcher.stop()
        history.close()
        console.print("\n[bold blue]Goodbye![/bold blue]")
