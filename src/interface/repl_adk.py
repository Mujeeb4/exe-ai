"""
Interactive REPL for Exe using ADK Runner.
Provides real-time streaming feedback and event visualization.
"""

from pathlib import Path
from typing import Optional
import asyncio
import time
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.console import Group
from .console import console
from ..config import ExeConfig
from ..core.runner import ExeRunner


async def start_adk_repl(config: ExeConfig, repo_context: Optional[str] = None):
    """
    Start the interactive REPL using ADK runner with event streaming.
    
    Args:
        config: Exe configuration
        repo_context: Complete repository context string
    """
    console.print("[bold green]Exe (ADK) is ready![/bold green] Type 'exit' to quit.\n")
    console.print("[dim]Using Google Agent Development Kit for enhanced intelligence[/dim]\n")
    
    # Create runner
    runner = ExeRunner(
        config=config,
        repo_context=repo_context,
        use_database=True,  # Persist sessions
        use_simple_agent=False  # Use multi-agent architecture
    )
    
    # Create session
    user_id = "local_user"
    session_id = f"repl_{int(time.time())}"
    
    await runner.create_session(user_id, session_id)
    
    console.print(f"[dim]Session ID: {session_id}[/dim]")
    if config.focus_path:
        console.print(f"[dim]Focus: {config.focus_path}[/dim]")
    console.print()
    
    try:
        turn = 0
        while True:
            # Get user input
            try:
                user_input = await asyncio.to_thread(
                    console.input, "[bold cyan]You:[/bold cyan] "
                )
            except (EOFError, KeyboardInterrupt):
                break
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Handle special commands
            if user_input.startswith('/'):
                await handle_command(user_input, runner, user_id, session_id, config)
                continue
            
            if not user_input.strip():
                continue
            
            turn += 1
            
            # Stream response with live updates
            await stream_response(
                runner=runner,
                user_id=user_id,
                session_id=session_id,
                query=user_input,
                turn=turn
            )
            
            console.print()  # Add spacing
    
    finally:
        # Cleanup
        modified_files = await runner.get_modified_files(user_id, session_id)
        if modified_files:
            console.print(f"\n[yellow]Modified {len(modified_files)} file(s) this session:[/yellow]")
            for f in modified_files:
                console.print(f"  - {f}")
        
        await runner.close_session(user_id, session_id)
        console.print("\n[dim]Session closed. Goodbye![/dim]")


async def stream_response(
    runner: ExeRunner,
    user_id: str,
    session_id: str,
    query: str,
    turn: int
):
    """
    Stream response with rich live updates.
    
    Args:
        runner: ExeRunner instance
        user_id: User ID
        session_id: Session ID
        query: User query
        turn: Conversation turn number
    """
    # State tracking
    current_agent = "exe"
    current_tool = None
    response_text = ""
    tool_calls = []
    
    with Live(
        Panel(
            Spinner("dots", text="Thinking..."),
            title=f"[bold green]Exe[/bold green] (Turn {turn})",
            border_style="green"
        ),
        console=console,
        refresh_per_second=10
    ) as live:
        
        async for event in runner.query(user_id, session_id, query):
            
            # Agent started
            if hasattr(event, 'type') and event.type == "agent_start":
                current_agent = event.author
                live.update(Panel(
                    f"[cyan]Agent:[/cyan] {current_agent}\n[dim]Processing...[/dim]",
                    title=f"[bold green]Exe[/bold green]",
                    border_style="cyan"
                ))
            
            # Tool call started
            elif hasattr(event, 'type') and event.type == "tool_call":
                current_tool = getattr(event, 'tool_name', 'unknown')
                tool_args = getattr(event, 'tool_args', {})
                
                tool_calls.append({
                    "tool": current_tool,
                    "args": tool_args
                })
                
                # Create tool display
                tool_display = f"[yellow]🔧 Tool:[/yellow] {current_tool}\n"
                if tool_args:
                    tool_display += "[dim]Arguments:[/dim]\n"
                    for key, value in tool_args.items():
                        # Truncate long values
                        val_str = str(value)
                        if len(val_str) > 100:
                            val_str = val_str[:100] + "..."
                        tool_display += f"  • {key}: {val_str}\n"
                
                live.update(Panel(
                    tool_display,
                    title=f"[bold green]Exe[/bold green] - Tool Execution",
                    border_style="yellow"
                ))
            
            # Tool result
            elif hasattr(event, 'type') and event.type == "tool_result":
                result = getattr(event, 'result', 'Done')
                live.update(Panel(
                    f"[green]✓ Tool completed:[/green] {current_tool}\n[dim]{str(result)[:200]}...[/dim]" if len(str(result)) > 200 else f"[green]✓ Tool completed:[/green] {current_tool}\n{result}",
                    title=f"[bold green]Exe[/bold green]",
                    border_style="green"
                ))
            
            # Text delta (streaming response)
            elif hasattr(event, 'type') and event.type == "text_delta":
                delta = getattr(event, 'content', '')
                response_text += delta
                
                # Update with accumulated response
                live.update(Panel(
                    response_text + "▌",  # Add cursor
                    title=f"[bold green]Exe[/bold green]",
                    border_style="green"
                ))
            
            # Final response
            elif event.is_final_response():
                response_text = event.content.parts[0].text
                
                # Render final response
                try:
                    # Try to render as markdown
                    rendered = Markdown(response_text)
                except:
                    # Fallback to plain text
                    rendered = response_text
                
                # Add tool summary if tools were used
                if tool_calls:
                    tool_summary = create_tool_summary(tool_calls)
                    rendered = Group(rendered, tool_summary)
                
                live.update(Panel(
                    rendered,
                    title=f"[bold green]Exe[/bold green] (Turn {turn})",
                    border_style="green"
                ))


def create_tool_summary(tool_calls: list) -> Table:
    """
    Create a summary table of tool calls.
    
    Args:
        tool_calls: List of tool call dictionaries
        
    Returns:
        Rich Table with tool summary
    """
    table = Table(title="Tools Used", show_header=True, header_style="bold magenta")
    table.add_column("Tool", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    # Count tool usage
    tool_counts = {}
    for tc in tool_calls:
        tool = tc["tool"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    for tool, count in tool_counts.items():
        table.add_row(tool, str(count))
    
    return table


async def handle_command(
    command: str,
    runner: ExeRunner,
    user_id: str,
    session_id: str,
    config: ExeConfig
):
    """
    Handle special REPL commands.
    
    Args:
        command: Command string (starts with /)
        runner: ExeRunner instance
        user_id: User ID
        session_id: Session ID
        config: Exe config
    """
    parts = command[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    
    if cmd == "focus":
        if arg:
            await runner.update_focus(user_id, session_id, arg)
            console.print(f"[green]Focus set to:[/green] {arg}")
        else:
            console.print("[yellow]Usage:[/yellow] /focus <path>")
    
    elif cmd == "clear-focus" or cmd == "unfocus":
        await runner.update_focus(user_id, session_id, None)
        console.print("[green]Focus cleared[/green]")
    
    elif cmd == "status":
        session = await runner.get_session(user_id, session_id)
        if session:
            focus = session.state.get("focus_path", "Not set")
            modified = session.state.get("modified_files", [])
            
            table = Table(title="Session Status", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Session ID", session_id)
            table.add_row("Focus Path", focus)
            table.add_row("Modified Files", str(len(modified)))
            table.add_row("Agent Mode", "Multi-Agent (Coordinator)")
            
            console.print(table)
            
            if modified:
                console.print("\n[yellow]Modified files:[/yellow]")
                for f in modified:
                    console.print(f"  - {f}")
    
    elif cmd == "help":
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("/focus <path>", "Set focus to specific path")
        help_table.add_row("/clear-focus", "Clear focus (search entire codebase)")
        help_table.add_row("/status", "Show session status")
        help_table.add_row("/help", "Show this help message")
        help_table.add_row("exit", "Exit Exe")
        
        console.print(help_table)
    
    else:
        console.print(f"[red]Unknown command:[/red] /{cmd}")
        console.print("[dim]Type /help for available commands[/dim]")


def start_repl(config: ExeConfig, db, repo_context: Optional[str] = None):
    """
    Legacy sync wrapper for backward compatibility.
    Starts the ADK async REPL.
    
    Args:
        config: Exe configuration
        db: Vector database (not used in ADK version)
        repo_context: Repository context string
    """
    asyncio.run(start_adk_repl(config, repo_context))
