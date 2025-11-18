"""
Main entry point for Exe CLI.
Handles commands: init, start, focus
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from .config import ConfigManager, ExeConfig
from .interface.console import console
from .interface.repl import start_repl
from .ingestion.scanner import Scanner
from .memory.db import VectorDB
from .session import session_state

app = typer.Typer(help="Exe: Local-First Agentic Code Assistant")


@app.command()
def init():
    """Initialize Exe in the current directory."""
    console.print("[bold blue]Initializing Exe...[/bold blue]")
    
    config_manager = ConfigManager()
    
    if config_manager.exists():
        console.print("[yellow]Exe is already initialized.[/yellow]")
        return
    
    # Prompt for API key
    api_key = typer.prompt("Enter your OpenAI API key", hide_input=True)
    auto_apply = typer.confirm("Enable auto-apply mode? (skips confirmation prompts)", default=False)
    
    config = ExeConfig(api_key=api_key, auto_apply=auto_apply)
    config_manager.save(config)
    
    console.print("[green]✓[/green] Exe initialized successfully!")
    console.print(f"Config saved to: {config_manager.config_file}")


@app.command()
def start():
    """Start Exe and begin indexing the codebase."""
    config_manager = ConfigManager()
    
    if not config_manager.exists():
        console.print("[red]Error:[/red] Exe is not initialized. Run 'exe init' first.")
        raise typer.Exit(1)
    
    config = config_manager.load()
    if config is None:
        console.print("[red]Error:[/red] Failed to load config. Please run 'exe init' again.")
        raise typer.Exit(1)
    
    console.print("[bold blue]Starting Exe...[/bold blue]")
    
    # Initialize session state
    session_state.active = True
    session_state.project_root = Path.cwd()
    
    # Load focus path from config if set
    if config.focus_path:
        try:
            session_state.set_focus(config.focus_path)
            console.print(f"[cyan]Focus mode:[/cyan] {config.focus_path}")
        except ValueError as e:
            console.print(f"[yellow]Warning:[/yellow] {e}")
            console.print("[yellow]Clearing invalid focus path from config.[/yellow]")
            config.focus_path = None
            config_manager.save(config)
    
    # Initialize components
    from .memory.embedder import Embedder
    
    db = VectorDB()
    embedder = Embedder(config.api_key)
    scanner = Scanner()
    
    # Scan and index codebase
    console.print("Indexing codebase...")
    scanner.scan_and_index(Path.cwd(), db, embedder)
    
    console.print("[green]✓[/green] Indexing complete!")
    
    # Start REPL
    start_repl(config, db)


@app.command()
def focus(path: str):
    """Set focus mode to a specific directory or file."""
    config_manager = ConfigManager()
    
    if not config_manager.exists():
        console.print("[red]Error:[/red] Exo not initialized. Run 'exo init' first.")
        raise typer.Exit(1)
    
    # Load config first
    config = config_manager.load()
    if config is None:
        console.print("[red]Error:[/red] Failed to load config. Please run 'exe init' again.")
        raise typer.Exit(1)
    
    # Validate path exists
    focus_path = Path(path)
    if not focus_path.is_absolute():
        focus_path = Path.cwd() / focus_path
    
    if not focus_path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)
    
    # Save to config
    config.focus_path = str(focus_path.relative_to(Path.cwd()))
    config_manager.save(config)
    
    console.print(f"[green]✓[/green] Focus mode set to: {config.focus_path}")
    console.print("[dim]This will take effect on next 'exe start'[/dim]")


@app.command()
def clear_focus():
    """Clear focus mode and search the entire codebase."""
    config_manager = ConfigManager()
    
    if not config_manager.exists():
        console.print("[red]Error:[/red] Exe not initialized. Run 'exe init' first.")
        raise typer.Exit(1)
    
    config = config_manager.load()
    if config is None:
        console.print("[red]Error:[/red] Failed to load config. Please run 'exe init' again.")
        raise typer.Exit(1)
    
    config.focus_path = None
    config_manager.save(config)
    
    console.print("[green]✓[/green] Focus mode cleared")
    console.print("[dim]Exe will search the entire codebase[/dim]")


if __name__ == "__main__":
    app()
