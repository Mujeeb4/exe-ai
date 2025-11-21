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
    api_key = typer.prompt("Enter your Google API key (for Google ADK)", hide_input=True)
    auto_apply = typer.confirm("Enable auto-apply mode? (skips confirmation prompts)", default=False)
    
    # Model selection
    console.print("\n[bold cyan]Model Selection[/bold cyan]")
    console.print("Fetching available models from Google ADK...")
    
    from .utils.model_selector import ModelSelector
    
    try:
        selector = ModelSelector(api_key)
        models = selector.fetch_available_models()
        
        if models:
            console.print(f"\n[green]Found {len(models)} available models[/green]")
            
            # Get recommendations
            recommendations = selector.get_recommended_models(models)
            console.print(f"\n[dim]Recommended configuration:[/dim]")
            console.print(f"  [dim]• Router (intent classification): {recommendations['router']}[/dim]")
            console.print(f"  [dim]• Coder (code generation): {recommendations['coder']}[/dim]")
            
            use_custom = typer.confirm("\nWould you like to choose custom models?", default=False)
            
            if use_custom:
                # Display models and let user choose
                selector.display_models_table(models, "Available Models")
                
                console.print("\n[yellow]Router Model[/yellow] (for fast intent classification):")
                router_model = selector.select_model_interactive(models, "Select router model")
                
                console.print("\n[yellow]Coder Model[/yellow] (for code generation and answers):")
                selector.display_models_table(models, "Available Models")
                coder_model = selector.select_model_interactive(models, "Select coder model")
            else:
                router_model = recommendations['router']
                coder_model = recommendations['coder']
                console.print(f"[green]✓[/green] Using recommended models")
        else:
            # Fallback to defaults
            router_model = "gemini-1.5-flash-8b"
            coder_model = "gemini-2.0-flash-exp"
            console.print("[yellow]Using default models[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch models ({e}). Using defaults.[/yellow]")
        router_model = "gemini-1.5-flash-8b"
        coder_model = "gemini-2.0-flash-exp"
    
    config = ExeConfig(
        api_key=api_key,
        auto_apply=auto_apply,
        router_model=router_model,
        coder_model=coder_model
    )
    config_manager.save(config)
    
    console.print("\n[green]✓[/green] Exe initialized successfully!")
    console.print(f"[dim]Config saved to: {config_manager.config_file}[/dim]")
    console.print(f"\n[cyan]Configuration:[/cyan]")
    console.print(f"  • Router Model: {router_model}")
    console.print(f"  • Coder Model: {coder_model}")
    console.print(f"  • Auto-apply: {auto_apply}")


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
    repo_context = scanner.scan_and_index(Path.cwd(), db, embedder, build_repo_context=True)
    
    console.print("[green]✓[/green] Indexing complete!")
    console.print("[green]✓[/green] Repository context built!")
    
    # Start REPL with full repository context
    start_repl(config, db, repo_context)


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


@app.command()
def models():
    """Configure AI models for router and coder."""
    config_manager = ConfigManager()
    
    if not config_manager.exists():
        console.print("[red]Error:[/red] Exe not initialized. Run 'exe init' first.")
        raise typer.Exit(1)
    
    config = config_manager.load()
    if config is None:
        console.print("[red]Error:[/red] Failed to load config. Please run 'exe init' again.")
        raise typer.Exit(1)
    
    console.print("[bold cyan]Model Configuration[/bold cyan]")
    console.print(f"\n[dim]Current models:[/dim]")
    console.print(f"  • Router: {config.router_model}")
    console.print(f"  • Coder: {config.coder_model}")
    
    change = typer.confirm("\nWould you like to change models?", default=True)
    if not change:
        return
    
    console.print("\nFetching available models...")
    
    from .utils.model_selector import ModelSelector
    
    try:
        selector = ModelSelector(config.api_key)
        available_models = selector.fetch_available_models()
        
        if not available_models:
            console.print("[yellow]Could not fetch models. Using defaults.[/yellow]")
            return
        
        console.print(f"[green]Found {len(available_models)} available models[/green]\n")
        
        # Router model selection
        console.print("[bold yellow]Router Model[/bold yellow] (for fast intent classification):")
        selector.display_models_table(available_models, "Available Models")
        router_model = selector.select_model_interactive(available_models, "Select router model")
        
        # Coder model selection
        console.print("\n[bold yellow]Coder Model[/bold yellow] (for code generation and answers):")
        selector.display_models_table(available_models, "Available Models")
        coder_model = selector.select_model_interactive(available_models, "Select coder model")
        
        # Save configuration
        config.router_model = router_model
        config.coder_model = coder_model
        config_manager.save(config)
        
        console.print("\n[green]✓[/green] Models updated successfully!")
        console.print(f"  • Router: {router_model}")
        console.print(f"  • Coder: {coder_model}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
