"""
Main entry point for Exe CLI.
Handles commands: init, start, focus, models, apikeys
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
        console.print("[dim]Use 'exe models' to change models or 'exe apikeys' to update API keys.[/dim]")
        return
    
    from .utils.model_selector import ModelSelector, get_required_providers, get_provider_display_name
    
    # Step 1: Model Selection First
    console.print("\n[bold cyan]Step 1: Model Selection[/bold cyan]")
    console.print("[dim]Select models for your AI assistant.[/dim]\n")
    
    selector = ModelSelector()  # No API key needed for default model list
    models = selector.fetch_available_models()
    
    console.print(f"[green]Found {len(models)} available models[/green]")
    console.print("[dim]Tip: Select 'Refresh from Google' to fetch the latest models dynamically.[/dim]")
    
    # Get recommendations
    recommendations = selector.get_recommended_models(models)
    console.print(f"\n[dim]Recommended configuration:[/dim]")
    console.print(f"  [dim]• Router: {recommendations['router']}[/dim]")
    console.print(f"  [dim]• Coder: {recommendations['coder']}[/dim]")
    console.print(f"  [dim]• Embeddings: text-embedding-004 (Google)[/dim]")
    
    console.print("\n[yellow]Options:[/yellow]")
    console.print("  1. Use recommended models")
    console.print("  2. Choose custom models from list")
    console.print("  3. Refresh models from Google (requires API key)")
    
    try:
        choice = console.input("\nSelect option (1-3): ")
        choice = int(choice)
    except (ValueError, KeyboardInterrupt):
        choice = 1
    
    if choice == 3:
        # Refresh models from Google
        console.print("\n[cyan]Fetching latest models from Google...[/cyan]")
        console.print("[dim]This requires a Google API key to access the live model list.[/dim]")
        temp_api_key = typer.prompt("Enter Google API key (for fetching models)", hide_input=True)
        
        selector = ModelSelector(api_key=temp_api_key)
        models = selector.fetch_available_models()
        console.print(f"[green]✓ Found {len(models)} models from Google![/green]")
        
        # Now let them choose - update recommendations
        recommendations = selector.get_recommended_models(models)
        choice = 2  # Fall through to custom selection
    
    if choice == 2:
        # Router model
        console.print("\n[yellow]Router Model[/yellow] (for fast intent classification):")
        selector.display_models_table(models, "Available Models")
        router_model = selector.select_model_interactive(models, "Select router model")
        
        # Coder model
        console.print("\n[yellow]Coder Model[/yellow] (for code generation):")
        selector.display_models_table(models, "Available Models")
        coder_model = selector.select_model_interactive(models, "Select coder model")
        
        # Embedding model
        console.print("\n[yellow]Embedding Model[/yellow] (for code indexing):")
        embedding_model, embedding_provider = selector.select_embedding_model_interactive()
    else:
        # Use recommended (choice 1 or default)
        router_model = recommendations['router']
        coder_model = recommendations['coder']
        embedding_model = "text-embedding-004"
        embedding_provider = "google"
        console.print(f"[green]✓[/green] Using recommended models")
    
    # Step 2: Determine required providers
    required_providers = get_required_providers(router_model, coder_model, embedding_provider)
    
    console.print(f"\n[bold cyan]Step 2: API Keys[/bold cyan]")
    console.print(f"[dim]Based on your model selection, you need API keys for:[/dim]")
    for provider in sorted(required_providers):
        console.print(f"  • {get_provider_display_name(provider)}")
    
    # Step 3: Collect API keys for required providers
    api_keys = {}
    
    if "google" in required_providers:
        console.print("\n[yellow]Google API Key[/yellow]")
        console.print("[dim]Get yours at: https://aistudio.google.com/apikey[/dim]")
        api_keys["google_api_key"] = typer.prompt("Enter Google API key", hide_input=True)
    
    if "openai" in required_providers:
        console.print("\n[yellow]OpenAI API Key[/yellow]")
        console.print("[dim]Get yours at: https://platform.openai.com/api-keys[/dim]")
        api_keys["openai_api_key"] = typer.prompt("Enter OpenAI API key", hide_input=True)
    
    if "anthropic" in required_providers:
        console.print("\n[yellow]Anthropic API Key[/yellow]")
        console.print("[dim]Get yours at: https://console.anthropic.com/settings/keys[/dim]")
        api_keys["anthropic_api_key"] = typer.prompt("Enter Anthropic API key", hide_input=True)
    
    # Step 4: Additional options
    auto_apply = typer.confirm("\nEnable auto-apply mode? (skips confirmation prompts)", default=False)
    
    # Create and save config
    config = ExeConfig(
        router_model=router_model,
        coder_model=coder_model,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        auto_apply=auto_apply,
        **api_keys
    )
    config_manager.save(config)
    
    console.print("\n[green]✓[/green] Exe initialized successfully!")
    console.print(f"[dim]Config saved to: {config_manager.config_file}[/dim]")
    console.print(f"\n[cyan]Configuration:[/cyan]")
    console.print(f"  • Router Model: {router_model}")
    console.print(f"  • Coder Model: {coder_model}")
    console.print(f"  • Embedding: {embedding_model} ({embedding_provider})")
    console.print(f"  • Auto-apply: {auto_apply}")
    console.print(f"  • API Keys: {', '.join(get_provider_display_name(p) for p in required_providers)}")
    console.print("\n[dim]Run 'exe start' to begin![/dim]")


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
    
    # Setup environment variables for all providers
    config.setup_environment()
    
    db = VectorDB()
    
    # Initialize embedder with correct provider and API key
    embedding_api_key = config.get_api_key_for_provider(config.embedding_provider)
    if not embedding_api_key:
        console.print(f"[red]Error:[/red] No API key configured for {config.embedding_provider}.")
        console.print("[dim]Run 'exe apikeys' to configure API keys.[/dim]")
        raise typer.Exit(1)
    
    embedder = Embedder(
        api_key=embedding_api_key,
        model=config.embedding_model,
        provider=config.embedding_provider
    )
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
    console.print(f"  • Embedding: {config.embedding_model} ({config.embedding_provider})")
    
    change = typer.confirm("\nWould you like to change models?", default=True)
    if not change:
        return
    
    from .utils.model_selector import ModelSelector, get_required_providers, get_provider_display_name
    
    # Offer to refresh from Google
    console.print("\n[yellow]Options:[/yellow]")
    console.print("  1. Choose from default model list")
    console.print("  2. Refresh models from Google (fetch latest)")
    
    try:
        refresh_choice = console.input("\nSelect option (1-2): ")
        refresh_choice = int(refresh_choice)
    except (ValueError, KeyboardInterrupt):
        refresh_choice = 1
    
    try:
        if refresh_choice == 2:
            console.print("\n[cyan]Fetching latest models from Google...[/cyan]")
            # Use existing Google API key if available
            api_key = config.google_api_key
            if not api_key:
                console.print("[dim]Google API key required to fetch live models.[/dim]")
                api_key = typer.prompt("Enter Google API key", hide_input=True)
                config.google_api_key = api_key  # Save for later
            
            selector = ModelSelector(api_key=api_key)
        else:
            selector = ModelSelector()
        
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
        
        # Embedding model selection
        console.print("\n[bold yellow]Embedding Model[/bold yellow] (for code indexing):")
        embedding_model, embedding_provider = selector.select_embedding_model_interactive()
        
        # Check if we need new API keys
        required_providers = get_required_providers(router_model, coder_model, embedding_provider)
        missing_keys = []
        
        for provider in required_providers:
            if not config.get_api_key_for_provider(provider):
                missing_keys.append(provider)
        
        if missing_keys:
            console.print(f"\n[yellow]New API keys required:[/yellow]")
            for provider in missing_keys:
                console.print(f"\n[yellow]{get_provider_display_name(provider)} API Key[/yellow]")
                key = typer.prompt(f"Enter {get_provider_display_name(provider)} API key", hide_input=True)
                if provider == "google":
                    config.google_api_key = key
                elif provider == "openai":
                    config.openai_api_key = key
                elif provider == "anthropic":
                    config.anthropic_api_key = key
        
        # Save configuration
        config.router_model = router_model
        config.coder_model = coder_model
        config.embedding_model = embedding_model
        config.embedding_provider = embedding_provider
        config_manager.save(config)
        
        console.print("\n[green]✓[/green] Models updated successfully!")
        console.print(f"  • Router: {router_model}")
        console.print(f"  • Coder: {coder_model}")
        console.print(f"  • Embedding: {embedding_model} ({embedding_provider})")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def apikeys():
    """Manage API keys for different providers."""
    config_manager = ConfigManager()
    
    if not config_manager.exists():
        console.print("[red]Error:[/red] Exe not initialized. Run 'exe init' first.")
        raise typer.Exit(1)
    
    config = config_manager.load()
    if config is None:
        console.print("[red]Error:[/red] Failed to load config. Please run 'exe init' again.")
        raise typer.Exit(1)
    
    from .utils.model_selector import get_provider_display_name
    
    console.print("[bold cyan]API Key Management[/bold cyan]")
    
    # Show current API key status
    console.print("\n[dim]Current API Keys:[/dim]")
    
    providers = [
        ("google", config.google_api_key),
        ("openai", config.openai_api_key),
        ("anthropic", config.anthropic_api_key)
    ]
    
    for provider, key in providers:
        status = "[green]✓ Configured[/green]" if key else "[red]✗ Not set[/red]"
        console.print(f"  • {get_provider_display_name(provider)}: {status}")
    
    # Show which providers are required for current models
    required = config.get_required_providers()
    console.print(f"\n[dim]Required for current models:[/dim]")
    for provider in sorted(required):
        has_key = config.get_api_key_for_provider(provider) is not None
        status = "[green]✓[/green]" if has_key else "[red]✗ Missing[/red]"
        console.print(f"  • {get_provider_display_name(provider)}: {status}")
    
    # Prompt for action
    console.print("\n[yellow]Options:[/yellow]")
    console.print("  1. Update Google API key")
    console.print("  2. Update OpenAI API key")
    console.print("  3. Update Anthropic API key")
    console.print("  4. Update all configured providers")
    console.print("  5. Cancel")
    
    try:
        choice = console.input("\nSelect option (1-5): ")
        choice = int(choice)
        
        if choice == 1:
            console.print("\n[yellow]Google API Key[/yellow]")
            console.print("[dim]Get yours at: https://aistudio.google.com/apikey[/dim]")
            config.google_api_key = typer.prompt("Enter Google API key", hide_input=True)
            config_manager.save(config)
            console.print("[green]✓[/green] Google API key updated!")
            
        elif choice == 2:
            console.print("\n[yellow]OpenAI API Key[/yellow]")
            console.print("[dim]Get yours at: https://platform.openai.com/api-keys[/dim]")
            config.openai_api_key = typer.prompt("Enter OpenAI API key", hide_input=True)
            config_manager.save(config)
            console.print("[green]✓[/green] OpenAI API key updated!")
            
        elif choice == 3:
            console.print("\n[yellow]Anthropic API Key[/yellow]")
            console.print("[dim]Get yours at: https://console.anthropic.com/settings/keys[/dim]")
            config.anthropic_api_key = typer.prompt("Enter Anthropic API key", hide_input=True)
            config_manager.save(config)
            console.print("[green]✓[/green] Anthropic API key updated!")
            
        elif choice == 4:
            # Update all required providers
            for provider in sorted(required):
                console.print(f"\n[yellow]{get_provider_display_name(provider)} API Key[/yellow]")
                key = typer.prompt(f"Enter {get_provider_display_name(provider)} API key", hide_input=True)
                if provider == "google":
                    config.google_api_key = key
                elif provider == "openai":
                    config.openai_api_key = key
                elif provider == "anthropic":
                    config.anthropic_api_key = key
            
            config_manager.save(config)
            console.print("\n[green]✓[/green] All API keys updated!")
            
        elif choice == 5:
            console.print("[dim]Cancelled.[/dim]")
            return
        else:
            console.print("[red]Invalid choice.[/red]")
            
    except ValueError:
        console.print("[red]Invalid input. Please enter a number.[/red]")
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")


@app.command()
def uninstall():
    """Uninstall Exe and remove all configuration files."""
    import subprocess
    import shutil
    
    console.print("[bold red]Exe Uninstaller[/bold red]")
    console.print("\n[yellow]This will:[/yellow]")
    console.print("  • Remove the exe-ai Python package")
    console.print("  • Delete configuration files (~/.exe)")
    console.print("  • Remove local project data (.exo directory)")
    console.print("\n[dim]This will NOT uninstall Python or any other packages.[/dim]")
    
    confirm = typer.confirm("\nAre you sure you want to uninstall Exe?", default=False)
    if not confirm:
        console.print("[dim]Uninstall cancelled.[/dim]")
        return
    
    errors = []
    
    # Step 1: Remove configuration directory
    config_dir = Path.home() / ".exe"
    if config_dir.exists():
        try:
            shutil.rmtree(config_dir)
            console.print(f"[green]✓[/green] Removed configuration: {config_dir}")
        except Exception as e:
            errors.append(f"Could not remove {config_dir}: {e}")
            console.print(f"[yellow]⚠[/yellow] Could not remove {config_dir}: {e}")
    else:
        console.print(f"[dim]No configuration found at {config_dir}[/dim]")
    
    # Step 2: Remove local .exo directory (if in a project)
    local_exo = Path.cwd() / ".exo"
    if local_exo.exists():
        try:
            shutil.rmtree(local_exo)
            console.print(f"[green]✓[/green] Removed local data: {local_exo}")
        except Exception as e:
            errors.append(f"Could not remove {local_exo}: {e}")
            console.print(f"[yellow]⚠[/yellow] Could not remove {local_exo}: {e}")
    
    # Step 3: Remove lancedb directory (vector database)
    lancedb_dir = Path.cwd() / "lancedb"
    if lancedb_dir.exists():
        try:
            shutil.rmtree(lancedb_dir)
            console.print(f"[green]✓[/green] Removed vector database: {lancedb_dir}")
        except Exception as e:
            errors.append(f"Could not remove {lancedb_dir}: {e}")
            console.print(f"[yellow]⚠[/yellow] Could not remove {lancedb_dir}: {e}")
    
    # Step 4: Uninstall the Python package
    console.print("\n[cyan]Uninstalling exe-ai package...[/cyan]")
    try:
        result = subprocess.run(
            ["pip", "uninstall", "exe-ai", "-y"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            console.print("[green]✓[/green] Uninstalled exe-ai package")
        else:
            # Try pip3
            result = subprocess.run(
                ["pip3", "uninstall", "exe-ai", "-y"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                console.print("[green]✓[/green] Uninstalled exe-ai package")
            else:
                errors.append(f"pip uninstall failed: {result.stderr}")
                console.print(f"[yellow]⚠[/yellow] Could not uninstall package: {result.stderr}")
    except FileNotFoundError:
        errors.append("pip command not found")
        console.print("[yellow]⚠[/yellow] pip not found. Please run: pip uninstall exe-ai")
    
    # Summary
    console.print("\n" + "─" * 40)
    if errors:
        console.print(f"[yellow]Uninstall completed with {len(errors)} warning(s)[/yellow]")
        console.print("[dim]Some items may need manual removal.[/dim]")
    else:
        console.print("[green]✓ Exe has been completely uninstalled![/green]")
    
    console.print("\n[dim]Thank you for using Exe![/dim]")


if __name__ == "__main__":
    app()
