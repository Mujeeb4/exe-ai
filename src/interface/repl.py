"""
Interactive REPL (Read-Eval-Print Loop) for Exe.
Enhanced with command system, error handling, visual feedback, and undo support.
"""

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import shutil
import re
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from .console import console
from ..config import ExeConfig, ConfigManager
from ..memory.db import VectorDB
from ..memory.history import ChatHistory
from ..memory.embedder import Embedder
from ..core.adk_adapters import RouterAdapter, CoderAdapter
from ..ingestion.editor import Editor
from ..ingestion.watcher import FileWatcher
from ..session import session_state


def format_diff_with_colors(diff_content: str) -> Text:
    """
    Format unified diff with color coding.
    Green for additions (+), red for deletions (-), cyan for context.
    
    Args:
        diff_content: Unified diff string
        
    Returns:
        Rich Text object with colored diff
    """
    text = Text()
    
    for line in diff_content.split('\n'):
        if line.startswith('+++') or line.startswith('---'):
            # File headers in bold white
            text.append(line + '\n', style="bold white")
        elif line.startswith('@@'):
            # Hunk headers in cyan
            text.append(line + '\n', style="bold cyan")
        elif line.startswith('+'):
            # Additions in green
            text.append(line + '\n', style="green")
        elif line.startswith('-'):
            # Deletions in red
            text.append(line + '\n', style="red")
        elif line.startswith(' '):
            # Context lines in dim white
            text.append(line + '\n', style="dim white")
        else:
            # Other lines (like diff headers)
            text.append(line + '\n', style="white")
    
    return text


class PatchBackup:
    """Manages file backups for undo functionality."""
    
    def __init__(self, backup_dir: Path = Path(".exo/backups")):
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_history: List[Dict] = []
    
    def create_backup(self, file_path: Path) -> Optional[str]:
        """
        Create a backup of a file before modification.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Backup ID or None if backup failed
        """
        if not file_path.exists():
            return None
        
        try:
            # Create unique backup ID with timestamp
            backup_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{file_path.name}"
            backup_path = self.backup_dir / backup_id
            
            # Copy file to backup
            shutil.copy2(file_path, backup_path)
            
            # Record backup
            self.backup_history.append({
                "backup_id": backup_id,
                "original_path": str(file_path),
                "backup_path": str(backup_path),
                "timestamp": datetime.now().isoformat()
            })
            
            return backup_id
        except Exception as e:
            print(f"⚠ Failed to create backup for {file_path}: {e}")
            return None
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore a file from backup.
        
        Args:
            backup_id: Backup ID to restore
            
        Returns:
            True if restore succeeded
        """
        # Find backup in history
        backup_info = None
        for backup in self.backup_history:
            if backup["backup_id"] == backup_id:
                backup_info = backup
                break
        
        if not backup_info:
            return False
        
        try:
            backup_path = Path(backup_info["backup_path"])
            original_path = Path(backup_info["original_path"])
            
            if not backup_path.exists():
                print(f"✗ Backup file not found: {backup_path}")
                return False
            
            # Restore file
            shutil.copy2(backup_path, original_path)
            return True
        except Exception as e:
            print(f"✗ Failed to restore backup: {e}")
            return False
    
    def get_last_backup(self) -> Optional[Dict]:
        """Get the most recent backup."""
        if not self.backup_history:
            return None
        return self.backup_history[-1]
    
    def get_backup_count(self) -> int:
        """Get total number of backups."""
        return len(self.backup_history)
    
    def cleanup_old_backups(self, keep_last: int = 10):
        """
        Remove old backups, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent backups to keep
        """
        if len(self.backup_history) <= keep_last:
            return
        
        to_remove = self.backup_history[:-keep_last]
        for backup in to_remove:
            try:
                Path(backup["backup_path"]).unlink(missing_ok=True)
                self.backup_history.remove(backup)
            except Exception as e:
                print(f"⚠ Failed to cleanup backup: {e}")


class ReplSession:
    """Manages REPL session state and statistics."""
    
    def __init__(self, config: ExeConfig, session_id: int):
        self.config = config
        self.session_id = session_id
        self.modified_files: List[str] = []
        self.query_count = 0
        self.patch_count = 0
        self.auto_apply_enabled = config.auto_apply
        
        # Undo support
        self.backup_manager = PatchBackup()
        self.patch_history: List[Dict] = []  # Track applied patches with metadata
    
    def record_query(self):
        """Record a user query."""
        self.query_count += 1
    
    def record_patch(self, file_path: str):
        """Record a patched file."""
        self.patch_count += 1
        if file_path not in self.modified_files:
            self.modified_files.append(file_path)
    
    def record_patch_batch(self, files: List[str], backup_ids: List[str]):
        """
        Record a batch of successfully applied patches with their backups.
        
        Args:
            files: List of modified file paths
            backup_ids: List of backup IDs for the files
        """
        self.patch_history.append({
            "timestamp": datetime.now().isoformat(),
            "files": files,
            "backup_ids": backup_ids
        })
    
    def undo_last_patch(self) -> bool:
        """
        Undo the last applied patch batch.
        
        Returns:
            True if undo succeeded
        """
        if not self.patch_history:
            console.print("[yellow]⚠ No patches to undo[/yellow]")
            return False
        
        # Get last patch batch
        last_patch = self.patch_history.pop()
        
        console.print(f"\n[cyan]⏮ Undoing last patch ({len(last_patch['files'])} files)...[/cyan]")
        
        # Restore backups
        success_count = 0
        for backup_id in last_patch["backup_ids"]:
            if self.backup_manager.restore_backup(backup_id):
                success_count += 1
        
        if success_count == len(last_patch["backup_ids"]):
            console.print(f"[green]✓ Successfully undid patch ({success_count} files restored)[/green]")
            
            # Update stats
            self.patch_count -= len(last_patch["files"])
            for file_path in last_patch["files"]:
                if file_path in self.modified_files:
                    self.modified_files.remove(file_path)
            
            return True
        else:
            console.print(f"[red]✗ Partial undo ({success_count}/{len(last_patch['backup_ids'])} files restored)[/red]")
            return False
    
    def toggle_auto_apply(self) -> bool:
        """Toggle auto-apply mode and return new state."""
        self.auto_apply_enabled = not self.auto_apply_enabled
        
        # Persist to config
        config_manager = ConfigManager()
        config = config_manager.load()
        config.auto_apply = self.auto_apply_enabled
        config_manager.save(config)
        
        return self.auto_apply_enabled


def show_startup_info(config: ExeConfig, session_id: int):
    """Display startup information."""
    console.print("[bold green]Exe is ready![/bold green]", end=" ")
    console.print("[dim]Type /help for commands or 'exit' to quit.[/dim]\n")
    
    # Configuration panel
    info_lines = []
    info_lines.append(f"[cyan]Session ID:[/cyan] {session_id}")
    info_lines.append(f"[cyan]Router Model:[/cyan] {config.router_model}")
    info_lines.append(f"[cyan]Coder Model:[/cyan] {config.coder_model}")
    
    if config.focus_path:
        info_lines.append(f"[cyan]Focus Mode:[/cyan] {config.focus_path}")
    else:
        info_lines.append(f"[cyan]Focus Mode:[/cyan] [dim]Off (searching entire codebase)[/dim]")
    
    if config.auto_apply:
        info_lines.append(f"[cyan]Auto-apply:[/cyan] [yellow]ON[/yellow] [dim](patches applied without confirmation)[/dim]")
    else:
        info_lines.append(f"[cyan]Auto-apply:[/cyan] [dim]OFF[/dim]")
    
    panel = Panel(
        "\n".join(info_lines),
        title="[bold]Session Info[/bold]",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def handle_command(
    command: str,
    repl_session: ReplSession,
    config: ExeConfig,
    router: RouterAdapter,
    history: ChatHistory,
    watcher: FileWatcher
) -> bool:
    """
    Handle special REPL commands.
    
    Args:
        command: Command string (starts with /)
        repl_session: REPL session instance
        config: Exe config
        router: Router instance
        history: Chat history instance
        watcher: File watcher instance
        
    Returns:
        True if should continue REPL loop, False to exit
    """
    parts = command[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    
    if cmd == "help":
        show_help()
    
    elif cmd == "focus":
        if arg:
            try:
                # Validate path exists
                focus_path = Path(arg)
                if not focus_path.is_absolute():
                    focus_path = Path.cwd() / focus_path
                
                if not focus_path.exists():
                    console.print(f"[red]✗ Path does not exist:[/red] {arg}")
                    return True
                
                # Update session state
                session_state.set_focus(arg)
                
                # Update router
                router.focus_path = session_state.get_focus()
                
                # Update config
                config.focus_path = session_state.get_focus()
                ConfigManager().save(config)
                
                console.print(f"[green]✓ Focus set to:[/green] {router.focus_path}")
                console.print("[dim]Vector search will be restricted to this path[/dim]")
            except Exception as e:
                console.print(f"[red]✗ Error setting focus:[/red] {e}")
        else:
            console.print("[yellow]Usage:[/yellow] /focus <path>")
            console.print("[dim]Example: /focus src/core[/dim]")
    
    elif cmd in ["clear-focus", "unfocus", "clearfocus"]:
        session_state.clear_focus()
        router.focus_path = None
        config.focus_path = None
        ConfigManager().save(config)
        console.print("[green]✓ Focus cleared[/green]")
        console.print("[dim]Searching entire codebase[/dim]")
    
    elif cmd == "stats" or cmd == "status":
        show_stats(repl_session, watcher)
    
    elif cmd == "history":
        show_history(history, repl_session.session_id, arg)
    
    elif cmd in ["toggle-auto-apply", "auto", "autoapply"]:
        new_state = repl_session.toggle_auto_apply()
        config.auto_apply = new_state
        ConfigManager().save(config)
        
        if new_state:
            console.print("[yellow]⚠ Auto-apply:[/yellow] [green]ON[/green]")
            console.print("[dim]Patches will be applied automatically without confirmation[/dim]")
        else:
            console.print("[cyan]Auto-apply:[/cyan] [dim]OFF[/dim]")
            console.print("[dim]You will be asked to confirm before applying patches[/dim]")
    
    elif cmd == "undo":
        repl_session.undo_last_patch()
    
    elif cmd == "clear" or cmd == "cls":
        console.clear()
        show_startup_info(config, repl_session.session_id)
    
    elif cmd == "exit" or cmd == "quit":
        return False
    
    else:
        console.print(f"[red]✗ Unknown command:[/red] /{cmd}")
        console.print("[dim]Type /help for available commands[/dim]")
    
    return True


def show_help():
    """Display help message with available commands."""
    help_table = Table(title="Available Commands", show_header=True, border_style="cyan")
    help_table.add_column("Command", style="cyan", no_wrap=True)
    help_table.add_column("Description", style="white")
    
    help_table.add_row("/help", "Show this help message")
    help_table.add_row("/focus <path>", "Set focus to specific path")
    help_table.add_row("/clear-focus", "Clear focus (search entire codebase)")
    help_table.add_row("/stats", "Show session statistics")
    help_table.add_row("/history [n]", "Show conversation history (last n messages)")
    help_table.add_row("/toggle-auto-apply", "Toggle auto-apply mode on/off")
    help_table.add_row("/undo", "Undo the last applied patch")
    help_table.add_row("/clear", "Clear screen and show session info")
    help_table.add_row("exit or quit", "Exit Exe")
    
    console.print(help_table)
    console.print("\n[dim]Tip: Commands can be abbreviated (e.g., /h for /help)[/dim]")


def show_stats(repl_session: ReplSession, watcher: FileWatcher):
    """Display session statistics."""
    table = Table(title="Session Statistics", show_header=False, border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Session ID", str(repl_session.session_id))
    table.add_row("Queries", str(repl_session.query_count))
    table.add_row("Patches Applied", str(repl_session.patch_count))
    table.add_row("Files Modified", str(len(repl_session.modified_files)))
    table.add_row("Backups Created", str(repl_session.backup_manager.get_backup_count()))
    table.add_row("Undo Stack Size", str(len(repl_session.patch_history)))
    table.add_row("Watcher Status", "Running" if watcher.is_running() else "Stopped")
    table.add_row("Focus Mode", repl_session.config.focus_path or "[dim]Off[/dim]")
    table.add_row("Auto-apply", "ON" if repl_session.auto_apply_enabled else "OFF")
    
    console.print(table)
    
    if repl_session.modified_files:
        console.print("\n[yellow]Modified files this session:[/yellow]")
        for f in repl_session.modified_files:
            console.print(f"  [dim]•[/dim] {f}")


def show_history(history: ChatHistory, session_id: int, limit_arg: Optional[str]):
    """Display conversation history."""
    try:
        limit = int(limit_arg) if limit_arg else 10
    except ValueError:
        console.print(f"[red]✗ Invalid limit:[/red] {limit_arg}")
        return
    
    messages = history.get_session_messages(session_id, limit=limit)
    
    if not messages:
        console.print("[dim]No conversation history yet[/dim]")
        return
    
    console.print(f"\n[bold]Last {len(messages)} messages:[/bold]\n")
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."
        
        if role == "user":
            console.print(f"[bold cyan]You:[/bold cyan] {content}")
        else:
            console.print(f"[bold green]Exe:[/bold green] {content}")
        console.print()


def extract_file_paths_from_patch(patch_content: str) -> List[str]:
    """
    Extract file paths from unified diff patch.
    
    Args:
        patch_content: Unified diff content
        
    Returns:
        List of file paths mentioned in the patch
    """
    file_paths = []
    for line in patch_content.splitlines():
        if line.startswith('---') or line.startswith('+++'):
            # Extract path from diff header
            parts = line.split()
            if len(parts) >= 2:
                path = parts[1]
                # Remove a/ or b/ prefix if present
                if path.startswith('a/') or path.startswith('b/'):
                    path = path[2:]
                if path not in file_paths:
                    file_paths.append(path)
    return file_paths


def start_repl(config: ExeConfig, db: VectorDB, repo_context: Optional[str] = None):
    """
    Start the interactive REPL with enhanced features.
    
    Args:
        config: Exe configuration
        db: Vector database instance
        repo_context: Complete repository context string
    """
    # Initialize components
    embedder = Embedder(config.api_key)
    router = RouterAdapter(config.api_key, config.focus_path, repo_context, model=config.router_model)
    coder = CoderAdapter(config.api_key, repo_context, model=config.coder_model)
    editor = Editor()
    history = ChatHistory()
    
    # Create session
    session_id = history.create_session(str(Path.cwd()))
    repl_session = ReplSession(config, session_id)
    
    # Show startup info
    show_startup_info(config, session_id)
    
    # Start file watcher
    watcher = FileWatcher(Path.cwd(), db, embedder, editor)
    watcher.start()
    
    try:
        while True:
            # Get user input
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            
            # Handle exit commands
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Handle empty input
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input.startswith('/'):
                should_continue = handle_command(
                    user_input,
                    repl_session,
                    config,
                    router,
                    history,
                    watcher
                )
                if not should_continue:
                    break
                continue
            
            # Record query
            repl_session.record_query()
            
            # Save user message
            history.add_message(session_id, "user", user_input)
            
            try:
                # Route query
                console.print("[dim]🤔 Analyzing query...[/dim]")
                router_output = router.route(user_input, db, embedder)
                
                # Get relevant chunks
                chunks = router_output.relevant_chunks or []
                
                console.print(f"[dim]📚 Found {len(chunks)} relevant code chunks[/dim]")
                
                # Retrieve recent conversation context
                conversation_history = history.get_session_messages(session_id, limit=6)
                
                # Generate response
                console.print("[dim]✨ Generating response...[/dim]\n")
                coder_output = coder.process(user_input, router_output, chunks, conversation_history)
                
                # Display response
                console.print("[bold green]Exe:[/bold green]")
                
                if coder_output.type == "answer":
                    # Display markdown answer
                    console.print(Markdown(coder_output.content))
                
                elif coder_output.type == "patch":
                    # Display diff with rich color formatting
                    formatted_diff = format_diff_with_colors(coder_output.content)
                    console.print(Panel(formatted_diff, title="[bold]Proposed Changes[/bold]", border_style="yellow"))
                    
                    # Extract file paths from patch
                    files_to_modify = extract_file_paths_from_patch(coder_output.content)
                    if not files_to_modify and coder_output.files_to_modify:
                        files_to_modify = coder_output.files_to_modify
                    
                    # Ask for confirmation (unless auto_apply is on)
                    if repl_session.auto_apply_enabled:
                        apply = True
                        console.print("\n[yellow]⚡ Auto-apply enabled - applying patch...[/yellow]")
                    else:
                        apply = console.input("\n[yellow]Apply this patch?[/yellow] [y/N]: ").lower() == 'y'
                    
                    if apply:
                        success_count = 0
                        fail_count = 0
                        backup_ids = []
                        modified_files = []
                        failed_files = []
                        
                        # Apply patch to files
                        for file_path in files_to_modify:
                            # Create backup before patching
                            backup_id = repl_session.backup_manager.create_backup(Path(file_path))
                            
                            result = editor.apply_patch_with_details(Path(file_path), coder_output.content)
                            
                            if result["success"]:
                                console.print(f"[green]✓ Patched:[/green] {file_path}")
                                repl_session.record_patch(file_path)
                                modified_files.append(file_path)
                                if backup_id:
                                    backup_ids.append(backup_id)
                                success_count += 1
                            else:
                                error_msg = result.get("error", "Unknown error")
                                console.print(f"[red]✗ Failed to patch:[/red] {file_path}")
                                console.print(f"[dim]  Reason: {error_msg}[/dim]")
                                failed_files.append((file_path, error_msg))
                                fail_count += 1
                        
                        # Record patch batch for undo
                        if modified_files and backup_ids:
                            repl_session.record_patch_batch(modified_files, backup_ids)
                        
                        # Summary
                        if success_count > 0:
                            console.print(f"\n[green]✓ Successfully patched {success_count} file(s)[/green]")
                            console.print(f"[dim]💾 Backups created ({len(backup_ids)} files) - use /undo to revert[/dim]")
                        if fail_count > 0:
                            console.print(f"\n[red]✗ Failed to patch {fail_count} file(s)[/red]")
                            console.print("\n[yellow]Failure details:[/yellow]")
                            for file_path, error_msg in failed_files:
                                console.print(f"  [red]•[/red] {file_path}: {error_msg}")
                            console.print("\n[dim]💡 Tip: File may have been modified externally. Try regenerating the patch.[/dim]")
                    else:
                        console.print("[dim]Patch not applied[/dim]")
                
                # Save assistant message
                history.add_message(session_id, "assistant", coder_output.content[:1000])  # Truncate for storage
            
            except Exception as e:
                console.print(f"\n[red]✗ Error:[/red] {e}")
                console.print("[dim]The error has been logged. Please try rephrasing your query.[/dim]")
            
            console.print()  # Blank line for readability
    
    finally:
        # Cleanup and show summary
        watcher.stop()
        
        if repl_session.modified_files:
            console.print(f"\n[yellow]📝 Modified {len(repl_session.modified_files)} file(s) this session:[/yellow]")
            for f in repl_session.modified_files:
                console.print(f"  [dim]•[/dim] {f}")
        
        console.print(f"\n[dim]Total queries: {repl_session.query_count}[/dim]")
        console.print(f"[dim]Total patches: {repl_session.patch_count}[/dim]")
        
        history.close()
        console.print("\n[bold blue]👋 Goodbye![/bold blue]")
