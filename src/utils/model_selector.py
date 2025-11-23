"""
Model selection utilities for Google ADK.
Fetches available models and allows user selection.
"""

from typing import List, Dict, Optional
from google import genai
from rich.console import Console
from rich.table import Table

console = Console()


class ModelSelector:
    """Handles fetching and selecting Google ADK models."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    def fetch_available_models(self) -> List[Dict[str, str]]:
        """
        Fetch list of available models from Google ADK.
        
        Returns:
            List of model dictionaries with name, display_name, and description
        """
        try:
            models = []
            
            # Add default models first (includes OpenAI/Anthropic)
            defaults = self._get_default_models()
            models.extend(defaults)
            
            # Try to fetch dynamic Gemini models
            try:
                model_list = self.client.models.list()
                for model in model_list:
                    if hasattr(model, 'supported_generation_methods'):
                        if 'generateContent' in model.supported_generation_methods:
                            # Check if already in defaults to avoid duplicates
                            if not any(m['name'] == model.name for m in models):
                                models.append({
                                    'name': model.name,
                                    'display_name': getattr(model, 'display_name', model.name),
                                    'description': getattr(model, 'description', 'No description available')
                                })
            except Exception as e:
                console.print(f"[yellow]Warning: Could not fetch dynamic Gemini models: {e}[/yellow]")
            
            return models
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch models: {e}[/yellow]")
            # Return default models as fallback
            return self._get_default_models()
    
    def _get_default_models(self) -> List[Dict[str, str]]:
        """Return a list of known default Google models."""
        return [
            {
                'name': 'gemini-2.0-flash-exp',
                'display_name': 'Gemini 2.0 Flash (Experimental)',
                'description': 'Latest experimental model - fast and efficient'
            },
            {
                'name': 'gemini-1.5-flash',
                'display_name': 'Gemini 1.5 Flash',
                'description': 'Fast model optimized for speed'
            },
            {
                'name': 'gemini-1.5-pro',
                'display_name': 'Gemini 1.5 Pro',
                'description': 'Advanced model for complex reasoning'
            },
            {
                'name': 'gpt-4o',
                'display_name': 'GPT-4o (OpenAI)',
                'description': 'OpenAI flagship model'
            },
            {
                'name': 'claude-3-5-sonnet-latest',
                'display_name': 'Claude 3.5 Sonnet (Anthropic)',
                'description': 'Anthropic most intelligent model'
            },
            {
                'name': 'o1-preview',
                'display_name': 'o1 Preview (OpenAI)',
                'description': 'OpenAI reasoning model'
            }
        ]
    
    def display_models_table(self, models: List[Dict[str, str]], title: str = "Available Models"):
        """Display models in a formatted table."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Model Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Description", style="white")
        
        for idx, model in enumerate(models, 1):
            table.add_row(
                str(idx),
                model['name'],
                model['display_name'],
                model['description'][:60] + "..." if len(model['description']) > 60 else model['description']
            )
        
        console.print(table)
    
    def select_model_interactive(self, models: List[Dict[str, str]], prompt: str) -> str:
        """
        Allow user to select a model interactively.
        
        Args:
            models: List of available models
            prompt: Prompt text to display
            
        Returns:
            Selected model name
        """
        while True:
            try:
                choice = console.input(f"\n{prompt} (1-{len(models)}): ")
                idx = int(choice) - 1
                
                if 0 <= idx < len(models):
                    selected = models[idx]['name']
                    console.print(f"[green]✓[/green] Selected: {models[idx]['display_name']}")
                    return selected
                else:
                    console.print(f"[red]Invalid choice. Please enter a number between 1 and {len(models)}[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled. Using default model.[/yellow]")
                return models[0]['name']
    
    def get_recommended_models(self, models: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Get recommended models for different use cases.
        
        Returns:
            Dictionary with 'router' and 'coder' model recommendations
        """
        recommendations = {
            'router': 'gemini-1.5-flash-8b',  # Fast model for simple intent classification
            'coder': 'gemini-2.0-flash-exp'    # Advanced model for code generation
        }
        
        # Verify recommended models exist in available models
        available_names = [m['name'] for m in models]
        
        if recommendations['router'] not in available_names:
            recommendations['router'] = models[0]['name'] if models else 'gemini-2.0-flash-exp'
        
        if recommendations['coder'] not in available_names:
            recommendations['coder'] = models[0]['name'] if models else 'gemini-2.0-flash-exp'
        
        return recommendations
