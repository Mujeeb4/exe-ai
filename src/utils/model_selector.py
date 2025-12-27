"""
Model selection utilities for Google ADK.
Fetches available models, allows user selection, and validates model+API key combinations.
"""

from typing import List, Dict, Optional, Set, Tuple
from rich.console import Console
from rich.table import Table

console = Console()

# Provider constants
PROVIDER_PREFIXES = {
    "google": ["gemini-"],
    "openai": ["gpt-", "o1-", "chatgpt-"],
    "anthropic": ["claude-"]
}

# Embedding models by provider
EMBEDDING_MODELS = {
    "google": [
        {
            "name": "text-embedding-004",
            "display_name": "Google Text Embedding",
            "description": "Google's latest embedding model (768 dims)"
        }
    ],
    "openai": [
        {
            "name": "text-embedding-3-small",
            "display_name": "OpenAI Small",
            "description": "Fast and efficient embeddings (1536 dims)"
        },
        {
            "name": "text-embedding-3-large",
            "display_name": "OpenAI Large",
            "description": "High-quality embeddings (3072 dims)"
        }
    ]
}


def get_provider_for_model(model_name: str) -> str:
    """Determine provider from model name."""
    model_lower = model_name.lower()
    if model_lower.startswith("gemini-") or "gemini" in model_lower:
        return "google"
    elif model_lower.startswith(("gpt-", "o1-", "o3-", "chatgpt-")) or "gpt" in model_lower:
        return "openai"
    elif model_lower.startswith("claude-") or "claude" in model_lower:
        return "anthropic"
    return "unknown"


def get_required_providers(router_model: str, coder_model: str, embedding_provider: str) -> Set[str]:
    """Get set of providers needed based on selected models."""
    providers = set()
    providers.add(get_provider_for_model(router_model))
    providers.add(get_provider_for_model(coder_model))
    providers.add(embedding_provider)
    providers.discard("unknown")  # Remove unknown if present
    return providers


def get_provider_display_name(provider: str) -> str:
    """Get human-readable name for provider."""
    names = {
        "google": "Google (Gemini)",
        "openai": "OpenAI",
        "anthropic": "Anthropic (Claude)"
    }
    return names.get(provider, provider.title())


def validate_google_model(model_name: str, api_key: str) -> Tuple[bool, str]:
    """Validate Google model by making a test API call."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        # Quick test with minimal tokens
        response = model.generate_content("Say OK", 
            generation_config={"max_output_tokens": 5})
        return True, "Model validated successfully"
    except Exception as e:
        return False, str(e)


def validate_openai_model(model_name: str, api_key: str) -> Tuple[bool, str]:
    """Validate OpenAI model by making a test API call."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        # Quick test with minimal tokens
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=1
        )
        return True, "Model validated successfully"
    except Exception as e:
        return False, str(e)


def validate_anthropic_model(model_name: str, api_key: str) -> Tuple[bool, str]:
    """Validate Anthropic model by making a test API call."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Quick test with minimal tokens
        client.messages.create(
            model=model_name,
            max_tokens=1,
            messages=[{"role": "user", "content": "OK"}]
        )
        return True, "Model validated successfully"
    except Exception as e:
        return False, str(e)


def validate_model(model_name: str, api_key: str, provider: str = None) -> Tuple[bool, str]:
    """Validate model+API key combination by making a test call."""
    if provider is None:
        provider = get_provider_for_model(model_name)
    
    console.print(f"[cyan]Testing {model_name}...[/cyan]")
    
    if provider == "google":
        return validate_google_model(model_name, api_key)
    elif provider == "openai":
        return validate_openai_model(model_name, api_key)
    elif provider == "anthropic":
        return validate_anthropic_model(model_name, api_key)
    else:
        return False, f"Unknown provider for model: {model_name}"


def fetch_google_models_live(api_key: str) -> List[Dict[str, str]]:
    """Fetch live models from Google Generative AI API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        models = []
        for model in genai.list_models():
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    models.append({
                        'name': model.name.replace('models/', ''),
                        'display_name': getattr(model, 'display_name', model.name),
                        'description': getattr(model, 'description', '')[:80],
                        'provider': 'google'
                    })
        return models
    except Exception as e:
        console.print(f"[yellow]Could not fetch Google models: {e}[/yellow]")
        return []


def fetch_openai_models_live(api_key: str) -> List[Dict[str, str]]:
    """Fetch live models from OpenAI API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        models = []
        for model in client.models.list():
            # Filter to only chat/completion models
            if any(x in model.id for x in ['gpt', 'o1', 'o3', 'chatgpt']):
                models.append({
                    'name': model.id,
                    'display_name': model.id,
                    'description': f"OpenAI model (owned by {model.owned_by})",
                    'provider': 'openai'
                })
        return sorted(models, key=lambda x: x['name'], reverse=True)[:15]  # Top 15
    except Exception as e:
        console.print(f"[yellow]Could not fetch OpenAI models: {e}[/yellow]")
        return []


def fetch_anthropic_models_live(api_key: str) -> List[Dict[str, str]]:
    """Fetch known Anthropic models (they don't have a list API, so use known models)."""
    # Anthropic doesn't have a public list models API, so we use known latest models
    return [
        {
            'name': 'claude-sonnet-4-20250514',
            'display_name': 'Claude Sonnet 4',
            'description': 'Latest Claude Sonnet model',
            'provider': 'anthropic'
        },
        {
            'name': 'claude-3-5-sonnet-latest',
            'display_name': 'Claude 3.5 Sonnet',
            'description': 'Most intelligent Claude model',
            'provider': 'anthropic'
        },
        {
            'name': 'claude-3-5-haiku-latest',
            'display_name': 'Claude 3.5 Haiku',
            'description': 'Fast and efficient Claude model',
            'provider': 'anthropic'
        },
        {
            'name': 'claude-3-opus-latest',
            'display_name': 'Claude 3 Opus',
            'description': 'Most powerful Claude 3 model',
            'provider': 'anthropic'
        }
    ]




class ModelSelector:
    """Handles fetching and selecting Google ADK models."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception:
                pass  # Client creation failed, will use defaults
    
    def fetch_available_models(self) -> List[Dict[str, str]]:
        """
        Fetch list of available models from Google ADK.
        
        Returns:
            List of model dictionaries with name, display_name, description, and provider
        """
        try:
            models = []
            
            # Add default models first (includes OpenAI/Anthropic)
            defaults = self._get_default_models()
            models.extend(defaults)
            
            # Try to fetch dynamic Gemini models if we have an API key
            if self.client:
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
                                        'description': getattr(model, 'description', 'No description available'),
                                        'provider': 'google'
                                    })
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not fetch dynamic Gemini models: {e}[/yellow]")
            
            return models
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch models: {e}[/yellow]")
            # Return default models as fallback
            return self._get_default_models()
    
    def _get_default_models(self) -> List[Dict[str, str]]:
        """Return a list of known default models with provider info."""
        return [
            {
                'name': 'gemini-2.0-flash-exp',
                'display_name': 'Gemini 2.0 Flash (Experimental)',
                'description': 'Latest experimental model - fast and efficient',
                'provider': 'google'
            },
            {
                'name': 'gemini-1.5-flash',
                'display_name': 'Gemini 1.5 Flash',
                'description': 'Fast model optimized for speed',
                'provider': 'google'
            },
            {
                'name': 'gemini-1.5-pro',
                'display_name': 'Gemini 1.5 Pro',
                'description': 'Advanced model for complex reasoning',
                'provider': 'google'
            },
            {
                'name': 'gpt-4o',
                'display_name': 'GPT-4o (OpenAI)',
                'description': 'OpenAI flagship model',
                'provider': 'openai'
            },
            {
                'name': 'claude-3-5-sonnet-latest',
                'display_name': 'Claude 3.5 Sonnet (Anthropic)',
                'description': 'Anthropic most intelligent model',
                'provider': 'anthropic'
            },
            {
                'name': 'o1-preview',
                'display_name': 'o1 Preview (OpenAI)',
                'description': 'OpenAI reasoning model',
                'provider': 'openai'
            }
        ]
    
    def get_embedding_models(self) -> List[Dict[str, str]]:
        """Get available embedding models."""
        models = []
        for provider, provider_models in EMBEDDING_MODELS.items():
            for model in provider_models:
                models.append({
                    **model,
                    'provider': provider
                })
        return models
    
    def display_models_table(self, models: List[Dict[str, str]], title: str = "Available Models"):
        """Display models in a formatted table."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Model Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Provider", style="yellow")
        table.add_column("Description", style="white")
        
        for idx, model in enumerate(models, 1):
            provider = model.get('provider', get_provider_for_model(model['name']))
            table.add_row(
                str(idx),
                model['name'],
                model['display_name'],
                get_provider_display_name(provider),
                model['description'][:50] + "..." if len(model['description']) > 50 else model['description']
            )
        
        console.print(table)
    
    def display_embedding_models_table(self, title: str = "Embedding Models"):
        """Display embedding models in a formatted table."""
        models = self.get_embedding_models()
        self.display_models_table(models, title)
        return models
    
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
    
    def select_embedding_model_interactive(self) -> tuple:
        """
        Allow user to select an embedding model interactively.
        
        Returns:
            Tuple of (model_name, provider)
        """
        models = self.get_embedding_models()
        self.display_models_table(models, "Embedding Models")
        
        selected_name = self.select_model_interactive(models, "Select embedding model")
        
        # Find provider for selected model
        for model in models:
            if model['name'] == selected_name:
                return selected_name, model['provider']
        
        return selected_name, "google"
    
    def get_recommended_models(self, models: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Get recommended models for different use cases.
        
        Returns:
            Dictionary with 'router' and 'coder' model recommendations
        """
        recommendations = {
            'router': 'gemini-2.0-flash-exp',  # Fast model for simple intent classification
            'coder': 'gemini-2.0-flash-exp'    # Advanced model for code generation
        }
        
        # Verify recommended models exist in available models
        available_names = [m['name'] for m in models]
        
        if recommendations['router'] not in available_names:
            recommendations['router'] = models[0]['name'] if models else 'gemini-2.0-flash-exp'
        
        if recommendations['coder'] not in available_names:
            recommendations['coder'] = models[0]['name'] if models else 'gemini-2.0-flash-exp'
        
        return recommendations
    
    def enter_custom_model_interactive(self, role: str = "model") -> Tuple[str, str]:
        """
        Allow user to enter a custom model name manually.
        
        Args:
            role: Description of what the model is for (e.g., "router", "coder")
        
        Returns:
            Tuple of (model_name, provider)
        """
        console.print(f"\n[bold yellow]Enter Custom {role.title()} Model[/bold yellow]")
        console.print("[dim]Examples: gemini-2.5-pro, gpt-4.5-turbo, claude-4-sonnet[/dim]")
        
        model_name = console.input("\nModel name: ").strip()
        
        if not model_name:
            console.print("[yellow]No model entered. Using default.[/yellow]")
            return "gemini-2.0-flash-exp", "google"
        
        # Detect provider
        provider = get_provider_for_model(model_name)
        
        if provider == "unknown":
            console.print("\n[yellow]Could not auto-detect provider. Please select:[/yellow]")
            console.print("  1. Google (Gemini)")
            console.print("  2. OpenAI (GPT)")
            console.print("  3. Anthropic (Claude)")
            try:
                choice = int(console.input("Select provider (1-3): "))
                provider = {1: "google", 2: "openai", 3: "anthropic"}.get(choice, "google")
            except (ValueError, KeyboardInterrupt):
                provider = "google"
        
        console.print(f"[green]✓[/green] Model: {model_name} (Provider: {get_provider_display_name(provider)})")
        return model_name, provider
    
    def fetch_all_live_models(self, api_keys: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Fetch live models from all providers where API keys are available.
        
        Args:
            api_keys: Dict with provider names as keys and API keys as values
                      e.g., {"google": "...", "openai": "...", "anthropic": "..."}
        
        Returns:
            Combined list of models from all available providers
        """
        all_models = []
        
        if api_keys.get("google"):
            console.print("[dim]Fetching Google models...[/dim]")
            google_models = fetch_google_models_live(api_keys["google"])
            all_models.extend(google_models)
            console.print(f"[green]✓[/green] Found {len(google_models)} Google models")
        
        if api_keys.get("openai"):
            console.print("[dim]Fetching OpenAI models...[/dim]")
            openai_models = fetch_openai_models_live(api_keys["openai"])
            all_models.extend(openai_models)
            console.print(f"[green]✓[/green] Found {len(openai_models)} OpenAI models")
        
        if api_keys.get("anthropic"):
            console.print("[dim]Fetching Anthropic models...[/dim]")
            anthropic_models = fetch_anthropic_models_live(api_keys["anthropic"])
            all_models.extend(anthropic_models)
            console.print(f"[green]✓[/green] Found {len(anthropic_models)} Anthropic models")
        
        return all_models

