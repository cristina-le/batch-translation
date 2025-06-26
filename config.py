"""
Configuration file for batch translation system.
"""

# Default configuration
DEFAULT_CONFIG = {
    # File paths
    'jp_file': "app/data/batch_jp.txt",
    'output_file': "app/data/batch_output.txt",
    'en_ref_file': "app/data/batch_en.txt",
    
    # Translation settings
    'translator_version': 2,  # Version 1 or 2
    'chunk_size': 20,
    'speaker_aware': True,
    
    # Model settings (will use defaults if not specified)
    'model': None,  # None = use default for version
    'temperature': None,  # None = use default for version
    'context_window': None,  # None = use default for version
    'quality_threshold': 8.5,  # Only for version 2
}

# Version-specific defaults
VERSION_DEFAULTS = {
    1: {
        'model': "google/gemini-2.0-flash-001",
        'temperature': 0.2,
        'context_window': 3,
    },
    2: {
        'model': "openai/gpt-4o",
        'temperature': 0.05,
        'context_window': 7,
        'quality_threshold': 8.5,
    }
}

# Preset configurations for different use cases
PRESETS = {
    'fast': {
        'translator_version': 1,
        'model': "google/gemini-2.0-flash-001",
        'temperature': 0.1,
        'chunk_size': 15,
        'speaker_aware': False,
        'context_window': 6,
    },
    'quality': {
        'translator_version': 2,
        'model': "google/gemini-2.0-flash-001",
        'temperature': 0.1,
        'chunk_size': 15,
        'speaker_aware': False,
        'quality_threshold': 9.0,
        'context_window': 6,
    }
}


def get_config(preset=None, **overrides):
    """
    Get configuration with optional preset and overrides.
    
    Args:
        preset: Name of preset configuration ('fast', 'quality')
        **overrides: Any configuration values to override
        
    Returns:
        Dictionary with final configuration
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Apply preset if specified
    if preset and preset in PRESETS:
        config.update(PRESETS[preset])
    
    # Apply any overrides
    config.update(overrides)
    
    # Apply version-specific defaults for None values
    version = config['translator_version']
    if version in VERSION_DEFAULTS:
        defaults = VERSION_DEFAULTS[version]
        for key, value in defaults.items():
            if config.get(key) is None:
                config[key] = value
    
    return config
