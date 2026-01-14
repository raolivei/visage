"""
Visage Shared Package

Common types, prompts, and utilities shared across services.
"""

from .prompts import (
    STYLE_PRESETS,
    GENERATION_DEFAULTS,
    LORA_TRAINING_DEFAULTS,
    QUALITY_THRESHOLDS,
    get_prompt_for_style,
    get_available_styles,
)

__all__ = [
    "STYLE_PRESETS",
    "GENERATION_DEFAULTS",
    "LORA_TRAINING_DEFAULTS",
    "QUALITY_THRESHOLDS",
    "get_prompt_for_style",
    "get_available_styles",
]
