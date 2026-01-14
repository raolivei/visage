"""
Visage Prompt Templates

Curated prompts for professional headshot generation.
"""

# Style presets with their prompts
STYLE_PRESETS = {
    "corporate": {
        "name": "Corporate / LinkedIn",
        "description": "Professional business headshot suitable for LinkedIn and corporate profiles",
        "prompt": (
            "professional corporate headshot portrait of {trigger}, "
            "wearing business attire, neutral background, "
            "soft studio lighting, sharp focus on face, "
            "confident expression, high resolution, "
            "LinkedIn profile photo style, clean and polished"
        ),
        "negative_prompt": (
            "cartoon, anime, illustration, painting, drawing, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "casual clothing, outdoor, busy background"
        ),
    },
    "studio": {
        "name": "Studio Portrait",
        "description": "Classic studio portrait with dramatic lighting",
        "prompt": (
            "professional studio portrait of {trigger}, "
            "dramatic Rembrandt lighting, dark background, "
            "high contrast, sharp focus, professional photography, "
            "elegant pose, confident expression, "
            "magazine quality, 8k resolution"
        ),
        "negative_prompt": (
            "cartoon, anime, illustration, flat lighting, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "outdoor, natural light, busy background"
        ),
    },
    "natural": {
        "name": "Natural Light",
        "description": "Warm, approachable headshot with natural lighting",
        "prompt": (
            "natural light portrait of {trigger}, "
            "golden hour lighting, soft bokeh background, "
            "warm tones, genuine smile, approachable expression, "
            "outdoor professional headshot, sharp focus on eyes, "
            "high resolution, authentic and friendly"
        ),
        "negative_prompt": (
            "cartoon, anime, illustration, harsh lighting, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "studio, artificial light, dark, moody"
        ),
    },
    "executive": {
        "name": "Executive",
        "description": "Premium executive portrait for C-suite and leadership",
        "prompt": (
            "executive portrait of {trigger}, "
            "premium business attire, sophisticated background, "
            "professional studio lighting, authoritative presence, "
            "confident and distinguished expression, "
            "Fortune 500 executive style, high-end corporate photography, "
            "sharp focus, impeccable quality"
        ),
        "negative_prompt": (
            "cartoon, anime, illustration, casual, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "young, inexperienced, informal, outdoor"
        ),
    },
    "creative": {
        "name": "Creative Professional",
        "description": "Modern, creative headshot for designers and artists",
        "prompt": (
            "creative professional portrait of {trigger}, "
            "modern aesthetic, interesting lighting, "
            "subtle color grading, artistic composition, "
            "confident creative expression, contemporary style, "
            "design industry headshot, unique but professional, "
            "high quality, sharp focus"
        ),
        "negative_prompt": (
            "cartoon, anime, illustration, corporate boring, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "traditional, conservative, formal suit"
        ),
    },
}

# Default generation parameters
GENERATION_DEFAULTS = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "scheduler": "DPMSolverMultistepScheduler",
}

# LoRA training defaults
LORA_TRAINING_DEFAULTS = {
    "trigger_token": "@visageUser",
    "learning_rate": 1e-4,
    "train_steps": 1000,
    "rank": 32,
    "alpha": 32,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
}

# Quality thresholds for filtering
QUALITY_THRESHOLDS = {
    "min_face_similarity": 0.7,  # Minimum similarity to reference photos
    "min_quality_score": 0.6,   # Minimum overall quality score
    "max_artifact_score": 0.3,  # Maximum allowed artifact score
}


def get_prompt_for_style(style: str, trigger_token: str = "@visageUser") -> tuple[str, str]:
    """
    Get the prompt and negative prompt for a given style.
    
    Args:
        style: Style preset key (corporate, studio, natural, executive, creative)
        trigger_token: The LoRA trigger token to use
        
    Returns:
        Tuple of (prompt, negative_prompt)
    """
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLE_PRESETS.keys())}")
    
    preset = STYLE_PRESETS[style]
    prompt = preset["prompt"].format(trigger=trigger_token)
    negative_prompt = preset["negative_prompt"]
    
    return prompt, negative_prompt


def get_available_styles() -> list[dict]:
    """
    Get list of available style presets with metadata.
    
    Returns:
        List of style dictionaries with id, name, and description
    """
    return [
        {
            "id": key,
            "name": preset["name"],
            "description": preset["description"],
        }
        for key, preset in STYLE_PRESETS.items()
    ]
