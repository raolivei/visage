"""
Visage Prompt Templates

Curated prompts for professional headshot generation.
Includes 15+ styles with multiple variations for diversity.
"""

import random
from typing import Optional


# =============================================================================
# Style Presets - 15+ Professional Styles
# =============================================================================

STYLE_PRESETS = {
    # -------------------------------------------------------------------------
    # Business / Corporate Styles (Commercial Quality)
    # -------------------------------------------------------------------------
    "corporate": {
        "name": "Corporate / LinkedIn",
        "description": "Professional business headshot suitable for LinkedIn and corporate profiles",
        "category": "business",
        "variations": [
            {
                "prompt": (
                    "professional headshot portrait photo of {trigger}, "
                    "wearing navy blue sweater over white collared shirt, "
                    "outdoors on tree-lined street, soft natural daylight, "
                    "beautiful green bokeh background, "
                    "warm natural skin tones, subtle smile, looking at camera, "
                    "shallow depth of field, 85mm lens, f/1.8 aperture, "
                    "professional photography, photorealistic, 8k uhd"
                ),
            },
            {
                "prompt": (
                    "professional business headshot of {trigger}, "
                    "wearing teal plaid button-up shirt, "
                    "modern bright office with warm pendant lights, "
                    "soft diffused lighting, friendly genuine smile, "
                    "blurred office background with bokeh lights, "
                    "natural skin texture, eye-level angle, "
                    "Canon EOS R5, 85mm portrait lens, f/2.0, "
                    "commercial headshot photography, photorealistic"
                ),
            },
            {
                "prompt": (
                    "corporate profile photo of {trigger}, "
                    "smart casual attire, polo shirt, "
                    "outdoor urban setting with soft natural light, "
                    "pleasant neutral expression, approachable and confident, "
                    "creamy background bokeh, professional color grading, "
                    "high-end portrait photography, Sony A7R IV, "
                    "85mm f/1.4 GM lens, photorealistic, 8k"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, painting, drawing, cgi, 3d render, "
            "blurry, low quality, distorted face, extra limbs, bad anatomy, "
            "text, watermark, logo, signature, frame, border, "
            "harsh shadows, overexposed, underexposed, "
            "plastic skin, airbrushed, uncanny valley, "
            "bad teeth, crossed eyes, asymmetric face"
        ),
    },
    
    "executive": {
        "name": "Executive C-Suite",
        "description": "Premium executive portrait for C-suite and leadership",
        "category": "business",
        "variations": [
            {
                "prompt": (
                    "executive portrait of {trigger}, "
                    "premium business attire, sophisticated dark background, "
                    "professional studio lighting, authoritative presence, "
                    "confident and distinguished expression, "
                    "Fortune 500 executive style, high-end corporate photography"
                ),
            },
            {
                "prompt": (
                    "CEO headshot of {trigger}, "
                    "elegant suit, modern office background with depth, "
                    "Rembrandt lighting, powerful yet approachable demeanor, "
                    "leadership quality portrait, magazine cover worthy"
                ),
            },
            {
                "prompt": (
                    "senior executive portrait of {trigger}, "
                    "tailored professional attire, subtle gradient background, "
                    "dramatic side lighting, commanding presence, "
                    "board room ready, prestigious corporate photography"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, casual wear, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "young, inexperienced, informal, outdoor, messy"
        ),
    },
    
    "legal_finance": {
        "name": "Legal / Finance",
        "description": "Conservative, authoritative headshot for legal and finance professionals",
        "category": "business",
        "variations": [
            {
                "prompt": (
                    "professional attorney headshot of {trigger}, "
                    "conservative business attire, law firm setting, "
                    "even professional lighting, serious and trustworthy expression, "
                    "traditional portrait style, authoritative presence"
                ),
            },
            {
                "prompt": (
                    "financial advisor portrait of {trigger}, "
                    "formal professional attire, neutral background, "
                    "clean lighting, confident and reliable expression, "
                    "trustworthy appearance, sharp and professional"
                ),
            },
            {
                "prompt": (
                    "partner headshot of {trigger}, "
                    "elegant professional dress, sophisticated background, "
                    "flattering studio lighting, distinguished and capable expression, "
                    "traditional high-end portrait photography"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, casual, trendy, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "flashy, colorful, informal, smiling broadly"
        ),
    },
    
    # -------------------------------------------------------------------------
    # Studio / Classic Styles
    # -------------------------------------------------------------------------
    "studio_classic": {
        "name": "Studio Classic",
        "description": "Classic studio portrait with dramatic lighting",
        "category": "studio",
        "variations": [
            {
                "prompt": (
                    "professional studio portrait of {trigger}, "
                    "dramatic Rembrandt lighting, dark charcoal background, "
                    "high contrast, sharp focus, professional photography, "
                    "elegant composition, confident expression, magazine quality"
                ),
            },
            {
                "prompt": (
                    "classic studio headshot of {trigger}, "
                    "butterfly lighting setup, neutral backdrop, "
                    "professional studio quality, timeless portrait style, "
                    "flattering angles, refined and polished"
                ),
            },
            {
                "prompt": (
                    "fine art portrait of {trigger}, "
                    "split lighting for drama, gradient background, "
                    "cinematic quality, artistic composition, "
                    "high-end studio photography, striking and memorable"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, flat lighting, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "outdoor, natural light, busy background, amateur"
        ),
    },
    
    "minimalist": {
        "name": "Minimalist Clean",
        "description": "Simple, modern minimalist headshot",
        "category": "studio",
        "variations": [
            {
                "prompt": (
                    "minimalist portrait of {trigger}, "
                    "clean white background, soft even lighting, "
                    "simple composition, modern aesthetic, "
                    "sharp focus on face, high resolution, "
                    "contemporary professional headshot"
                ),
            },
            {
                "prompt": (
                    "clean modern headshot of {trigger}, "
                    "pure white backdrop, diffused studio lighting, "
                    "minimal distractions, focused on subject, "
                    "sleek and contemporary, professional quality"
                ),
            },
            {
                "prompt": (
                    "simple elegant portrait of {trigger}, "
                    "seamless light background, balanced exposure, "
                    "uncluttered composition, modern professional style, "
                    "crisp and refined"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, busy background, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "dramatic lighting, dark, moody, cluttered"
        ),
    },
    
    # -------------------------------------------------------------------------
    # Natural / Outdoor Styles
    # -------------------------------------------------------------------------
    "natural_light": {
        "name": "Natural Light",
        "description": "Warm, approachable headshot with natural lighting",
        "category": "natural",
        "variations": [
            {
                "prompt": (
                    "natural light portrait of {trigger}, "
                    "golden hour lighting, soft bokeh background, "
                    "warm tones, genuine smile, approachable expression, "
                    "outdoor professional headshot, sharp focus on eyes"
                ),
            },
            {
                "prompt": (
                    "outdoor headshot of {trigger}, "
                    "soft diffused daylight, natural greenery background, "
                    "warm and inviting, authentic expression, "
                    "lifestyle professional portrait"
                ),
            },
            {
                "prompt": (
                    "window light portrait of {trigger}, "
                    "soft natural side lighting, neutral indoor setting, "
                    "warm natural tones, relaxed professional expression, "
                    "authentic and engaging"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, harsh lighting, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "studio, artificial light, dark, moody, overprocessed"
        ),
    },
    
    "friendly_approachable": {
        "name": "Friendly Approachable",
        "description": "Warm, welcoming headshot that builds connection",
        "category": "natural",
        "variations": [
            {
                "prompt": (
                    "friendly professional headshot of {trigger}, "
                    "warm smile, approachable expression, "
                    "soft natural lighting, pleasant background, "
                    "inviting and trustworthy, genuine warmth"
                ),
            },
            {
                "prompt": (
                    "welcoming portrait of {trigger}, "
                    "bright eyes, authentic smile, "
                    "light and airy setting, positive energy, "
                    "personable and engaging, professional yet friendly"
                ),
            },
            {
                "prompt": (
                    "approachable business headshot of {trigger}, "
                    "genuine expression, soft lighting, "
                    "warm color tones, connection-building portrait, "
                    "professional and relatable"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, serious, stern, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "intimidating, cold, distant, formal"
        ),
    },
    
    # -------------------------------------------------------------------------
    # Creative / Modern Styles
    # -------------------------------------------------------------------------
    "creative_modern": {
        "name": "Creative Modern",
        "description": "Modern, creative headshot for designers and artists",
        "category": "creative",
        "variations": [
            {
                "prompt": (
                    "creative professional portrait of {trigger}, "
                    "modern aesthetic, interesting colored lighting, "
                    "subtle color grading, artistic composition, "
                    "confident creative expression, contemporary style"
                ),
            },
            {
                "prompt": (
                    "artistic headshot of {trigger}, "
                    "unique lighting setup, creative background, "
                    "modern design sensibility, expressive portrait, "
                    "design industry style, memorable and distinctive"
                ),
            },
            {
                "prompt": (
                    "contemporary portrait of {trigger}, "
                    "bold composition, subtle neon accent lighting, "
                    "editorial quality, creative professional aesthetic, "
                    "fresh and modern"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, corporate boring, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "traditional, conservative, formal suit, outdated"
        ),
    },
    
    "tech_startup": {
        "name": "Tech Startup",
        "description": "Casual, modern tech industry vibe",
        "category": "creative",
        "variations": [
            {
                "prompt": (
                    "tech startup headshot of {trigger}, "
                    "smart casual attire, modern office background, "
                    "confident and innovative expression, "
                    "Silicon Valley style, approachable tech professional"
                ),
            },
            {
                "prompt": (
                    "founder portrait of {trigger}, "
                    "casual professional look, contemporary workspace setting, "
                    "energetic and visionary expression, "
                    "startup culture aesthetic, modern and fresh"
                ),
            },
            {
                "prompt": (
                    "tech industry headshot of {trigger}, "
                    "relaxed professional attire, blurred office background, "
                    "innovative and approachable, sharp focus, "
                    "modern professional portrait"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, formal suit, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "traditional, stuffy, corporate, outdated"
        ),
    },
    
    # -------------------------------------------------------------------------
    # Professional Services Styles
    # -------------------------------------------------------------------------
    "healthcare": {
        "name": "Healthcare Professional",
        "description": "Clean, trustworthy headshot for medical professionals",
        "category": "professional",
        "variations": [
            {
                "prompt": (
                    "healthcare professional portrait of {trigger}, "
                    "clean white coat or professional attire, "
                    "clinical yet warm setting, trustworthy expression, "
                    "caring and competent, medical professional headshot"
                ),
            },
            {
                "prompt": (
                    "doctor headshot of {trigger}, "
                    "professional medical attire, soft clinical lighting, "
                    "compassionate and knowledgeable expression, "
                    "reassuring presence, high quality medical portrait"
                ),
            },
            {
                "prompt": (
                    "medical professional portrait of {trigger}, "
                    "clean professional look, neutral background, "
                    "confident and caring demeanor, healthcare aesthetic, "
                    "trustworthy and approachable"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, casual, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "scary, intimidating, cold, sterile"
        ),
    },
    
    "academic": {
        "name": "Academic / Professor",
        "description": "Intellectual, approachable headshot for academics",
        "category": "professional",
        "variations": [
            {
                "prompt": (
                    "academic portrait of {trigger}, "
                    "professorial attire, library or office background, "
                    "intellectual yet approachable expression, "
                    "scholarly presence, university profile style"
                ),
            },
            {
                "prompt": (
                    "professor headshot of {trigger}, "
                    "smart professional dress, academic setting, "
                    "thoughtful and engaging expression, "
                    "knowledgeable and welcoming, scholarly portrait"
                ),
            },
            {
                "prompt": (
                    "researcher portrait of {trigger}, "
                    "professional academic attire, neutral background, "
                    "curious and intelligent expression, "
                    "approachable expert, institutional quality"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, casual, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "stuffy, intimidating, old-fashioned, boring"
        ),
    },
    
    "real_estate": {
        "name": "Real Estate Agent",
        "description": "Friendly, professional headshot for real estate",
        "category": "professional",
        "variations": [
            {
                "prompt": (
                    "real estate agent headshot of {trigger}, "
                    "professional friendly attire, warm smile, "
                    "trustworthy and approachable expression, "
                    "polished professional look, real estate industry style"
                ),
            },
            {
                "prompt": (
                    "realtor portrait of {trigger}, "
                    "smart professional dress, confident smile, "
                    "welcoming and reliable expression, "
                    "client-facing professional headshot"
                ),
            },
            {
                "prompt": (
                    "real estate professional headshot of {trigger}, "
                    "business casual attire, bright and friendly, "
                    "trustworthy and personable, sharp focus, "
                    "professional marketing photo style"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, serious, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "unfriendly, cold, distant, unprofessional"
        ),
    },
    
    # -------------------------------------------------------------------------
    # Personal Branding Styles
    # -------------------------------------------------------------------------
    "speaker_author": {
        "name": "Speaker / Author",
        "description": "Confident, engaging headshot for thought leaders",
        "category": "personal_brand",
        "variations": [
            {
                "prompt": (
                    "keynote speaker portrait of {trigger}, "
                    "confident and charismatic expression, "
                    "professional stage-ready look, dynamic presence, "
                    "thought leader aesthetic, engaging and inspiring"
                ),
            },
            {
                "prompt": (
                    "author headshot of {trigger}, "
                    "intelligent and approachable expression, "
                    "professional yet personable, warm lighting, "
                    "book jacket quality portrait"
                ),
            },
            {
                "prompt": (
                    "thought leader portrait of {trigger}, "
                    "commanding yet approachable presence, "
                    "professional polish, confident expression, "
                    "influential and engaging, high-end portrait"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, shy, timid, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "boring, forgettable, low energy"
        ),
    },
    
    "bold_confident": {
        "name": "Bold Confident",
        "description": "Strong, impactful headshot that commands attention",
        "category": "personal_brand",
        "variations": [
            {
                "prompt": (
                    "bold confident portrait of {trigger}, "
                    "powerful presence, dramatic lighting, "
                    "strong eye contact, commanding expression, "
                    "high impact professional headshot"
                ),
            },
            {
                "prompt": (
                    "striking headshot of {trigger}, "
                    "confident and assertive expression, "
                    "dynamic composition, professional edge, "
                    "memorable and impactful"
                ),
            },
            {
                "prompt": (
                    "powerful portrait of {trigger}, "
                    "bold lighting, strong presence, "
                    "confident and capable expression, "
                    "attention-grabbing professional photo"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, weak, timid, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "boring, soft, forgettable, passive"
        ),
    },
    
    # -------------------------------------------------------------------------
    # Entertainment / Creative Styles  
    # -------------------------------------------------------------------------
    "actor_model": {
        "name": "Actor / Model",
        "description": "Dramatic 3/4 angle portrait for performers",
        "category": "entertainment",
        "variations": [
            {
                "prompt": (
                    "actor headshot of {trigger}, "
                    "three-quarter angle, dramatic lighting, "
                    "expressive eyes, charismatic presence, "
                    "casting director quality, professional acting headshot"
                ),
            },
            {
                "prompt": (
                    "model portrait of {trigger}, "
                    "editorial style, artistic lighting, "
                    "striking composition, captivating expression, "
                    "high fashion aesthetic, memorable presence"
                ),
            },
            {
                "prompt": (
                    "performer headshot of {trigger}, "
                    "dynamic angle, professional lighting setup, "
                    "engaging expression, industry standard quality, "
                    "versatile acting portrait"
                ),
            },
        ],
        "negative_prompt": (
            "cartoon, anime, illustration, boring, flat, "
            "blurry, low quality, distorted face, extra limbs, "
            "bad anatomy, text, watermark, logo, "
            "corporate, stiff, lifeless"
        ),
    },
}


# =============================================================================
# Default Parameters
# =============================================================================

GENERATION_DEFAULTS = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "scheduler": "DPMSolverMultistepScheduler",
    "num_images_per_style": 5,  # Generate multiple per style
}

LORA_TRAINING_DEFAULTS = {
    "trigger_token": "@visageUser",
    "learning_rate": 1e-4,
    "train_steps": 1500,
    "rank": 32,
    "alpha": 32,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "resolution": 1024,
}

QUALITY_THRESHOLDS = {
    "min_face_similarity": 0.65,
    "min_aesthetic_score": 5.5,
    "min_technical_score": 0.5,
    "max_artifact_score": 0.35,
    "target_pass_rate": 0.35,  # 35% of images should pass
}


# =============================================================================
# Helper Functions
# =============================================================================

# Style aliases for simpler names
STYLE_ALIASES = {
    "creative": "creative_modern",
    "studio": "studio_classic", 
    "natural": "natural_light",
    "friendly": "friendly_approachable",
    "tech": "tech_startup",
    "medical": "healthcare",
    "professor": "academic",
    "lawyer": "legal_finance",
    "speaker": "speaker_author",
    "actor": "actor_model",
    "bold": "bold_confident",
    "clean": "minimalist",
}


def get_prompt_for_style(
    style: str, 
    trigger_token: str = "@visageUser",
    variation: Optional[int] = None,
) -> tuple[str, str]:
    """
    Get the prompt and negative prompt for a given style.
    
    Args:
        style: Style preset key (supports aliases like 'creative' -> 'creative_modern')
        trigger_token: The LoRA trigger token to use
        variation: Specific variation index (None = random)
        
    Returns:
        Tuple of (prompt, negative_prompt)
    """
    # Resolve aliases
    resolved_style = STYLE_ALIASES.get(style, style)
    
    if resolved_style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLE_PRESETS.keys())} or aliases: {list(STYLE_ALIASES.keys())}")
    
    preset = STYLE_PRESETS[resolved_style]
    variations = preset["variations"]
    
    # Select variation
    if variation is not None:
        idx = variation % len(variations)
    else:
        idx = random.randint(0, len(variations) - 1)
        
    selected = variations[idx]
    prompt = selected["prompt"].format(trigger=trigger_token)
    negative_prompt = preset["negative_prompt"]
    
    return prompt, negative_prompt


def get_all_prompts_for_style(
    style: str,
    trigger_token: str = "@visageUser",
) -> list[tuple[str, str]]:
    """
    Get all prompt variations for a given style.
    
    Args:
        style: Style preset key
        trigger_token: The LoRA trigger token to use
        
    Returns:
        List of (prompt, negative_prompt) tuples
    """
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style: {style}")
        
    preset = STYLE_PRESETS[style]
    results = []
    
    for var in preset["variations"]:
        prompt = var["prompt"].format(trigger=trigger_token)
        results.append((prompt, preset["negative_prompt"]))
        
    return results


def get_available_styles() -> list[dict]:
    """
    Get list of available style presets with metadata.
    
    Returns:
        List of style dictionaries with id, name, description, and category
    """
    return [
        {
            "id": key,
            "name": preset["name"],
            "description": preset["description"],
            "category": preset["category"],
            "num_variations": len(preset["variations"]),
        }
        for key, preset in STYLE_PRESETS.items()
    ]


def get_styles_by_category(category: str) -> list[dict]:
    """
    Get styles filtered by category.
    
    Args:
        category: One of 'business', 'studio', 'natural', 'creative', 
                  'professional', 'personal_brand', 'entertainment'
                  
    Returns:
        List of matching style dictionaries
    """
    return [
        {
            "id": key,
            "name": preset["name"],
            "description": preset["description"],
            "category": preset["category"],
        }
        for key, preset in STYLE_PRESETS.items()
        if preset["category"] == category
    ]


def get_categories() -> list[str]:
    """Get list of all available categories."""
    categories = set()
    for preset in STYLE_PRESETS.values():
        categories.add(preset["category"])
    return sorted(list(categories))


def get_recommended_styles(
    industry: Optional[str] = None,
    num_styles: int = 5,
) -> list[str]:
    """
    Get recommended styles based on industry or use case.
    
    Args:
        industry: Optional industry hint (tech, finance, healthcare, creative, etc.)
        num_styles: Number of styles to recommend
        
    Returns:
        List of style IDs
    """
    # Industry to style mapping
    industry_map = {
        "tech": ["tech_startup", "creative_modern", "minimalist", "friendly_approachable", "corporate"],
        "finance": ["legal_finance", "executive", "corporate", "studio_classic", "bold_confident"],
        "healthcare": ["healthcare", "friendly_approachable", "natural_light", "corporate", "academic"],
        "creative": ["creative_modern", "actor_model", "natural_light", "minimalist", "bold_confident"],
        "legal": ["legal_finance", "executive", "corporate", "studio_classic", "speaker_author"],
        "real_estate": ["real_estate", "friendly_approachable", "natural_light", "corporate", "bold_confident"],
        "academic": ["academic", "speaker_author", "natural_light", "studio_classic", "friendly_approachable"],
        "default": ["corporate", "natural_light", "studio_classic", "friendly_approachable", "minimalist"],
    }
    
    styles = industry_map.get(industry, industry_map["default"])
    return styles[:num_styles]
