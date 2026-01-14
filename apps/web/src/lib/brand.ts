/**
 * Visage Brand System
 * 
 * Typography-first, calm, professional, timeless.
 * No AI clichés, no gradients, no faces.
 */

export const brand = {
  name: "Visage",
  tagline: "Professional AI Headshots",
  description: "Self-hosted AI headshot generator with per-user LoRA training",
  
  // Color tokens
  colors: {
    // Primary palette - warm neutrals
    background: {
      primary: "#0a0a0b",      // Near black
      secondary: "#111113",    // Slightly lighter
      tertiary: "#18181b",     // Card backgrounds
      elevated: "#1f1f23",     // Elevated surfaces
    },
    foreground: {
      primary: "#fafaf9",      // Warm white
      secondary: "#a8a8a3",    // Muted text
      tertiary: "#71716b",     // Subtle text
      inverse: "#0a0a0b",      // On light backgrounds
    },
    // Accent - warm amber (restrained)
    accent: {
      DEFAULT: "#d4a574",      // Primary accent
      muted: "#b8956a",        // Subdued accent
      subtle: "rgba(212, 165, 116, 0.15)", // Background tint
    },
    // Semantic colors
    success: "#4ade80",
    warning: "#fbbf24",
    error: "#f87171",
    info: "#60a5fa",
    // Border colors
    border: {
      DEFAULT: "#27272a",
      subtle: "#1f1f23",
      accent: "rgba(212, 165, 116, 0.3)",
    },
  },
  
  // Typography
  typography: {
    // Font families
    fonts: {
      // Display: Elegant, editorial feel
      display: "'Libre Baskerville', 'Georgia', serif",
      // Body: Clean, humanist sans
      body: "'DM Sans', 'Inter', system-ui, sans-serif",
      // Mono: Technical contexts
      mono: "'JetBrains Mono', 'Fira Code', monospace",
    },
    // Font sizes (rem)
    sizes: {
      xs: "0.75rem",     // 12px
      sm: "0.875rem",    // 14px
      base: "1rem",      // 16px
      lg: "1.125rem",    // 18px
      xl: "1.25rem",     // 20px
      "2xl": "1.5rem",   // 24px
      "3xl": "1.875rem", // 30px
      "4xl": "2.25rem",  // 36px
      "5xl": "3rem",     // 48px
      "6xl": "3.75rem",  // 60px
    },
    // Line heights
    leading: {
      tight: "1.1",
      snug: "1.25",
      normal: "1.5",
      relaxed: "1.625",
      loose: "2",
    },
    // Letter spacing
    tracking: {
      tighter: "-0.05em",
      tight: "-0.025em",
      normal: "0",
      wide: "0.025em",
      wider: "0.05em",
      widest: "0.1em",
    },
  },
  
  // Spacing scale
  spacing: {
    px: "1px",
    0: "0",
    1: "0.25rem",   // 4px
    2: "0.5rem",    // 8px
    3: "0.75rem",   // 12px
    4: "1rem",      // 16px
    5: "1.25rem",   // 20px
    6: "1.5rem",    // 24px
    8: "2rem",      // 32px
    10: "2.5rem",   // 40px
    12: "3rem",     // 48px
    16: "4rem",     // 64px
    20: "5rem",     // 80px
    24: "6rem",     // 96px
  },
  
  // Border radius
  radius: {
    none: "0",
    sm: "0.25rem",   // 4px
    DEFAULT: "0.5rem", // 8px
    md: "0.75rem",   // 12px
    lg: "1rem",      // 16px
    xl: "1.5rem",    // 24px
    "2xl": "2rem",   // 32px
    full: "9999px",
  },
  
  // Shadows
  shadows: {
    sm: "0 1px 2px 0 rgba(0, 0, 0, 0.3)",
    DEFAULT: "0 4px 6px -1px rgba(0, 0, 0, 0.4)",
    md: "0 8px 15px -3px rgba(0, 0, 0, 0.4)",
    lg: "0 20px 25px -5px rgba(0, 0, 0, 0.5)",
    xl: "0 25px 50px -12px rgba(0, 0, 0, 0.6)",
    glow: "0 0 40px rgba(212, 165, 116, 0.15)",
  },
  
  // Animation
  animation: {
    duration: {
      fast: "150ms",
      normal: "300ms",
      slow: "500ms",
    },
    easing: {
      default: "cubic-bezier(0.4, 0, 0.2, 1)",
      in: "cubic-bezier(0.4, 0, 1, 1)",
      out: "cubic-bezier(0, 0, 0.2, 1)",
      inOut: "cubic-bezier(0.4, 0, 0.2, 1)",
    },
  },
  
  // Breakpoints
  breakpoints: {
    sm: "640px",
    md: "768px",
    lg: "1024px",
    xl: "1280px",
    "2xl": "1536px",
  },
} as const;

// Logo variants
export const logo = {
  // Primary wordmark (use this most places)
  wordmark: {
    text: "Visage",
    font: "Libre Baskerville",
    weight: 400,
    tracking: "0.02em",
  },
  // Uppercase variant (headers, admin)
  uppercase: {
    text: "VISAGE",
    font: "DM Sans",
    weight: 500,
    tracking: "0.15em",
  },
  // Monogram (favicons, small UI)
  monogram: "V",
} as const;

export type Brand = typeof brand;
export type Logo = typeof logo;
