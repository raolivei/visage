import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Background colors - warm dark neutrals
        background: {
          DEFAULT: "#0a0a0b",
          secondary: "#111113",
          tertiary: "#18181b",
          elevated: "#1f1f23",
        },
        // Foreground colors - warm whites
        foreground: {
          DEFAULT: "#fafaf9",
          secondary: "#a8a8a3",
          tertiary: "#71716b",
          inverse: "#0a0a0b",
        },
        // Accent - warm amber (restrained, editorial)
        accent: {
          DEFAULT: "#d4a574",
          muted: "#b8956a",
          subtle: "rgba(212, 165, 116, 0.15)",
          50: "#fefcf8",
          100: "#fdf6eb",
          200: "#fae8cc",
          300: "#f5d5a3",
          400: "#e8b87a",
          500: "#d4a574",
          600: "#b8956a",
          700: "#8f7352",
          800: "#6b5640",
          900: "#4a3c2d",
        },
        // Border colors
        border: {
          DEFAULT: "#27272a",
          subtle: "#1f1f23",
          accent: "rgba(212, 165, 116, 0.3)",
        },
        // Semantic colors
        success: {
          DEFAULT: "#4ade80",
          muted: "rgba(74, 222, 128, 0.15)",
        },
        warning: {
          DEFAULT: "#fbbf24",
          muted: "rgba(251, 191, 36, 0.15)",
        },
        error: {
          DEFAULT: "#f87171",
          muted: "rgba(248, 113, 113, 0.15)",
        },
        info: {
          DEFAULT: "#60a5fa",
          muted: "rgba(96, 165, 250, 0.15)",
        },
        // Legacy visage palette (for backward compatibility)
        visage: {
          50: "#fafaf9",
          100: "#f1f0ee",
          200: "#e2e1de",
          300: "#c8c6c1",
          400: "#a8a8a3",
          500: "#71716b",
          600: "#52524d",
          700: "#3a3a36",
          800: "#27272a",
          900: "#18181b",
          950: "#0a0a0b",
        },
      },
      fontFamily: {
        // Display: Elegant serif for headings and logo
        display: ["'Libre Baskerville'", "Georgia", "serif"],
        // Sans: Clean humanist for body text
        sans: ["'DM Sans'", "Inter", "system-ui", "sans-serif"],
        // Mono: Technical contexts
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      letterSpacing: {
        logo: "0.02em",
        "logo-caps": "0.15em",
      },
      animation: {
        "fade-in": "fadeIn 0.6s ease-out forwards",
        "fade-in-up": "fadeInUp 0.8s ease-out forwards",
        "slide-up": "slideUp 0.5s ease-out forwards",
        "slide-down": "slideDown 0.3s ease-out forwards",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "scale-in": "scaleIn 0.2s ease-out forwards",
        "bounce": "bounce 2s infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        fadeInUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(30px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideDown: {
          "0%": { opacity: "0", transform: "translateY(-10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        scaleIn: {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        bounce: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
      },
      boxShadow: {
        glow: "0 0 40px rgba(212, 165, 116, 0.15)",
        "glow-lg": "0 0 60px rgba(212, 165, 116, 0.2)",
      },
      borderRadius: {
        "4xl": "2rem",
      },
    },
  },
  plugins: [],
};

export default config;
