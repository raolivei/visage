"use client";

import { cn } from "@/lib/utils";

interface LogoProps {
  variant?: "wordmark" | "uppercase" | "monogram";
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
}

const sizes = {
  sm: { wordmark: "text-lg", uppercase: "text-sm", monogram: "text-xl" },
  md: { wordmark: "text-2xl", uppercase: "text-base", monogram: "text-3xl" },
  lg: { wordmark: "text-4xl", uppercase: "text-xl", monogram: "text-5xl" },
  xl: { wordmark: "text-6xl", uppercase: "text-2xl", monogram: "text-7xl" },
};

/**
 * Visage Logo Component
 * 
 * Typography-driven wordmark logo.
 * No icons, no symbols — refined and confident.
 */
export function Logo({ 
  variant = "wordmark", 
  size = "md",
  className 
}: LogoProps) {
  if (variant === "monogram") {
    return (
      <span
        className={cn(
          "font-display font-normal tracking-tight text-foreground select-none",
          sizes[size].monogram,
          className
        )}
        aria-label="Visage"
      >
        V
      </span>
    );
  }

  if (variant === "uppercase") {
    return (
      <span
        className={cn(
          "font-sans font-medium tracking-[0.15em] text-foreground select-none",
          sizes[size].uppercase,
          className
        )}
        aria-label="Visage"
      >
        VISAGE
      </span>
    );
  }

  // Default: wordmark (primary logo)
  return (
    <span
      className={cn(
        "font-display font-normal tracking-[0.02em] text-foreground select-none",
        sizes[size].wordmark,
        className
      )}
      aria-label="Visage"
    >
      Visage
    </span>
  );
}

/**
 * SVG Logo for exports and static usage
 */
export function LogoSVG({ 
  variant = "wordmark",
  width = 120,
  height = 32,
  color = "currentColor",
  className,
}: {
  variant?: "wordmark" | "uppercase" | "monogram";
  width?: number;
  height?: number;
  color?: string;
  className?: string;
}) {
  if (variant === "monogram") {
    return (
      <svg
        width={height}
        height={height}
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
        aria-label="Visage"
      >
        <text
          x="50%"
          y="50%"
          dominantBaseline="central"
          textAnchor="middle"
          fill={color}
          fontFamily="'Libre Baskerville', Georgia, serif"
          fontSize="32"
          fontWeight="400"
        >
          V
        </text>
      </svg>
    );
  }

  if (variant === "uppercase") {
    return (
      <svg
        width={width}
        height={height}
        viewBox="0 0 120 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
        aria-label="Visage"
      >
        <text
          x="0"
          y="50%"
          dominantBaseline="central"
          fill={color}
          fontFamily="'DM Sans', sans-serif"
          fontSize="16"
          fontWeight="500"
          letterSpacing="0.15em"
        >
          VISAGE
        </text>
      </svg>
    );
  }

  // Default: wordmark
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 120 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Visage"
    >
      <text
        x="0"
        y="50%"
        dominantBaseline="central"
        fill={color}
        fontFamily="'Libre Baskerville', Georgia, serif"
        fontSize="24"
        fontWeight="400"
        letterSpacing="0.02em"
      >
        Visage
      </text>
    </svg>
  );
}

export default Logo;
