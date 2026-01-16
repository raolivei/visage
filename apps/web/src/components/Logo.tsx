"use client";

import { cn } from "@/lib/utils";
import { VisageIcon } from "./VisageIcon";

interface LogoProps {
  variant?: "wordmark" | "uppercase" | "monogram" | "icon" | "lockup";
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
  iconClassName?: string;
}

const sizes = {
  sm: {
    wordmark: "text-lg",
    uppercase: "text-sm",
    monogram: "text-xl",
    icon: 24,
    lockupIcon: 28,
    lockupText: "text-lg",
  },
  md: {
    wordmark: "text-2xl",
    uppercase: "text-base",
    monogram: "text-3xl",
    icon: 32,
    lockupIcon: 36,
    lockupText: "text-xl",
  },
  lg: {
    wordmark: "text-4xl",
    uppercase: "text-xl",
    monogram: "text-5xl",
    icon: 48,
    lockupIcon: 48,
    lockupText: "text-2xl",
  },
  xl: {
    wordmark: "text-6xl",
    uppercase: "text-2xl",
    monogram: "text-7xl",
    icon: 64,
    lockupIcon: 64,
    lockupText: "text-3xl",
  },
};

/**
 * Visage Logo Component
 *
 * Variants:
 * - wordmark: "Visage" in Libre Baskerville serif
 * - uppercase: "VISAGE" in DM Sans
 * - monogram: Letter "V"
 * - icon: Concentric circles mark
 * - lockup: Icon + wordmark together
 */
export function Logo({
  variant = "wordmark",
  size = "md",
  className,
  iconClassName,
}: LogoProps) {
  const sizeConfig = sizes[size];

  // Icon variant
  if (variant === "icon") {
    return <VisageIcon size={sizeConfig.icon} className={cn(className)} />;
  }

  // Lockup: Icon + Wordmark
  if (variant === "lockup") {
    return (
      <div className={cn("flex items-center gap-3", className)}>
        <VisageIcon size={sizeConfig.lockupIcon} className={iconClassName} />
        <span
          className={cn(
            "font-display font-normal tracking-logo text-foreground",
            sizeConfig.lockupText
          )}
        >
          Visage
        </span>
      </div>
    );
  }

  // Monogram: Letter V
  if (variant === "monogram") {
    return (
      <span
        className={cn(
          "font-display font-normal tracking-tight text-foreground select-none",
          sizeConfig.monogram,
          className
        )}
        aria-label="Visage"
      >
        V
      </span>
    );
  }

  // Uppercase: VISAGE
  if (variant === "uppercase") {
    return (
      <span
        className={cn(
          "font-sans font-medium tracking-logo-caps text-foreground select-none",
          sizeConfig.uppercase,
          className
        )}
        aria-label="Visage"
      >
        VISAGE
      </span>
    );
  }

  // Default: Wordmark (Visage in serif)
  return (
    <span
      className={cn(
        "font-display font-normal tracking-logo text-foreground select-none",
        sizeConfig.wordmark,
        className
      )}
      aria-label="Visage"
    >
      Visage
    </span>
  );
}

export default Logo;
