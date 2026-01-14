"use client";

import { cn } from "@/lib/utils";

interface VisageIconProps {
  size?: number;
  className?: string;
  strokeWidth?: number;
}

/**
 * Visage Brand Icon
 * 
 * Concentric flowing lines forming a circular mark.
 * Represents focus, imaging, and refinement.
 */
export function VisageIcon({ 
  size = 64, 
  className,
  strokeWidth = 2.5,
}: VisageIconProps) {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 64 64" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      className={cn("text-accent", className)}
      aria-label="Visage"
    >
      <g 
        stroke="currentColor" 
        strokeWidth={strokeWidth} 
        fill="none" 
        strokeLinecap="round"
      >
        {/* Outer circle */}
        <circle cx="32" cy="32" r="26" />
        
        {/* Right side flowing curves */}
        <path d="M32 10 Q52 12 54 32 Q52 52 32 54" />
        <path d="M32 16 Q46 17 48 32 Q46 47 32 48" />
        <path d="M32 22 Q40 23 42 32 Q40 41 32 42" />
        <path d="M32 28 Q35 28.5 36 32 Q35 35.5 32 36" />
        
        {/* Left side flowing curves (mirrored) */}
        <path d="M32 10 Q12 12 10 32 Q12 52 32 54" />
        <path d="M32 16 Q18 17 16 32 Q18 47 32 48" />
        <path d="M32 22 Q24 23 22 32 Q24 41 32 42" />
        <path d="M32 28 Q29 28.5 28 32 Q29 35.5 32 36" />
      </g>
    </svg>
  );
}

/**
 * Visage Icon with background
 * For use in places where a contained icon is needed
 */
export function VisageIconContained({
  size = 48,
  className,
}: {
  size?: number;
  className?: string;
}) {
  return (
    <div 
      className={cn(
        "flex items-center justify-center rounded-xl bg-background-tertiary",
        className
      )}
      style={{ width: size, height: size }}
    >
      <VisageIcon size={size * 0.65} strokeWidth={2} />
    </div>
  );
}

export default VisageIcon;
