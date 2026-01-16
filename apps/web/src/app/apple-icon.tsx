import { ImageResponse } from "next/og";

export const runtime = "edge";

export const size = {
  width: 180,
  height: 180,
};

export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#0a0a0b",
        borderRadius: 36,
      }}
    >
      {/* Concentric circles icon */}
      <svg width="120" height="120" viewBox="0 0 64 64" fill="none">
        <g stroke="#d4a574" strokeWidth="2.5" fill="none" strokeLinecap="round">
          <circle cx="32" cy="32" r="26" />
          <path d="M32 10 Q52 12 54 32 Q52 52 32 54" />
          <path d="M32 16 Q46 17 48 32 Q46 47 32 48" />
          <path d="M32 22 Q40 23 42 32 Q40 41 32 42" />
          <path d="M32 28 Q35 28.5 36 32 Q35 35.5 32 36" />
          <path d="M32 10 Q12 12 10 32 Q12 52 32 54" />
          <path d="M32 16 Q18 17 16 32 Q18 47 32 48" />
          <path d="M32 22 Q24 23 22 32 Q24 41 32 42" />
          <path d="M32 28 Q29 28.5 28 32 Q29 35.5 32 36" />
        </g>
      </svg>
    </div>,
    {
      ...size,
    }
  );
}
