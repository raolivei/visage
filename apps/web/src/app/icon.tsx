import { ImageResponse } from "next/og";

export const runtime = "edge";

export const size = {
  width: 32,
  height: 32,
};

export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0a0a0b",
          borderRadius: 6,
        }}
      >
        {/* Simplified concentric circles icon */}
        <svg
          width="24"
          height="24"
          viewBox="0 0 64 64"
          fill="none"
        >
          <g
            stroke="#d4a574"
            strokeWidth="4"
            fill="none"
            strokeLinecap="round"
          >
            <circle cx="32" cy="32" r="24" />
            <path d="M32 12 Q50 14 52 32 Q50 50 32 52" />
            <path d="M32 12 Q14 14 12 32 Q14 50 32 52" />
            <path d="M32 20 Q42 21 44 32 Q42 43 32 44" />
            <path d="M32 20 Q22 21 20 32 Q22 43 32 44" />
          </g>
        </svg>
      </div>
    ),
    {
      ...size,
    }
  );
}
