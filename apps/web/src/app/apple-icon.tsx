import { ImageResponse } from "next/og";

export const runtime = "edge";

export const size = {
  width: 180,
  height: 180,
};

export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          fontSize: 120,
          background: "#0a0a0b",
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fafaf9",
          fontFamily: "'Libre Baskerville', Georgia, serif",
          borderRadius: 40,
        }}
      >
        V
      </div>
    ),
    {
      ...size,
    }
  );
}
