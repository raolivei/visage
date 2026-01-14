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
          fontSize: 20,
          background: "#0a0a0b",
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fafaf9",
          fontFamily: "'Libre Baskerville', Georgia, serif",
          borderRadius: 8,
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
