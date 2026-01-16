/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "9000",
        pathname: "/visage/**",
      },
      {
        protocol: "https",
        hostname: "minio.eldertree.local",
        pathname: "/visage/**",
      },
    ],
  },
  env: {
    API_URL: process.env.API_URL || "http://localhost:8004",
  },
};

module.exports = nextConfig;
