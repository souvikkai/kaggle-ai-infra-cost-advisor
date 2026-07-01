import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    // Disable ESLint during production builds — lint locally instead.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Type errors are checked locally via tsc --noEmit before committing.
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
