import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Visage - AI Headshot Generator",
  description: "Professional AI-powered headshots for LinkedIn and business profiles",
  keywords: ["headshot", "AI", "portrait", "professional", "LinkedIn"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="flex flex-col min-h-screen">
          {/* Header */}
          <header className="sticky top-0 z-50 border-b border-visage-800/50 bg-visage-950/80 backdrop-blur-xl">
            <div className="container mx-auto px-6 py-4">
              <nav className="flex items-center justify-between">
                <a href="/" className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-400 to-accent-600 flex items-center justify-center">
                    <span className="text-visage-950 font-bold text-lg">V</span>
                  </div>
                  <span className="text-xl font-semibold text-visage-100">
                    Visage
                  </span>
                </a>
                
                <div className="flex items-center gap-4">
                  <a href="/packs" className="btn-ghost">
                    My Packs
                  </a>
                  <a href="/packs/new" className="btn-primary">
                    New Headshot
                  </a>
                </div>
              </nav>
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-visage-800/50 py-6">
            <div className="container mx-auto px-6 text-center text-visage-500 text-sm">
              <p>Visage — Self-hosted AI Headshot Generator</p>
              <p className="mt-1">Running on ElderTree</p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
