import { Camera, Sparkles, Download, ArrowRight } from "lucide-react";

export default function HomePage() {
  return (
    <div className="container mx-auto px-6 py-16">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto mb-20">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent-500/10 border border-accent-500/20 rounded-full text-accent-400 text-sm mb-6 animate-fade-in">
          <Sparkles className="w-4 h-4" />
          <span>AI-Powered Professional Headshots</span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-bold text-visage-50 mb-6 animate-slide-up">
          Transform Your Photos Into{" "}
          <span className="text-gradient">Studio-Quality</span>{" "}
          Headshots
        </h1>
        
        <p className="text-xl text-visage-400 mb-10 max-w-2xl mx-auto animate-slide-up delay-100">
          Upload your photos, choose your style, and get professional headshots 
          perfect for LinkedIn, company profiles, and more. All self-hosted on 
          your own infrastructure.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center animate-slide-up delay-200">
          <a href="/packs/new" className="btn-primary text-lg px-8 py-4">
            Get Started
            <ArrowRight className="w-5 h-5 ml-2" />
          </a>
          <a href="/packs" className="btn-secondary text-lg px-8 py-4">
            View My Packs
          </a>
        </div>
      </section>

      {/* Features */}
      <section className="grid md:grid-cols-3 gap-8 mb-20">
        <div className="glass-card p-8 animate-fade-in delay-200">
          <div className="w-14 h-14 rounded-xl bg-accent-500/10 flex items-center justify-center mb-6">
            <Camera className="w-7 h-7 text-accent-400" />
          </div>
          <h3 className="text-xl font-semibold text-visage-100 mb-3">
            Upload Your Photos
          </h3>
          <p className="text-visage-400">
            Upload 8-20 photos of yourself. Mix of angles, expressions, and 
            lighting for best results.
          </p>
        </div>

        <div className="glass-card p-8 animate-fade-in delay-300">
          <div className="w-14 h-14 rounded-xl bg-accent-500/10 flex items-center justify-center mb-6">
            <Sparkles className="w-7 h-7 text-accent-400" />
          </div>
          <h3 className="text-xl font-semibold text-visage-100 mb-3">
            AI Training & Generation
          </h3>
          <p className="text-visage-400">
            Our AI learns your unique features and generates dozens of 
            professional variations in your chosen styles.
          </p>
        </div>

        <div className="glass-card p-8 animate-fade-in delay-400">
          <div className="w-14 h-14 rounded-xl bg-accent-500/10 flex items-center justify-center mb-6">
            <Download className="w-7 h-7 text-accent-400" />
          </div>
          <h3 className="text-xl font-semibold text-visage-100 mb-3">
            Download Your Best
          </h3>
          <p className="text-visage-400">
            We auto-filter to keep only the best results. Pick your favorites 
            and download high-resolution images.
          </p>
        </div>
      </section>

      {/* Style Preview */}
      <section className="text-center mb-20">
        <h2 className="text-3xl font-bold text-visage-100 mb-4">
          Available Styles
        </h2>
        <p className="text-visage-400 mb-10 max-w-xl mx-auto">
          Choose from multiple professional styles tailored for different contexts
        </p>
        
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { name: "Corporate", desc: "LinkedIn & Business" },
            { name: "Studio", desc: "Dramatic Lighting" },
            { name: "Natural", desc: "Outdoor & Warm" },
            { name: "Executive", desc: "C-Suite Ready" },
            { name: "Creative", desc: "Modern & Artistic" },
          ].map((style, i) => (
            <div 
              key={style.name}
              className="glass-card p-6 hover:border-accent-500/30 transition-all cursor-pointer"
            >
              <div className="w-full aspect-square rounded-lg bg-visage-800 mb-4 flex items-center justify-center">
                <span className="text-4xl text-visage-600">
                  {style.name[0]}
                </span>
              </div>
              <h4 className="font-semibold text-visage-100">{style.name}</h4>
              <p className="text-sm text-visage-500">{style.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="glass-card p-12 text-center glow-accent">
        <h2 className="text-3xl font-bold text-visage-100 mb-4">
          Ready to Get Started?
        </h2>
        <p className="text-visage-400 mb-8 max-w-lg mx-auto">
          Create your first headshot pack in minutes. No cloud costs, 
          no subscriptions — all on your own infrastructure.
        </p>
        <a href="/packs/new" className="btn-primary text-lg px-10 py-4">
          Create Your First Pack
          <ArrowRight className="w-5 h-5 ml-2" />
        </a>
      </section>
    </div>
  );
}
