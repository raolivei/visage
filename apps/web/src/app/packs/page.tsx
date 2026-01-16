"use client";

import { useEffect, useState } from "react";
import {
  Plus,
  FolderOpen,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { api, Pack } from "@/lib/api";

const statusConfig = {
  created: { icon: FolderOpen, color: "badge-pending", label: "Created" },
  uploading: { icon: Loader2, color: "badge-processing", label: "Uploading" },
  validating: { icon: Loader2, color: "badge-processing", label: "Validating" },
  training: { icon: Loader2, color: "badge-processing", label: "Training" },
  generating: { icon: Loader2, color: "badge-processing", label: "Generating" },
  filtering: { icon: Loader2, color: "badge-processing", label: "Filtering" },
  completed: { icon: CheckCircle, color: "badge-success", label: "Completed" },
  failed: { icon: AlertCircle, color: "badge-error", label: "Failed" },
};

export default function PacksPage() {
  const [packs, setPacks] = useState<Pack[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPacks();
  }, []);

  async function loadPacks() {
    try {
      setLoading(true);
      const data = await api.listPacks();
      setPacks(data.packs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load packs");
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-16 text-center">
        <Loader2 className="w-8 h-8 animate-spin mx-auto text-accent-400" />
        <p className="mt-4 text-visage-400">Loading packs...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-6 py-16 text-center">
        <AlertCircle className="w-12 h-12 mx-auto text-red-400 mb-4" />
        <p className="text-visage-400">{error}</p>
        <button onClick={loadPacks} className="btn-secondary mt-4">
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-6 py-12">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-3xl font-bold text-visage-100 mb-2">
            My Headshot Packs
          </h1>
          <p className="text-visage-400">
            {packs.length} pack{packs.length !== 1 ? "s" : ""} created
          </p>
        </div>
        <a href="/packs/new" className="btn-primary">
          <Plus className="w-5 h-5 mr-2" />
          New Pack
        </a>
      </div>

      {/* Empty state */}
      {packs.length === 0 ? (
        <div className="glass-card p-16 text-center">
          <FolderOpen className="w-16 h-16 mx-auto text-visage-600 mb-6" />
          <h2 className="text-2xl font-semibold text-visage-100 mb-3">
            No packs yet
          </h2>
          <p className="text-visage-400 mb-8 max-w-md mx-auto">
            Create your first headshot pack to get started. Upload your photos
            and let our AI generate professional headshots.
          </p>
          <a href="/packs/new" className="btn-primary">
            <Plus className="w-5 h-5 mr-2" />
            Create Your First Pack
          </a>
        </div>
      ) : (
        /* Pack grid */
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {packs.map((pack) => {
            const status =
              statusConfig[pack.status as keyof typeof statusConfig] ||
              statusConfig.created;
            const StatusIcon = status.icon;
            const isProcessing = [
              "uploading",
              "validating",
              "training",
              "generating",
              "filtering",
            ].includes(pack.status);

            return (
              <a
                key={pack.id}
                href={`/packs/${pack.id}`}
                className="glass-card p-6 hover:border-accent-500/30 transition-all group"
              >
                {/* Preview grid */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="aspect-square rounded-lg bg-visage-800 flex items-center justify-center"
                    >
                      <span className="text-visage-600 text-xs">
                        {pack.output_count > i ? "🖼️" : ""}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Pack info */}
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-visage-100 group-hover:text-accent-400 transition-colors">
                      Pack #{pack.id.slice(0, 8)}
                    </h3>
                    <div className="flex items-center gap-2 mt-1 text-sm text-visage-500">
                      <Clock className="w-4 h-4" />
                      <span>
                        {new Date(pack.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  <div className={cn("badge", status.color)}>
                    <StatusIcon
                      className={cn(
                        "w-3.5 h-3.5 mr-1",
                        isProcessing && "animate-spin"
                      )}
                    />
                    {status.label}
                  </div>
                </div>

                {/* Stats */}
                <div className="flex items-center gap-4 mt-4 pt-4 border-t border-visage-800">
                  <div className="text-sm">
                    <span className="text-visage-400">Photos:</span>{" "}
                    <span className="text-visage-100">{pack.photo_count}</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-visage-400">Outputs:</span>{" "}
                    <span className="text-visage-100">{pack.output_count}</span>
                  </div>
                  {pack.style_preset && (
                    <div className="text-sm ml-auto">
                      <span className="text-accent-400 capitalize">
                        {pack.style_preset}
                      </span>
                    </div>
                  )}
                </div>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}
