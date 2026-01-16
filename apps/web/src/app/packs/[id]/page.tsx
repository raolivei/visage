"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  Play,
  Download,
  Loader2,
  AlertCircle,
  CheckCircle,
  Clock,
  Image as ImageIcon,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { api, Pack, Photo, Job, Output } from "@/lib/api";

export default function PackDetailPage() {
  const params = useParams();
  const packId = params.id as string;

  const [pack, setPack] = useState<Pack | null>(null);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [outputs, setOutputs] = useState<Output[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  // Load pack data
  useEffect(() => {
    loadPackData();

    // Poll for updates if processing
    const interval = setInterval(() => {
      if (
        pack &&
        ["training", "generating", "filtering"].includes(pack.status)
      ) {
        loadPackData();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [packId]);

  async function loadPackData() {
    try {
      const [packData, photosData, jobsData, outputsData] = await Promise.all([
        api.getPack(packId),
        api.listPhotos(packId),
        api.listJobs(packId),
        api.listOutputs(packId),
      ]);

      setPack(packData);
      setPhotos(photosData);
      setJobs(jobsData);
      setOutputs(outputsData.outputs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load pack");
    } finally {
      setLoading(false);
    }
  }

  async function handleStartGeneration() {
    try {
      setIsStarting(true);
      setError(null);
      await api.startGeneration(packId);
      await loadPackData();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start generation"
      );
    } finally {
      setIsStarting(false);
    }
  }

  async function handleToggleSelect(
    outputId: string,
    currentSelected: boolean
  ) {
    try {
      await api.selectOutputs(packId, [outputId], !currentSelected);
      setOutputs((prev) =>
        prev.map((o) =>
          o.id === outputId ? { ...o, is_selected: !currentSelected } : o
        )
      );
    } catch (err) {
      console.error("Failed to toggle selection:", err);
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-16 text-center">
        <Loader2 className="w-8 h-8 animate-spin mx-auto text-accent-400" />
        <p className="mt-4 text-visage-400">Loading pack...</p>
      </div>
    );
  }

  if (error || !pack) {
    return (
      <div className="container mx-auto px-6 py-16 text-center">
        <AlertCircle className="w-12 h-12 mx-auto text-red-400 mb-4" />
        <p className="text-visage-400">{error || "Pack not found"}</p>
        <a href="/packs" className="btn-secondary mt-4 inline-flex">
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Packs
        </a>
      </div>
    );
  }

  const isProcessing = [
    "training",
    "generating",
    "filtering",
    "validating",
  ].includes(pack.status);
  const canStartGeneration =
    pack.status === "uploading" || pack.status === "created";
  const currentJob = jobs.find(
    (j) => j.status === "running" || j.status === "pending"
  );
  const selectedCount = outputs.filter((o) => o.is_selected).length;

  return (
    <div className="container mx-auto px-6 py-12">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <a
            href="/packs"
            className="inline-flex items-center text-visage-400 hover:text-visage-200 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Packs
          </a>
          <h1 className="text-3xl font-bold text-visage-100">
            Pack #{pack.id.slice(0, 8)}
          </h1>
          <div className="flex items-center gap-4 mt-2 text-visage-400">
            <span className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              {new Date(pack.created_at).toLocaleString()}
            </span>
            {pack.style_preset && (
              <span className="text-accent-400 capitalize">
                {pack.style_preset}
              </span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          {canStartGeneration && photos.length >= 8 && (
            <button
              onClick={handleStartGeneration}
              disabled={isStarting}
              className="btn-primary"
            >
              {isStarting ? (
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <Play className="w-5 h-5 mr-2" />
              )}
              Start Generation
            </button>
          )}
          {pack.status === "completed" && selectedCount > 0 && (
            <button className="btn-primary">
              <Download className="w-5 h-5 mr-2" />
              Download ({selectedCount})
            </button>
          )}
        </div>
      </div>

      {/* Status banner */}
      {isProcessing && currentJob && (
        <div className="glass-card p-6 mb-8 border-accent-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-accent-500/20 flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-accent-400 animate-pulse" />
              </div>
              <div>
                <h3 className="font-semibold text-visage-100">
                  {currentJob.job_type === "train"
                    ? "Training AI Model"
                    : currentJob.job_type === "generate"
                      ? "Generating Headshots"
                      : "Processing"}
                </h3>
                <p className="text-visage-400 text-sm">
                  {currentJob.current_step || "Please wait..."}
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-accent-400">
                {currentJob.progress}%
              </div>
              <div className="w-32 h-2 bg-visage-800 rounded-full mt-2 overflow-hidden">
                <div
                  className="h-full bg-accent-500 transition-all duration-500"
                  style={{ width: `${currentJob.progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error banner */}
      {pack.status === "failed" && (
        <div className="glass-card p-6 mb-8 border-red-500/30 bg-red-500/5">
          <div className="flex items-center gap-4">
            <AlertCircle className="w-8 h-8 text-red-400" />
            <div>
              <h3 className="font-semibold text-red-400">Generation Failed</h3>
              <p className="text-visage-400">
                {pack.error_message || "An error occurred during processing"}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Uploaded Photos */}
        <div className="lg:col-span-1">
          <div className="glass-card p-6">
            <h2 className="font-semibold text-visage-100 mb-4 flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-accent-400" />
              Uploaded Photos ({photos.length})
            </h2>

            {photos.length === 0 ? (
              <p className="text-visage-500 text-sm">No photos uploaded yet</p>
            ) : (
              <div className="grid grid-cols-3 gap-2">
                {photos.map((photo) => (
                  <div
                    key={photo.id}
                    className="aspect-square rounded-lg bg-visage-800 overflow-hidden relative"
                  >
                    <div className="w-full h-full flex items-center justify-center text-visage-600">
                      <ImageIcon className="w-6 h-6" />
                    </div>
                    {photo.is_valid === "valid" && (
                      <div className="absolute bottom-1 right-1">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      </div>
                    )}
                    {photo.is_valid === "invalid" && (
                      <div className="absolute bottom-1 right-1">
                        <AlertCircle className="w-4 h-4 text-red-400" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {photos.length < 8 && (
              <p className="text-amber-400 text-sm mt-4">
                Need at least 8 photos to start generation
              </p>
            )}
          </div>

          {/* Job History */}
          {jobs.length > 0 && (
            <div className="glass-card p-6 mt-6">
              <h2 className="font-semibold text-visage-100 mb-4">
                Job History
              </h2>
              <div className="space-y-3">
                {jobs.map((job) => (
                  <div
                    key={job.id}
                    className="flex items-center justify-between p-3 bg-visage-800/50 rounded-lg"
                  >
                    <div>
                      <p className="text-sm text-visage-100 capitalize">
                        {job.job_type}
                      </p>
                      <p className="text-xs text-visage-500">
                        {new Date(job.created_at).toLocaleString()}
                      </p>
                    </div>
                    <span
                      className={cn(
                        "badge",
                        job.status === "completed" && "badge-success",
                        job.status === "failed" && "badge-error",
                        job.status === "running" && "badge-processing",
                        job.status === "pending" && "badge-pending"
                      )}
                    >
                      {job.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Generated Outputs */}
        <div className="lg:col-span-2">
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-semibold text-visage-100 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-accent-400" />
                Generated Headshots ({outputs.length})
              </h2>
              {outputs.length > 0 && (
                <span className="text-sm text-visage-400">
                  {selectedCount} selected
                </span>
              )}
            </div>

            {outputs.length === 0 ? (
              <div className="text-center py-12">
                <Sparkles className="w-12 h-12 mx-auto text-visage-700 mb-4" />
                <p className="text-visage-500">
                  {isProcessing
                    ? "Generating headshots..."
                    : "No headshots generated yet"}
                </p>
                {canStartGeneration && photos.length >= 8 && (
                  <button
                    onClick={handleStartGeneration}
                    disabled={isStarting}
                    className="btn-primary mt-4"
                  >
                    Start Generation
                  </button>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {outputs.map((output) => (
                  <button
                    key={output.id}
                    onClick={() =>
                      handleToggleSelect(output.id, output.is_selected)
                    }
                    className={cn(
                      "photo-card aspect-square",
                      output.is_selected && "ring-2 ring-accent-500"
                    )}
                  >
                    <div className="w-full h-full flex items-center justify-center text-visage-600">
                      <ImageIcon className="w-8 h-8" />
                    </div>

                    {/* Selection indicator */}
                    <div
                      className={cn(
                        "absolute top-2 right-2 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all",
                        output.is_selected
                          ? "border-accent-500 bg-accent-500"
                          : "border-visage-500 bg-visage-900/50"
                      )}
                    >
                      {output.is_selected && (
                        <CheckCircle className="w-4 h-4 text-visage-950" />
                      )}
                    </div>

                    {/* Score */}
                    {output.score && (
                      <div className="absolute bottom-2 left-2 px-2 py-1 bg-visage-900/80 rounded text-xs text-visage-300">
                        {Math.round(output.score * 100)}%
                      </div>
                    )}

                    {/* Style */}
                    {output.style_preset && (
                      <div className="absolute bottom-2 right-2 px-2 py-1 bg-accent-500/20 rounded text-xs text-accent-400 capitalize">
                        {output.style_preset}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
