"use client";

import { useEffect, useState, useCallback } from "react";
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
  RefreshCw,
  Timer,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { api, Pack, Photo, Job, Output } from "@/lib/api";

// Parse ETA from current_step (e.g., "Generating executive (15/20)")
function parseProgressInfo(currentStep: string | null): {
  currentStyle?: string;
  currentNum?: number;
  totalNum?: number;
} {
  if (!currentStep) return {};
  
  // Match "Generating {style} ({num}/{total})"
  const match = currentStep.match(/Generating (\w+) \((\d+)\/(\d+)\)/i);
  if (match) {
    return {
      currentStyle: match[1],
      currentNum: parseInt(match[2]),
      totalNum: parseInt(match[3]),
    };
  }
  
  // Match "Training step {num}/{total}"
  const trainMatch = currentStep.match(/step (\d+)\/(\d+)/i);
  if (trainMatch) {
    return {
      currentNum: parseInt(trainMatch[1]),
      totalNum: parseInt(trainMatch[2]),
    };
  }
  
  return {};
}

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
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Memoize loadPackData to avoid useEffect dependency issues
  const loadPackData = useCallback(async () => {
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
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load pack");
    } finally {
      setLoading(false);
    }
  }, [packId]);

  // Load pack data and poll for updates
  useEffect(() => {
    loadPackData();

    // Poll more frequently when there's an active job
    const interval = setInterval(() => {
      const hasActiveJob = jobs.some(
        (j) => j.status === "pending" || j.status === "processing" || j.status === "running"
      );
      const isPackProcessing = pack && ["training", "generating", "filtering", "validating"].includes(pack.status);
      
      if (hasActiveJob || isPackProcessing) {
        loadPackData();
      }
    }, 3000); // Poll every 3 seconds for more responsive UI

    return () => clearInterval(interval);
  }, [packId, jobs, pack, loadPackData]);

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
    "processing",
  ].includes(pack.status);
  const canStartGeneration =
    pack.status === "uploading" || pack.status === "created" || pack.status === "ready";
  const currentJob = jobs.find(
    (j) => j.status === "running" || j.status === "pending" || j.status === "processing"
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
            <a
              href={api.getDownloadUrl(packId)}
              className="btn-primary inline-flex items-center"
              download
            >
              <Download className="w-5 h-5 mr-2" />
              Download ({selectedCount})
            </a>
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
                  {currentJob.current_step || "Initializing..."}
                </p>
                {(() => {
                  const info = parseProgressInfo(currentJob.current_step);
                  if (info.currentStyle) {
                    return (
                      <p className="text-accent-400 text-xs mt-1">
                        Style: <span className="capitalize font-medium">{info.currentStyle}</span>
                        {info.currentNum && info.totalNum && (
                          <span className="text-visage-500 ml-2">
                            Image {info.currentNum}/{info.totalNum}
                          </span>
                        )}
                      </p>
                    );
                  }
                  return null;
                })()}
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-accent-400">
                {currentJob.progress}%
              </div>
              <div className="w-40 h-2 bg-visage-800 rounded-full mt-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-accent-600 to-accent-400 transition-all duration-1000 ease-out"
                  style={{ width: `${currentJob.progress}%` }}
                />
              </div>
              <div className="flex items-center justify-end gap-2 mt-2 text-xs text-visage-500">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Updated {Math.round((Date.now() - lastUpdate.getTime()) / 1000)}s ago
              </div>
            </div>
          </div>
          
          {/* Progress stages for generation */}
          {currentJob.job_type === "generate" && (
            <div className="mt-4 pt-4 border-t border-visage-800">
              <div className="flex justify-between text-xs text-visage-500">
                <span className={cn(currentJob.progress >= 0 && "text-accent-400")}>
                  ● Initialize
                </span>
                <span className={cn(currentJob.progress >= 15 && "text-accent-400")}>
                  ● Generate
                </span>
                <span className={cn(currentJob.progress >= 80 && "text-accent-400")}>
                  ● Filter
                </span>
                <span className={cn(currentJob.progress >= 95 && "text-accent-400")}>
                  ● Save
                </span>
              </div>
            </div>
          )}
          
          {/* Progress stages for training */}
          {currentJob.job_type === "train" && (
            <div className="mt-4 pt-4 border-t border-visage-800">
              <div className="flex justify-between text-xs text-visage-500">
                <span className={cn(currentJob.progress >= 0 && "text-accent-400")}>
                  ● Load Model
                </span>
                <span className={cn(currentJob.progress >= 10 && "text-accent-400")}>
                  ● Prepare Data
                </span>
                <span className={cn(currentJob.progress >= 20 && "text-accent-400")}>
                  ● Train LoRA
                </span>
                <span className={cn(currentJob.progress >= 90 && "text-accent-400")}>
                  ● Save Weights
                </span>
              </div>
            </div>
          )}
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
                <div className="flex items-center gap-4">
                  <span className="text-sm text-visage-400">
                    {selectedCount} selected
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={async () => {
                        const unselected = outputs.filter(o => !o.is_selected);
                        if (unselected.length > 0) {
                          await api.selectOutputs(packId, unselected.map(o => o.id), true);
                          setOutputs(prev => prev.map(o => ({ ...o, is_selected: true })));
                        }
                      }}
                      className="text-xs text-accent-400 hover:text-accent-300"
                    >
                      Select All
                    </button>
                    <span className="text-visage-600">|</span>
                    <button
                      onClick={async () => {
                        const selected = outputs.filter(o => o.is_selected);
                        if (selected.length > 0) {
                          await api.selectOutputs(packId, selected.map(o => o.id), false);
                          setOutputs(prev => prev.map(o => ({ ...o, is_selected: false })));
                        }
                      }}
                      className="text-xs text-visage-400 hover:text-visage-300"
                    >
                      Deselect All
                    </button>
                  </div>
                </div>
              )}
            </div>

            {outputs.length === 0 ? (
              <div className="text-center py-12">
                {isProcessing ? (
                  <>
                    <div className="relative w-16 h-16 mx-auto mb-4">
                      <Sparkles className="w-16 h-16 text-accent-400 animate-pulse" />
                      <div className="absolute inset-0 rounded-full border-2 border-accent-500/30 animate-ping" />
                    </div>
                    <p className="text-visage-300 font-medium">
                      Generating your AI headshots...
                    </p>
                    <p className="text-visage-500 text-sm mt-2">
                      This takes about 15-30 minutes. You can leave this page and come back later.
                    </p>
                    <p className="text-accent-400 text-xs mt-4">
                      <Timer className="w-3 h-3 inline mr-1" />
                      Outputs will appear here as each style completes
                    </p>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-12 h-12 mx-auto text-visage-700 mb-4" />
                    <p className="text-visage-500">
                      No headshots generated yet
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
                    {canStartGeneration && photos.length < 8 && (
                      <p className="text-amber-400 text-sm mt-4">
                        Upload at least {8 - photos.length} more photos to start generation
                      </p>
                    )}
                  </>
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
