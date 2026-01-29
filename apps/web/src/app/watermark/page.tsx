"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  X,
  AlertCircle,
  CheckCircle,
  Loader2,
  Download,
  Trash2,
  Sparkles,
  Image as ImageIcon,
  ArrowRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

interface UploadedFile {
  file: File;
  preview: string;
  status: "pending" | "uploading" | "processing" | "completed" | "error";
  error?: string;
  cleanedUrl?: string;
}

interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  output_urls: string[];
  errors: string[];
}

export default function WatermarkPage() {
  const router = useRouter();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Dropzone configuration
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
      status: "pending" as const,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/webp": [".webp"],
    },
    maxSize: 20 * 1024 * 1024, // 20MB
    multiple: true,
    maxFiles: 20,
  });

  // Remove file
  const removeFile = (index: number) => {
    setFiles((prev) => {
      const newFiles = [...prev];
      URL.revokeObjectURL(newFiles[index].preview);
      newFiles.splice(index, 1);
      return newFiles;
    });
  };

  // Clear all files
  const clearAll = () => {
    files.forEach((f) => URL.revokeObjectURL(f.preview));
    setFiles([]);
    setJobId(null);
    setJobStatus(null);
    setError(null);
  };

  // Poll for job status
  useEffect(() => {
    if (!jobId || !isProcessing) return;

    const pollInterval = setInterval(async () => {
      try {
        const status = await api.getWatermarkStatus(jobId);
        setJobStatus(status);

        if (status.status === "completed") {
          setIsProcessing(false);
          setFiles((prev) =>
            prev.map((f, i) => ({
              ...f,
              status: "completed" as const,
              cleanedUrl: status.output_urls[i] || undefined,
            })),
          );
        } else if (status.status === "failed") {
          setIsProcessing(false);
          setError("Watermark removal failed. Please try again.");
          setFiles((prev) =>
            prev.map((f) => ({ ...f, status: "error" as const })),
          );
        }
      } catch (err) {
        console.error("Failed to poll status:", err);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [jobId, isProcessing]);

  // Process images
  const handleProcess = async () => {
    if (files.length === 0) {
      setError("Please upload at least one image");
      return;
    }

    try {
      setIsProcessing(true);
      setError(null);

      // Update file statuses
      setFiles((prev) =>
        prev.map((f) => ({ ...f, status: "uploading" as const })),
      );

      // Upload and queue processing
      const result = await api.removeWatermarks(files.map((f) => f.file));
      setJobId(result.job_id);

      setFiles((prev) =>
        prev.map((f) => ({ ...f, status: "processing" as const })),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process images");
      setIsProcessing(false);
      setFiles((prev) => prev.map((f) => ({ ...f, status: "error" as const })));
    }
  };

  // Download results
  const handleDownload = () => {
    if (jobId) {
      window.open(api.getWatermarkDownloadUrl(jobId), "_blank");
    }
  };

  // Use in new pack
  const handleUseInPack = () => {
    // Store job info and redirect to pack creation
    if (jobId && jobStatus?.output_urls) {
      sessionStorage.setItem(
        "watermark_results",
        JSON.stringify({
          job_id: jobId,
          output_urls: jobStatus.output_urls,
        }),
      );
      router.push("/packs/new");
    }
  };

  const isComplete = jobStatus?.status === "completed";
  const hasFiles = files.length > 0;

  return (
    <div className="container mx-auto px-6 py-12 max-w-4xl">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent-500/10 mb-6">
          <Sparkles className="w-8 h-8 text-accent-400" />
        </div>
        <h1 className="text-3xl font-bold text-visage-100 mb-3">
          Remove Watermarks
        </h1>
        <p className="text-visage-400 max-w-lg mx-auto">
          Upload AI-generated headshots with watermarks and we&apos;ll clean
          them for you using advanced inpainting technology.
        </p>
      </div>

      {/* Upload area */}
      {!isComplete && (
        <div
          {...getRootProps()}
          className={cn(
            "dropzone mb-6",
            isDragActive && "active",
            isProcessing && "pointer-events-none opacity-50",
          )}
        >
          <input {...getInputProps()} disabled={isProcessing} />
          <Upload className="w-12 h-12 text-visage-500 mb-4" />
          <p className="text-lg text-visage-200 mb-2">
            {isDragActive
              ? "Drop your images here..."
              : "Drag & drop watermarked images"}
          </p>
          <p className="text-visage-500">
            or click to browse • JPEG, PNG, WebP • Max 20MB each • Up to 20
            images
          </p>
        </div>
      )}

      {/* Info card */}
      {!hasFiles && !isComplete && (
        <div className="glass-card p-6 mb-6">
          <h3 className="font-semibold text-visage-100 mb-3">How it works</h3>
          <ul className="space-y-2 text-sm text-visage-400">
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent-400" />
              Upload images from AI headshot generators with watermarks
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent-400" />
              Our AI automatically detects watermark regions
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent-400" />
              LaMa inpainting removes watermarks while preserving the image
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent-400" />
              Download cleaned images or use them directly in a new pack
            </li>
          </ul>
        </div>
      )}

      {/* File preview grid */}
      {hasFiles && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-visage-100">
              {isComplete ? "Cleaned Images" : "Selected Images"} (
              {files.length})
            </h3>
            {!isProcessing && (
              <button
                onClick={clearAll}
                className="text-sm text-visage-400 hover:text-visage-200 flex items-center gap-1"
              >
                <Trash2 className="w-4 h-4" />
                Clear all
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {files.map((f, i) => (
              <div key={i} className="relative group">
                <div className="aspect-square rounded-lg overflow-hidden bg-visage-800">
                  <img
                    src={isComplete && f.cleanedUrl ? f.cleanedUrl : f.preview}
                    alt={`Image ${i + 1}`}
                    className="w-full h-full object-cover"
                  />

                  {/* Status overlay */}
                  {f.status === "processing" && (
                    <div className="absolute inset-0 bg-visage-950/80 flex items-center justify-center">
                      <Loader2 className="w-8 h-8 text-accent-400 animate-spin" />
                    </div>
                  )}

                  {f.status === "completed" && (
                    <div className="absolute top-2 right-2">
                      <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
                        <CheckCircle className="w-4 h-4 text-white" />
                      </div>
                    </div>
                  )}

                  {f.status === "error" && (
                    <div className="absolute inset-0 bg-red-950/80 flex items-center justify-center">
                      <AlertCircle className="w-8 h-8 text-red-400" />
                    </div>
                  )}
                </div>

                {/* Remove button (only when not processing) */}
                {!isProcessing && !isComplete && (
                  <button
                    onClick={() => removeFile(i)}
                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-visage-900/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-4 h-4 text-visage-200" />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Progress indicator */}
          {isProcessing && jobStatus && (
            <div className="mt-6">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-visage-400">
                  {jobStatus.status === "pending"
                    ? "Waiting to start..."
                    : "Removing watermarks..."}
                </span>
                <span className="text-visage-200">{jobStatus.progress}%</span>
              </div>
              <div className="h-2 bg-visage-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent-500 rounded-full transition-all duration-300"
                  style={{ width: `${jobStatus.progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 text-red-400 mb-6 p-4 bg-red-950/20 rounded-lg">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        {!isComplete && hasFiles && (
          <button
            onClick={handleProcess}
            disabled={isProcessing || files.length === 0}
            className={cn(
              "btn-primary",
              (isProcessing || files.length === 0) &&
                "opacity-50 cursor-not-allowed",
            )}
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Remove Watermarks
              </>
            )}
          </button>
        )}

        {isComplete && (
          <>
            <button onClick={handleDownload} className="btn-primary">
              <Download className="w-5 h-5 mr-2" />
              Download All
            </button>

            <button onClick={handleUseInPack} className="btn-secondary">
              <ImageIcon className="w-5 h-5 mr-2" />
              Use in New Pack
              <ArrowRight className="w-5 h-5 ml-2" />
            </button>

            <button onClick={clearAll} className="btn-ghost">
              <Trash2 className="w-5 h-5 mr-2" />
              Start Over
            </button>
          </>
        )}
      </div>
    </div>
  );
}
