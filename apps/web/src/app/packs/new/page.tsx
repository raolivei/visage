"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  X,
  AlertCircle,
  CheckCircle,
  Loader2,
  ArrowRight,
  Image as ImageIcon,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { api, StylePreset } from "@/lib/api";

const STYLE_PRESETS: StylePreset[] = [
  {
    id: "corporate",
    name: "Corporate / LinkedIn",
    description: "Professional business headshot",
  },
  {
    id: "studio",
    name: "Studio Portrait",
    description: "Dramatic lighting, dark background",
  },
  {
    id: "natural",
    name: "Natural Light",
    description: "Warm, approachable outdoor style",
  },
  {
    id: "executive",
    name: "Executive",
    description: "Premium C-suite ready portrait",
  },
  {
    id: "creative",
    name: "Creative Professional",
    description: "Modern artistic style",
  },
];

interface UploadedFile {
  file: File;
  preview: string;
  status: "pending" | "uploading" | "uploaded" | "error";
  error?: string;
}

export default function NewPackPage() {
  const router = useRouter();
  const [step, setStep] = useState<"upload" | "style" | "review">("upload");
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [selectedStyles, setSelectedStyles] = useState<string[]>(["corporate"]);
  const [removeWatermarks, setRemoveWatermarks] = useState(false);
  const [, setPackId] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
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

  // Toggle style selection
  const toggleStyle = (styleId: string) => {
    setSelectedStyles((prev) =>
      prev.includes(styleId)
        ? prev.filter((s) => s !== styleId)
        : [...prev, styleId],
    );
  };

  // Create pack and upload photos
  const handleCreatePack = async () => {
    if (files.length < 8) {
      setError("Please upload at least 8 photos");
      return;
    }

    if (selectedStyles.length === 0) {
      setError("Please select at least one style");
      return;
    }

    try {
      setIsCreating(true);
      setError(null);

      // Create pack
      const pack = await api.createPack(selectedStyles);
      setPackId(pack.id);

      // Upload photos
      setFiles((prev) =>
        prev.map((f) => ({ ...f, status: "uploading" as const })),
      );

      const result = await api.uploadPhotos(
        pack.id,
        files.map((f) => f.file),
        { removeWatermarks },
      );

      // Update file statuses
      setFiles((prev) =>
        prev.map((f, i) => ({
          ...f,
          status: result.errors.some((e) => e.includes(f.file.name))
            ? ("error" as const)
            : ("uploaded" as const),
          error: result.errors.find((e) => e.includes(f.file.name)),
        })),
      );

      if (result.uploaded > 0) {
        // Navigate to pack page
        router.push(`/packs/${pack.id}`);
      } else {
        setError("No photos were uploaded successfully");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create pack");
      setIsCreating(false);
    }
  };

  return (
    <div className="container mx-auto px-6 py-12 max-w-4xl">
      {/* Progress steps */}
      <div className="flex items-center justify-center mb-12">
        {["upload", "style", "review"].map((s, i) => (
          <div key={s} className="flex items-center">
            <div
              className={cn(
                "w-10 h-10 rounded-full flex items-center justify-center font-semibold",
                step === s
                  ? "bg-accent-500 text-visage-950"
                  : ["upload", "style", "review"].indexOf(step) > i
                    ? "bg-accent-500/20 text-accent-400"
                    : "bg-visage-800 text-visage-500",
              )}
            >
              {i + 1}
            </div>
            {i < 2 && (
              <div
                className={cn(
                  "w-24 h-0.5 mx-2",
                  ["upload", "style", "review"].indexOf(step) > i
                    ? "bg-accent-500"
                    : "bg-visage-800",
                )}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step 1: Upload */}
      {step === "upload" && (
        <div className="animate-fade-in">
          <h1 className="text-3xl font-bold text-visage-100 mb-2 text-center">
            Upload Your Photos
          </h1>
          <p className="text-visage-400 mb-8 text-center max-w-lg mx-auto">
            Upload 8-20 photos of yourself. Mix different angles, expressions,
            and lighting for the best results.
          </p>

          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={cn("dropzone mb-6", isDragActive && "active")}
          >
            <input {...getInputProps()} />
            <Upload className="w-12 h-12 text-visage-500 mb-4" />
            <p className="text-lg text-visage-200 mb-2">
              {isDragActive
                ? "Drop your photos here..."
                : "Drag & drop photos here"}
            </p>
            <p className="text-visage-500">
              or click to browse • JPEG, PNG, WebP • Max 20MB each
            </p>
          </div>

          {/* Upload tips */}
          <div className="glass-card p-6 mb-6">
            <h3 className="font-semibold text-visage-100 mb-3">
              Photo Guidelines
            </h3>
            <ul className="grid md:grid-cols-2 gap-2 text-sm text-visage-400">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                Mix of angles (front, 3/4, profile)
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                Different expressions (neutral, smiling)
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                Various lighting conditions
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                Clear, sharp photos
              </li>
              <li className="flex items-center gap-2">
                <X className="w-4 h-4 text-red-400" />
                No sunglasses or hats
              </li>
              <li className="flex items-center gap-2">
                <X className="w-4 h-4 text-red-400" />
                No group photos
              </li>
            </ul>
          </div>

          {/* Watermark removal toggle */}
          <div className="glass-card p-6 mb-6">
            <label className="flex items-start gap-4 cursor-pointer">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  checked={removeWatermarks}
                  onChange={(e) => setRemoveWatermarks(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-visage-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent-500"></div>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className="w-4 h-4 text-accent-400" />
                  <span className="font-semibold text-visage-100">
                    Remove Watermarks
                  </span>
                </div>
                <p className="text-sm text-visage-400">
                  Enable this if your photos are from AI headshot generators and
                  have watermarks. We&apos;ll automatically detect and remove
                  them before training.
                </p>
              </div>
            </label>
          </div>

          {/* Preview grid */}
          {files.length > 0 && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-visage-100">
                  Selected Photos ({files.length}/20)
                </h3>
                <button
                  onClick={() => setFiles([])}
                  className="text-sm text-visage-400 hover:text-visage-200"
                >
                  Clear all
                </button>
              </div>
              <div className="grid grid-cols-4 md:grid-cols-6 gap-3">
                {files.map((f, i) => (
                  <div key={i} className="photo-card">
                    <img
                      src={f.preview}
                      alt={`Upload ${i + 1}`}
                      className="w-full h-full object-cover"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(i);
                      }}
                      className="absolute top-1 right-1 w-6 h-6 rounded-full bg-visage-900/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 text-red-400 mb-6">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}

          {/* Next button */}
          <div className="flex justify-end">
            <button
              onClick={() => setStep("style")}
              disabled={files.length < 8}
              className={cn(
                "btn-primary",
                files.length < 8 && "opacity-50 cursor-not-allowed",
              )}
            >
              Continue to Styles
              <ArrowRight className="w-5 h-5 ml-2" />
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Style Selection */}
      {step === "style" && (
        <div className="animate-fade-in">
          <h1 className="text-3xl font-bold text-visage-100 mb-2 text-center">
            Choose Your Styles
          </h1>
          <p className="text-visage-400 mb-8 text-center max-w-lg mx-auto">
            Select one or more styles for your headshots. Each style will
            generate multiple variations.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-8">
            {STYLE_PRESETS.map((style) => (
              <button
                key={style.id}
                onClick={() => toggleStyle(style.id)}
                className={cn(
                  "glass-card p-6 text-left transition-all",
                  selectedStyles.includes(style.id)
                    ? "border-accent-500 bg-accent-500/10"
                    : "hover:border-visage-600",
                )}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-visage-100 mb-1">
                      {style.name}
                    </h3>
                    <p className="text-sm text-visage-400">
                      {style.description}
                    </p>
                  </div>
                  <div
                    className={cn(
                      "w-6 h-6 rounded-full border-2 flex items-center justify-center",
                      selectedStyles.includes(style.id)
                        ? "border-accent-500 bg-accent-500"
                        : "border-visage-600",
                    )}
                  >
                    {selectedStyles.includes(style.id) && (
                      <CheckCircle className="w-4 h-4 text-visage-950" />
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>

          {/* Navigation */}
          <div className="flex justify-between">
            <button onClick={() => setStep("upload")} className="btn-secondary">
              Back
            </button>
            <button
              onClick={() => setStep("review")}
              disabled={selectedStyles.length === 0}
              className={cn(
                "btn-primary",
                selectedStyles.length === 0 && "opacity-50 cursor-not-allowed",
              )}
            >
              Review & Create
              <ArrowRight className="w-5 h-5 ml-2" />
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Review */}
      {step === "review" && (
        <div className="animate-fade-in">
          <h1 className="text-3xl font-bold text-visage-100 mb-2 text-center">
            Review Your Pack
          </h1>
          <p className="text-visage-400 mb-8 text-center max-w-lg mx-auto">
            Confirm your selections and create your headshot pack.
          </p>

          <div className="glass-card p-8 mb-8">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Photos summary */}
              <div>
                <h3 className="font-semibold text-visage-100 mb-4 flex items-center gap-2">
                  <ImageIcon className="w-5 h-5 text-accent-400" />
                  Photos ({files.length})
                </h3>
                <div className="grid grid-cols-4 gap-2">
                  {files.slice(0, 8).map((f, i) => (
                    <div
                      key={i}
                      className="aspect-square rounded-lg overflow-hidden"
                    >
                      <img
                        src={f.preview}
                        alt={`Preview ${i + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ))}
                </div>
                {files.length > 8 && (
                  <p className="text-sm text-visage-500 mt-2">
                    +{files.length - 8} more photos
                  </p>
                )}
              </div>

              {/* Styles summary */}
              <div>
                <h3 className="font-semibold text-visage-100 mb-4">
                  Selected Styles ({selectedStyles.length})
                </h3>
                <div className="space-y-2">
                  {selectedStyles.map((styleId) => {
                    const style = STYLE_PRESETS.find((s) => s.id === styleId);
                    return (
                      <div
                        key={styleId}
                        className="flex items-center gap-3 p-3 bg-visage-800/50 rounded-lg"
                      >
                        <CheckCircle className="w-5 h-5 text-accent-400" />
                        <span className="text-visage-100">{style?.name}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Estimate */}
            <div className="mt-8 pt-6 border-t border-visage-800">
              <p className="text-visage-400">
                <strong className="text-visage-100">Estimated time:</strong>{" "}
                {removeWatermarks ? "20-35" : "15-30"} minutes for{" "}
                {removeWatermarks && "watermark removal, "}training and
                generation
              </p>
              <p className="text-visage-400 mt-1">
                <strong className="text-visage-100">Expected outputs:</strong> ~
                {selectedStyles.length * 20} images (best 5-10 per style)
              </p>
              {removeWatermarks && (
                <p className="text-accent-400 mt-2 flex items-center gap-2">
                  <Sparkles className="w-4 h-4" />
                  Watermark removal enabled
                </p>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 text-red-400 mb-6 justify-center">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}

          {/* Navigation */}
          <div className="flex justify-between">
            <button
              onClick={() => setStep("style")}
              disabled={isCreating}
              className="btn-secondary"
            >
              Back
            </button>
            <button
              onClick={handleCreatePack}
              disabled={isCreating}
              className="btn-primary"
            >
              {isCreating ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Creating Pack...
                </>
              ) : (
                <>
                  Create Pack
                  <ArrowRight className="w-5 h-5 ml-2" />
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
