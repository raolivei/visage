/**
 * Visage API Client
 *
 * Type-safe client for the Visage API.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004";

// ============================================================================
// Types
// ============================================================================

export interface Pack {
  id: string;
  status: string;
  style_preset: string | null;
  style_presets: string[];
  trigger_token: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  photo_count: number;
  output_count: number;
}

export interface PackListResponse {
  packs: Pack[];
  total: number;
}

export interface Photo {
  id: string;
  pack_id: string;
  original_filename: string;
  quality_score: number | null;
  is_valid: string;
  face_detected: string;
  created_at: string;
}

export interface PhotoUploadResponse {
  uploaded: number;
  photos: Photo[];
  errors: string[];
}

export interface Job {
  id: string;
  pack_id: string;
  job_type: string;
  status: string;
  progress: number;
  current_step: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface Output {
  id: string;
  pack_id: string;
  style_preset: string | null;
  score: number | null;
  is_selected: boolean;
  created_at: string;
}

export interface OutputListResponse {
  outputs: Output[];
  total: number;
  selected_count: number;
}

export interface StylePreset {
  id: string;
  name: string;
  description: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  services: Record<string, string>;
}

// Watermark removal types
export interface WatermarkJobResponse {
  job_id: string;
  status: string;
  message: string;
  input_count: number;
  input_keys: string[];
}

export interface WatermarkStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  output_keys: string[];
  output_urls: string[];
  errors: string[];
}

// ============================================================================
// API Client
// ============================================================================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Health
  async health(): Promise<HealthResponse> {
    return this.fetch("/health");
  }

  // Packs
  async listPacks(limit = 20, offset = 0): Promise<PackListResponse> {
    return this.fetch(`/api/packs?limit=${limit}&offset=${offset}`);
  }

  async getPack(id: string): Promise<Pack> {
    return this.fetch(`/api/packs/${id}`);
  }

  async createPack(stylePresets: string[] = ["corporate"]): Promise<Pack> {
    return this.fetch("/api/packs", {
      method: "POST",
      body: JSON.stringify({ style_presets: stylePresets }),
    });
  }

  async deletePack(id: string): Promise<void> {
    await this.fetch(`/api/packs/${id}`, { method: "DELETE" });
  }

  // Photos
  async uploadPhotos(
    packId: string,
    files: File[],
    options: { removeWatermarks?: boolean } = {}
  ): Promise<PhotoUploadResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    const params = new URLSearchParams();
    if (options.removeWatermarks) {
      params.append("remove_watermarks", "true");
    }

    const url = `${this.baseUrl}/api/packs/${packId}/photos${params.toString() ? `?${params.toString()}` : ""}`;
    
    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Upload failed: ${response.status}`);
    }

    return response.json();
  }

  async listPhotos(packId: string): Promise<Photo[]> {
    return this.fetch(`/api/packs/${packId}/photos`);
  }

  // Jobs
  async startGeneration(
    packId: string,
    stylePresets?: string[],
    numImagesPerStyle = 20
  ): Promise<{ job: Job; message: string }> {
    return this.fetch(`/api/packs/${packId}/generate`, {
      method: "POST",
      body: JSON.stringify({
        style_presets: stylePresets,
        num_images_per_style: numImagesPerStyle,
      }),
    });
  }

  async listJobs(packId: string): Promise<Job[]> {
    return this.fetch(`/api/packs/${packId}/jobs`);
  }

  // Outputs
  async listOutputs(
    packId: string,
    includeFiltered = false
  ): Promise<OutputListResponse> {
    return this.fetch(
      `/api/packs/${packId}/outputs?include_filtered=${includeFiltered}`
    );
  }

  async selectOutputs(
    packId: string,
    outputIds: string[],
    selected = true
  ): Promise<{ updated: number }> {
    return this.fetch(`/api/packs/${packId}/outputs/select`, {
      method: "POST",
      body: JSON.stringify({ output_ids: outputIds, selected }),
    });
  }

  async getOutputUrl(
    packId: string,
    outputId: string
  ): Promise<{ url: string; expires_in: number }> {
    return this.fetch(`/api/packs/${packId}/outputs/${outputId}/url`);
  }

  /**
   * Get download URL for selected outputs as ZIP.
   */
  getDownloadUrl(packId: string, selectedOnly = true): string {
    return `${this.baseUrl}/api/packs/${packId}/outputs/download?selected_only=${selectedOnly}`;
  }

  // Styles
  async listStyles(): Promise<{ styles: StylePreset[] }> {
    return this.fetch("/api/packs/styles");
  }

  // =========================================================================
  // Watermark Removal
  // =========================================================================

  /**
   * Upload images and queue watermark removal.
   */
  async removeWatermarks(files: File[]): Promise<WatermarkJobResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    const response = await fetch(`${this.baseUrl}/api/watermark/remove`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(
        error.detail || `Watermark removal failed: ${response.status}`
      );
    }

    return response.json();
  }

  /**
   * Get status of a watermark removal job.
   */
  async getWatermarkStatus(jobId: string): Promise<WatermarkStatusResponse> {
    return this.fetch(`/api/watermark/status/${jobId}`);
  }

  /**
   * Get download URL for watermark removal results.
   */
  getWatermarkDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/api/watermark/download/${jobId}`;
  }

  /**
   * Delete a watermark removal job and its files.
   */
  async deleteWatermarkJob(jobId: string): Promise<void> {
    await this.fetch(`/api/watermark/job/${jobId}`, { method: "DELETE" });
  }
}

// Export singleton instance
export const api = new ApiClient();
