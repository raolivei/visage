# GitHub Issue: Watermark Removal Feature

**Create this issue manually at: https://github.com/raolivei/visage/issues/new**

---

## Title

Watermark Removal Feature

## Body

### Summary

Add watermark removal capability to Visage, allowing users to upload AI-generated headshots with watermarks and have them automatically cleaned.

### Use Case

Users can use external AI headshot tools (which often add watermarks) and then use Visage to:

1. Remove watermarks from the generated images
2. Use the cleaned images as training data for LoRA fine-tuning

### Implementation

- **Standalone Tool**: `/watermark` page for uploading images, removing watermarks, and downloading cleaned results
- **Pack Integration**: Toggle in pack creation flow to auto-remove watermarks during photo upload
- **Detection**: Automatic heuristic-based detection focusing on corners, semi-transparent overlays, and text patterns
- **Removal**: LaMa (Large Mask Inpainting) model via `simple-lama-inpainting` library

### Components Added

#### Worker (`apps/worker/`)

- `src/pipeline/watermark_remover.py` - WatermarkDetector and WatermarkRemover classes

#### API (`apps/api/`)

- `src/routes/watermark.py` - Endpoints for standalone watermark removal
- `src/services/queue.py` - Watermark job queue methods
- `src/models/photo.py` - New columns: `watermark_removed`, `original_s3_key`, `watermark_job_id`

#### Frontend (`apps/web/`)

- `src/app/watermark/page.tsx` - Standalone watermark removal tool
- `src/lib/api.ts` - API client methods for watermark endpoints
- `src/app/packs/new/page.tsx` - Toggle for watermark removal in pack creation

### API Endpoints

- `POST /api/watermark/remove` - Upload images for watermark removal
- `GET /api/watermark/status/{job_id}` - Poll job status
- `GET /api/watermark/download/{job_id}` - Download cleaned images as ZIP
- `DELETE /api/watermark/job/{job_id}` - Cleanup job data

### Status

- [x] Worker watermark removal pipeline
- [x] API endpoints
- [x] Database schema updates
- [x] Frontend standalone tool
- [x] Frontend pack integration
- [ ] End-to-end testing
- [ ] Documentation

### Labels

`enhancement`, `feature`
