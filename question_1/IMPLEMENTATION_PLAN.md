# Implementation Plan: Image Catalog, Object Detection, Metadata & Composite Image Generation

## Executive Summary
This document provides a production-grade, step-by-step implementation plan for the **Image Catalog System** defined in [`question-1-image-catalog.md`](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question_1/question-1-image-catalog.md). The solution delivers:
1. **Catalog & Metadata Storage**: Reusable image catalog with strict separation of auto-detected vs. user-modified fields.
2. **Dual-Domain Object Detection**: Accurate detection across both realistic/photographic images and stylized/cartoon illustrations, plus geometric shapes and color swatches.
3. **Manual Correction Engine**: Source-of-truth preservation for manual edits until explicit re-runs.
4. **Deterministic Composite Generator**: Natural-language driven image composition utilizing actual catalog assets (no silent synthetic hallucination), with spatial positioning, scaling, and layering.
5. **Validation Suite**: Complete test suite and test data covering all 8 specified test catalog images and 7 scenarios.

---

## 1. System Architecture & Technology Selection

### 1.1 Architecture Diagram

```mermaid
graph TD
    UI[Web UI / CLI Client] --> API[FastAPI / Application Service Layer]
    
    subgraph Storage & Catalog Layer
        API --> CatalogRepo[Image Catalog Repository (SQLite / JSON Store)]
        CatalogRepo --> ImageAssets[Local Image Storage (media/)]
        CatalogRepo --> MetaDB[Metadata DB (Image & Object Level)]
    end

    subgraph Object Detection & Analysis Engine
        API --> DetEngine[Detection Engine Orchestrator]
        DetEngine --> ZeroShot[Open-Vocabulary / Multimodal Vision Model (YOLO-World / OWL-ViT / Gemini)]
        DetEngine --> ShapeColorCV[OpenCV Contour & Color Analyzer (Shapes & Swatches)]
        DetEngine --> ColorExtractor[K-Means / Color Palette Extractor]
    end

    subgraph Composite Generation Pipeline
        API --> CompEngine[Composite Generator Engine]
        CompEngine --> NLParser[Prompt NLP / LLM Spatial Planner]
        CompEngine --> AssetMatcher[Catalog Asset Retriever]
        CompEngine --> SegExtractor[Asset Cropper / Segmenter (rembg / Alpha / Mask)]
        CompEngine --> CanvasRenderer[Pillow / OpenCV Composition & Layering Engine]
        CanvasRenderer --> CatalogRepo
    end
```

### 1.2 Technology Choices & Trade-off Analysis

| Component | Recommended Choice | Alternatives | Justification & Trade-offs |
|---|---|---|---|
| **Core Language & Backend** | **Python 3.12+ (FastAPI)** | Node.js / Go | Python has the native ecosystem for Computer Vision (OpenCV, PIL), ML models, and image composition. FastAPI provides async REST endpoints and automatic OpenAPI docs. |
| **Object Detection Engine** | **Hybrid: Open-Vocabulary Model (YOLO-World / OWL-ViT / Vision LLM) + OpenCV** | Standard YOLOv8 (COCO 80) | **Standard YOLO fails on cartoon images, geometric shapes, and swatches.** Open-vocabulary detection or a multimodal vision API handles arbitrary categories, cartoon characters (boy, girl), and color regions seamlessly. OpenCV contour/color heuristic acts as a lightning-fast local fallback for basic shapes & swatches. |
| **Color Extraction** | **K-Means / ColorThief / HSV Quantization** | Raw dominant pixel count | Robust against lighting variations, shadows, and anti-aliasing artifacts on borders. |
| **Object Segmentation / Cutout** | **`rembg` (U2-Net / BiRefNet) + Alpha channel support** | Simple bounding-box crop | Bounding-box crop leaves background artifacts during compositing; automatic alpha cutout extracts clean foreground items (e.g. apple, boy) for realistic compositing. |
| **Spatial Planning for Composites** | **Structured LLM / Rule-based Rule Engine** | Pure hardcoded heuristics | Natural language prompt like *"boy on the left, girl on the right, shapes above"* requires extracting entity, count, relative position (`left`, `right`, `center`, `above`, `below`), scale, and z-index. |
| **Composition & Layering** | **Pillow (PIL) + NumPy / OpenCV** | ImageMagick / Canvas API | Native Python image processing, sub-pixel positioning, alpha blending, shadow addition, and scaling. |
| **Catalog & Metadata DB** | **SQLite (via SQLAlchemy/Pydantic)** | Raw JSON files / PostgreSQL | Zero-setup, single-file ACID persistence with JSON schema validation. Easy to inspect, migrate, and maintain source-of-truth flags. |
| **Frontend / Visualizer** | **Lightweight Interactive Web App (HTML/CSS/JS + Canvas) or Streamlit** | CLI only | Allows drag-and-drop bounding box adjustment, manual additions, visual inspection of detections, and interactive text-to-composite prompt testing. |

---

## 2. Metadata Schema Design

### 2.1 Metadata Model (`schema.py`)

A fundamental requirement is **clearly distinguishing automatically generated fields from user-editable fields**, and ensuring manual corrections remain the source of truth.

```json
{
  "image_id": "img_001_apple",
  "filename": "apple.png",
  "original_path": "media/catalog/apple.png",
  "dimensions": { "width": 800, "height": 600 },
  "created_at": "2026-10-11T01:30:00Z",
  "updated_at": "2026-10-11T01:35:00Z",
  "detection_status": "MANUALLY_EDITED", 
  "is_composite": false,
  "composition_metadata": null,
  "objects": [
    {
      "object_id": "obj_001",
      "bbox": { "x": 120, "y": 80, "width": 350, "height": 380 },
      "label": "apple",
      "display_name": "Red Apple",
      "color": "red",
      "color_hex": "#D12121",
      "confidence": 0.94,
      
      "provenance": {
        "source": "manual",              
        "is_user_edited": true,           
        "is_user_added": false,          
        "auto_detected_values": {        
          "label": "apple",
          "color": "dark red",
          "confidence": 0.88,
          "bbox": { "x": 115, "y": 75, "width": 360, "height": 390 }
        },
        "last_edited_at": "2026-10-11T01:35:00Z"
      }
    }
  ]
}
```

### 2.2 Source-of-Truth Rules
1. **Field-level and Object-level Provenance**: Each object has `source: "auto" | "manual"`.
2. **Persistence Guarantee**: If `detection_status == "MANUALLY_EDITED"` or any object has `is_user_edited: true`, running regular catalog reload will **never** overwrite manual data.
3. **Explicit Rerun Flag**: Auto-detection will only replace manual edits if explicitly called with `force_rerun=True` (and the system can optionally archive the previous manual edits in a history log).

---

## 3. Step-by-Step Implementation Process

### Phase 1: Project Setup & Core Domain Models
- **Step 1.1: Directory Layout & Virtual Environment Setup**
  ```text
  question_1/
  ├── app/
  │   ├── __init__.py
  │   ├── config.py                 # Paths, model configurations, API keys
  │   ├── models/                   # Pydantic schemas & SQLite models
  │   │   ├── catalog.py
  │   │   └── detection.py
  │   ├── services/
  │   │   ├── catalog_service.py    # Catalog CRUD & metadata lifecycle
  │   │   ├── detector_service.py   # Dual-mode object detector
  │   │   ├── color_service.py      # Dominant color & swatch classifier
  │   │   └── composite_service.py  # NLP parser, asset retrieval & canvas compositor
  │   ├── api/                      # REST API endpoints (FastAPI)
  │   └── static/                   # Interactive Web UI (HTML5 Canvas bbox editor & prompt generator)
  ├── data/
  │   ├── catalog/                  # Source images
  │   ├── composites/               # Generated composite outputs
  │   └── catalog.db                # SQLite database / JSON metadata
  ├── tests/
  │   ├── test_scenarios.py         # Automated test cases 1 through 7
  │   └── generate_test_assets.py   # Synthetic asset generator for the 8 test images
  ├── requirements.txt
  └── README.md
  ```
- **Step 1.2: Dependencies Installation**
  - Web & DB: `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy`, `aiofiles`.
  - Computer Vision & Graphics: `opencv-python-headless`, `pillow`, `numpy`, `rembg`, `scikit-learn` (for K-Means color clustering).
  - Open-Vocabulary / Multimodal AI: Support for local `owlvit` (Hugging Face transformers) or hosted Multimodal API (`google-genai` / OpenAI) with graceful fallback.

---

### Phase 2: Dual-Domain Object Detection & Color Analysis
- **Step 2.1: Open-Vocabulary / Stylized Model Integration**
  - Implement `DetectorService` capable of detecting both:
    1. Realistic objects (`apple`, `orange`, realistic humans).
    2. Stylized/cartoon characters (`boy`, `girl` cartoons).
  - Use open-vocabulary prompts or zero-shot prompts to detect labels without COCO 80 limitation.
- **Step 2.2: Geometric Shape & Color Swatch Detection (OpenCV)**
  - For images like `shapes` and `colors`, implement a deterministic contour analysis:
    - Binarization & contour approximation (approxPolyDP) -> classifies `triangle` (3 vertices), `square`/`rectangle` (4 vertices, aspect ratio check), `circle` (circularity ratio check).
    - Color segmentation: Convert bounding region to HSV/Lab, cluster dominant hues, map to standard color names (`red`, `blue`, `green`, `yellow`, `purple`, etc.).
- **Step 2.3: Color Extraction Service**
  - Run K-Means clustering (K=3) on the bounding box pixels (ignoring white/transparent background pixels) to identify the primary dominant color and hex code.
- **Step 2.4: Unified Detection Output**
  - Merge model outputs into standard format: `label`, `color`, `confidence`, `bbox: {x, y, width, height}`.
  - Compute detection status: `SUCCESS` (objects found) or `NO_OBJECTS_DETECTED` (when empty).

---

### Phase 3: Metadata Management & Manual Correction Lifecycle
- **Step 3.1: Catalog Ingestion & Automatic Indexing**
  - `add_image(file_path, display_name=None)`: saves image, extracts image dimensions, automatically executes detection engine, tags fields with `source: "auto"`.
- **Step 3.2: Manual Editing & Correction API**
  - `add_object(image_id, object_data)`: manually insert a missed object (`source: "manual"`, `is_user_added: true`).
  - `update_object(image_id, object_id, updates)`: edit bbox, label, color, or confidence. Preserves auto-detected values in `auto_detected_values` snapshot while making user edits the active source of truth. Sets image `detection_status = "MANUALLY_EDITED"`.
  - `delete_object(image_id, object_id)`: remove false positive detections.
- **Step 3.3: Persistence & Rerun Policy Verification**
  - `rerun_detection(image_id, force=False)`:
    - If `force=False` and image is `MANUALLY_EDITED`, reject or return existing manual data.
    - If `force=True`, re-execute detection engine and re-baseline metadata.

---

### Phase 4: Composite Image Generation Pipeline
- **Step 4.1: Natural-Language Instruction Parsing**
  - Create `NLParser` that interprets prompts:
    - e.g. *"Create an image containing one red apple and one orange"* -> targets: `[{"query": "red apple", "qty": 1}, {"query": "orange", "qty": 1}]`.
    - e.g. *"A boy standing on the left, a girl on the right, and colored shapes above"* -> extracts spatial arrangement, scale, and z-index ordering.
- **Step 4.2: Catalog Asset Retrieval (Strict Asset Match)**
  - Query catalog for existing images matching requested items.
  - Constraint validation: Ensure target image contains the requested object. **Raise error if asset missing; do NOT substitute with synthetic hallucinations.**
- **Step 4.3: Asset Extraction & Cutout**
  - Extract the specific detected object region from the catalog source image using its bounding box.
  - Apply background removal (`rembg` or alpha masking) so objects seamlessly blend onto new composite backgrounds without white square artifacts.
- **Step 4.4: Composition, Positioning, Layering & Scaling**
  - Create high-resolution canvas (e.g. 1200x800, white/neutral or user-specified background).
  - Solve layout constraints:
    - Side-by-side positioning (`left`, `center`, `right`).
    - Stacking (`above`, `below`).
    - Scale normalization (so an apple isn't 10x bigger than a boy).
    - Z-index layering (handling overlaps without unwanted clipping).
  - Alpha-composite layers onto the final image.
- **Step 4.5: Register Composite as New Catalog Entry**
  - Save composite output file to `data/composites/`.
  - Ingest into catalog with `is_composite: true`, linking parent source image IDs, prompt used, and generated objects metadata.

---

### Phase 5: Test Data Creation & Automated Verification
- **Step 5.1: Test Asset Generation (`generate_test_assets.py`)**
  - Programmatically create or fetch standard benchmark images for the 8 required items:
    1. `apple.png`: Single red apple on clean background.
    2. `orange.png`: Single orange fruit on clean background.
    3. `boy.png`: Cartoon illustration of a standing boy with colored clothes.
    4. `girl.png`: Cartoon illustration of a standing girl with colored clothes.
    5. `shapes.png`: Geometric circle, square, triangle, and rectangle.
    6. `colors.png`: Color swatches (red, blue, green, yellow, purple).
    7. `apple_orange.png`: Side-by-side apple and orange.
    8. `overlapping_shapes.png`: Multiple overlapping colored shapes.
    9. `empty_white.png`: Blank image to test `no detectable objects` scenario.
- **Step 5.2: Automated Scenario Test Suite (`test_scenarios.py`)**
  - Implement tests directly validating the 7 scenarios:
    - **Scenario 1**: Object detection on catalog images (names, colors, scores, bboxes).
    - **Scenario 2**: Manual object addition on missed detection.
    - **Scenario 3**: Object editing (label, color, bbox, confidence).
    - **Scenario 4**: Persistence verification (ensure edits persist across loads; rerun blocked unless `force=True`).
    - **Scenario 5**: Composite generation with `apple` and `orange`.
    - **Scenario 6**: Composite generation with `boy`, `girl`, and `shapes`.
    - **Scenario 7**: Handling overlapping objects and empty/undetectable images.

---

### Phase 6: Interactive Web UI & Documentation
- **Step 6.1: Interactive Canvas UI**
  - Visual gallery of catalog images.
  - Image inspector with bounding box overlays, color tags, and confidence chips.
  - Interactive bounding-box drawer/editor for manual correction.
  - Composite Generation Studio: Prompt input, asset preview, generated composite viewer.
- **Step 6.2: Comprehensive Documentation & Walkthrough**
  - Architecture breakdown, algorithm decisions, API schema, and test run results.

---

## 4. Key Milestones & Deliverables

```
[M1: Core Framework] ──> [M2: Dual Detectors] ──> [M3: Manual Correction] ──> [M4: Compositor] ──> [M5: Tests & UI]
```

1. **Milestone 1**: Pydantic schemas, SQLite catalog repository, and asset storage system.
2. **Milestone 2**: Detection pipeline (realistic + cartoon + shapes/swatches) with color extraction.
3. **Milestone 3**: Metadata editing API with persistence rules and manual source-of-truth protection.
4. **Milestone 4**: Natural language prompt parser, asset segmenter, and Pillow canvas compositor.
5. **Milestone 5**: 8 benchmark test images, automated test suite covering all 7 scenarios, and interactive UI.
