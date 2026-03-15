# AI Trap — System Design Document

**Project:** ai-trap-gstr
**Date:** 2026-03-14
**Status:** Active development

---

## Table of Contents

1. [Purpose & Overview](#1-purpose--overview)
2. [Hardware Platforms](#2-hardware-platforms)
3. [Software Architecture](#3-software-architecture)
4. [Component Reference](#4-component-reference)
5. [Data Flow](#5-data-flow)
6. [Capture Sessions](#6-capture-sessions)
7. [File & Directory Layout](#7-file--directory-layout)
8. [Database Schema](#8-database-schema)
9. [REST API Reference](#9-rest-api-reference)
10. [Server-Sent Events](#10-server-sent-events)
11. [Configuration Reference](#11-configuration-reference)
12. [Build System](#12-build-system)

---

## 1. Purpose & Overview

AI Trap is an embedded smart-camera system for automated insect detection and
classification in the field.  A Raspberry Pi 5 (or Luckfox Pico Zero) running
a YOLOv11n model on NCNN continuously monitors a camera feed, detects and
tracks insects, and saves high-quality JPEG crops with full EXIF metadata for
later analysis.

**Key capabilities**

| Capability | Detail |
|---|---|
| Real-time inference | YOLOv11n on NCNN at ≈30 FPS on Pi 5 |
| Multi-object tracking | ByteTracker — assigns stable IDs across frames |
| Crop saving | Best-confidence JPEG per track, async background writer |
| EXIF geo-tagging | GPS, timestamp, trap ID, confidence baked into every JPEG |
| Live MJPEG stream | Always-on 640×480 preview (port 9000) |
| REST API | Full control and monitoring (port 8080) |
| Server-Sent Events | Push notifications for detections and health (port 8081) |
| Capture sessions | Each start/stop cycle stores crops in a timestamped subdirectory |
| Sync API | Session-based transfer protocol for collecting crops in the field |
| SQLite persistence | Detection log and crop manifest with sync state tracking |

---

## 2. Hardware Platforms

### 2.1 Primary — Raspberry Pi 5 + IMX708

| Component | Detail |
|---|---|
| SBC | Raspberry Pi 5 (4 GB) |
| Camera | Sony IMX708 via Camera Module 3 |
| Interface | libcamera (CSI-2) |
| Capture resolution | 2304 × 1296 @ 30 fps (2×2 binned, full FOV) |
| Autofocus | Hardware VCM (continuous / auto / manual) |
| Binary | `yolo_libcamera` |

The Pi 5 is the primary deployment target.  The IMX708's hardware AF enables
sharp crops even when insects land at different distances from the trap.

### 2.2 Secondary — Luckfox Pico Zero + IMX415

| Component | Detail |
|---|---|
| SBC | Luckfox Pico Zero |
| Camera | Sony IMX415 |
| Interface | V4L2 |
| Binary | `yolo_v4l2` |

The Luckfox variant uses the standard Linux V4L2 subsystem.  It shares all
pipeline, persistence, and server code — only the capture layer differs.

---

## 3. Software Architecture

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Network Layer                               │
│   HttpServer :8080    SseServer :8081    MjpegStreamer :9000         │
└──────────────┬───────────────────────────────────────────────────────┘
               │ callbacks / pointers
┌──────────────▼───────────────────────────────────────────────────────┐
│                    Application Core (main_libcamera.cpp)             │
│  g_capturing  currentSessionId  startCaptureSession()               │
└──────┬─────────────┬────────────────────────┬────────────────────────┘
       │             │                        │
┌──────▼──────┐ ┌────▼──────────┐  ┌──────────▼────────┐
│  Inference  │ │  Persistence  │  │   Crop Pipeline   │
│  ncnn::Net  │ │  SqliteWriter │  │  CropSaver        │
│  YoloDecoder│ │  (detections) │  │  ExifWriter       │
│  ByteTracker│ └───────────────┘  │  SyncManager      │
└──────┬──────┘                    └───────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│                  Capture Layer                       │
│    LibcameraCapture   or   V4L2Capture              │
│    NV12 frames + modelInput (letterboxed ncnn::Mat) │
└─────────────────────────────────────────────────────┘
```

### 3.2 Threading Model

| Thread | Owner | Role |
|---|---|---|
| Main thread | `main()` | Runs stats loop; all component startup/shutdown |
| Camera callback | `LibcameraCapture` | Fires per-frame; runs inference, tracking, submit |
| CropSaver worker | `CropSaver` | NV12→RGB→JPEG conversion and disk write |
| SqliteWriter worker | `SqliteWriter` | Batched DB inserts (one transaction per wakeup) |
| HTTP accept | `HttpServer` | Accepts TCP connections; one detached thread per request |
| HTTP handler | `HttpServer` (detached) | Handles one request then exits |
| SSE accept | `SseServer` | Accepts SSE clients; pushes JSON events |
| MJPEG accept | `MjpegStreamer` | Accepts stream clients; pushes JPEG frames |

The inference/tracking path runs entirely on the camera callback thread —
no additional inference threads are needed at ≈30 FPS.

---

## 4. Component Reference

### 4.1 LibcameraCapture (`src/io/libcamera_capture.h`)

Wraps the libcamera C++ API for the Raspberry Pi camera stack.

- Configures the IMX708 sensor at the requested resolution and framerate
- Applies image quality controls: brightness, contrast, saturation, sharpness
- Manages autofocus: Continuous, Auto (one-shot), and Manual modes
- On each frame, delivers a `CaptureFrame` to the registered callback containing:
  - `nv12` — compact NV12 frame buffer (Y plane then UV plane)
  - `modelInput` — pre-processed, letterboxed `ncnn::Mat` at model input size
  - `width`, `height` — original capture dimensions
  - `scale`, `padLeft`, `padTop` — letterbox parameters for bbox rescaling
  - `frameId`, `timestampNs`, `afState`, `lensPosition`

### 4.2 YoloDecoder (`src/pipeline/decoder.h`)

Post-processes raw NCNN output into a list of `Detection` structs.

- Supports `anchor_grid` format (YOLOv11n default NCNN export)
- Applies confidence threshold, NMS, aspect-ratio filtering, and area filtering
- Output coordinates are rescaled to original frame pixels using letterbox params

### 4.3 ByteTracker (`src/pipeline/tracker.h`)

Multi-object tracker using the ByteTrack algorithm.

| Parameter | Default | Role |
|---|---|---|
| `high_threshold` | 0.50 | High-confidence detection gate (pass 1) |
| `low_threshold` | 0.20 | Low-confidence detection gate (pass 2) |
| `iou_threshold` | 0.30 | IoU gate for Hungarian matching |
| `min_hits` | 3 | Frames before a track is "confirmed" |
| `max_missed` | 30 | Frames before a lost track is deleted |

A track is **confirmed** after appearing in `min_hits` consecutive frames.
Only confirmed tracks trigger crop saving, DB logging, and SSE events.

### 4.4 CropSaver (`src/pipeline/crop_saver.h`)

Async JPEG writer.  One best-confidence crop is saved per track per session.

- `submit()` — called on the inference thread; checks confidence gate and
  per-track best record, enqueues a `CropJob` if this is a new best.
- Worker thread — dequeues jobs, extracts the NV12 sub-region, converts to
  RGB via `ncnn::yuv420sp2rgb_nv12`, encodes JPEG via `stb_image_write`.
- `startSession(dir)` — flushes the queue, clears per-track state, switches
  the output directory.  Called at the start of every capture session.
- `SavedCallback` — fired after each successful write; delivers
  `(trackId, classId, cls, conf, path, w, h, timestampUs)` to main.

**Per-track logic**

A new crop is saved only when:
1. `confidence ≥ minConfidence` (default 0.50)
2. `confidence ≥ previousBest + minConfidenceDelta` (default +0.05)
3. `saveCount < maxSavesPerTrack` (default 3)

The file is overwritten on disk; the fixed name per session is
`<class>_<trackId>.jpg`.

### 4.5 ExifWriter (`src/pipeline/exif_writer.h`)

Injects EXIF metadata into a JPEG after it is written.

Tags written: `DateTime`, `DateTimeOriginal`, `GPSLatitude`, `GPSLongitude`,
`GPSAltitude`, `ImageDescription`, `UserComment` (JSON payload), `Make`,
`Software`.

The `UserComment` JSON payload contains:
```json
{
  "trapId": "trap_001",
  "trackId": 42,
  "class": "insect",
  "confidence": 0.87,
  "timestampUs": 1741234567890123
}
```

### 4.6 SqliteWriter (`src/pipeline/persistence.h`)

Background writer for detection events.

- WAL journal mode + `NORMAL` sync pragma for SD-card performance
- Async write queue: all inserts batched in one transaction per wakeup
- Detection rows written **once per track on first confirmation** only
  (stationary insects do not generate new rows on every frame)
- Read API: `queryByTimeRange`, `queryByTrackId`, `queryByClass`,
  `queryRecent`, `queryClassCounts`, `getStats`, `pruneOlderThanDays`

### 4.7 SyncManager (`src/pipeline/sync_manager.h`)

Manages the `crops` table and a lightweight client sync protocol.

- `setCurrentSession(id)` — sets the active capture session; subsequent
  `registerCrop()` calls store crops under `<sessionId>/<file>`.
- `registerCrop()` — inserts one row per JPEG into the `crops` table with
  `synced=0`.
- `openSession()` — generates a random hex token; returns the full manifest
  of `synced=0` crops (re-queryable, stateless).
- `ackFiles()` — marks files `synced=1` once the client confirms download.
- `closeSession()` — deletes all `synced=1` files from disk; marks
  `synced=2` in the DB (audit trail preserved).
- `enforceStorageLimit()` — purges oldest acked crops when free space drops
  below the configured threshold.

### 4.8 HttpServer (`src/server/http_server.h`)

Single-threaded accept loop; one detached thread per request.

Full API documented in [§9](#9-rest-api-reference).

### 4.9 SseServer (`src/server/sse_server.h`)

Server-Sent Events broker on port 8081.

Clients connect with `GET /api/events` (direct) or are redirected there
from port 8080.  Events are pushed asynchronously; slow clients are
dropped to avoid head-of-line blocking.

### 4.10 MjpegStreamer (`src/server/mjpeg_streamer.h`)

Multipart MJPEG stream on port 9000.

The MJPEG stream is **always running** regardless of the capture (detection)
state.  This allows a client to preview the camera while detection is paused.
Default resolution 640×480 at JPEG quality 75.

---

## 5. Data Flow

### 5.1 Per-Frame Pipeline

```
LibcameraCapture callback
  │
  ├─► MjpegStreamer.pushFrame()          ← always runs (preview)
  │
  └─► [if g_capturing]
        │
        ├─► ncnn inference (YoloDecoder.decode)
        │
        ├─► ByteTracker.update()
        │
        ├─► [for each confirmed track at age == min_hits]
        │     ├─► printf + SSE detection event
        │     └─► SqliteWriter.writeBatch()    ← async, first-seen only
        │
        └─► [for each confirmed track]
              └─► CropSaver.submit()           ← async, confidence-gated
```

### 5.2 Crop Save Path

```
CropSaver.submit()
  │  [inference thread — locks m_trackMutex]
  │  checks confidence gate + per-track best record
  │  enqueues CropJob (NV12 copy + metadata)
  │
CropSaver worker thread
  │  dequeues CropJob
  │  extract NV12 sub-region → packed RGB → JPEG
  │  stbi_write_jpg → <session>/<class>_<trackId>.jpg
  │  ExifWriter.inject() → bake in EXIF tags
  │
  └─► SavedCallback (main thread context)
        ├─► SseServer.pushEvent(crop_saved)
        └─► SyncManager.registerCrop()    ← inserts row, synced=0
```

### 5.3 Field Sync Flow

```
1. POST /api/sync/session           → { sessionId, pending }
2. GET  /api/sync/session/{id}      → { crops: [{ file, bytes, trackId, ... }] }
3. GET  /api/crops/{session}/{file} → download JPEG (repeat per file)
4. POST /api/sync/ack               → { sessionId, files: [...] }  mark synced=1
5. DELETE /api/sync/session/{id}    → delete synced=1 files, mark synced=2
```

If the connection drops mid-session the session is silently abandoned.
A new session on the next visit picks up all `synced=0` files.

---

## 6. Capture Sessions

A **capture session** represents a single continuous detection run.
Starting capture creates a new session; stopping capture closes it.

### 6.1 Session Lifecycle

```
Boot / POST /api/capture {"active":true}
  │
  ├─► makeSessionId()           → e.g. "20260314_153042"
  ├─► CropSaver.startSession("crops/20260314_153042/")
  │     flush queue, clear per-track state, create directory
  ├─► SyncManager.setCurrentSession("20260314_153042")
  └─► currentSessionId = "20260314_153042"

POST /api/capture {"active":false}
  │
  ├─► g_capturing = false       → inference/saving halts
  ├─► SyncManager.setCurrentSession("")
  └─► currentSessionId = ""
```

### 6.2 Session ID Format

`YYYYMMDD_HHMMSS` using the local wall clock at the moment capture starts.
Example: `20260314_153042`

### 6.3 Crop File Paths

| Era | Path |
|---|---|
| Before sessions | `crops/insect_42.jpg` |
| With sessions | `crops/20260314_153042/insect_42.jpg` |

The `file` column in the `crops` table stores the session-relative key
(`20260314_153042/insect_42.jpg`).  The `path` column stores the full
filesystem path.

### 6.4 Session Isolation

Because `CropSaver.startSession()` clears the per-track best-confidence map,
track IDs from a previous session never suppress crops in a new session.
Each session captures its own complete set of best-confidence crops.

---

## 7. File & Directory Layout

```
ai-trap-gstr/
├── firmware/
│   ├── src/
│   │   ├── common/
│   │   │   ├── config_loader.h       TOML config parser (header-only)
│   │   │   ├── trap_events.h         SSE JSON event builders
│   │   │   └── stb_image_write.h     JPEG encoder (STB single-header)
│   │   ├── io/
│   │   │   ├── libcamera_capture.h/cpp   Pi 5 camera (libcamera)
│   │   │   └── v4l2_capture.h/cpp        Linux V4L2 fallback
│   │   ├── pipeline/
│   │   │   ├── decoder.h/cpp         YOLO post-processing
│   │   │   ├── tracker.h/cpp         ByteTracker
│   │   │   ├── crop_saver.h/cpp      Async JPEG writer + session mgmt
│   │   │   ├── exif_writer.h/cpp     EXIF metadata injection
│   │   │   ├── persistence.h/cpp     SQLite detection log
│   │   │   └── sync_manager.h/cpp    Crop manifest + field sync
│   │   └── server/
│   │       ├── http_server.h/cpp     REST API (port 8080)
│   │       ├── sse_server.h/cpp      Server-Sent Events (port 8081)
│   │       └── mjpeg_streamer.h/cpp  Live MJPEG stream (port 9000)
│   ├── main_libcamera.cpp            Pi 5 entry point
│   ├── main_v4l2.cpp                 V4L2 entry point
│   └── models/
│       └── yolo11n-320/
│           ├── model.ncnn.param
│           └── model.ncnn.bin
├── api/
│   ├── openapi.yaml                  OpenAPI 3.1 spec
│   └── generate_flutter.sh           Flutter client codegen
├── trap_config.toml                  Runtime configuration template
├── meson.build
├── meson_options.txt
├── DESIGN.md                         This document
└── API.md                            REST API contract

Runtime (working directory):
├── detections.db                     SQLite database
└── crops/
    ├── 20260314_153042/              Session 1
    │   ├── insect_1.jpg
    │   └── insect_7.jpg
    └── 20260314_160531/              Session 2
        ├── insect_1.jpg
        └── insect_3.jpg
```

---

## 8. Database Schema

### 8.1 `detections` table

Populated by `SqliteWriter`.  One row per track on first confirmation.

```sql
CREATE TABLE detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL,
    timestamp_us INTEGER NOT NULL,   -- µs since Unix epoch
    track_id     INTEGER NOT NULL,
    class_id     INTEGER NOT NULL,
    label        TEXT    NOT NULL,
    x1           REAL    NOT NULL,   -- bounding box, original frame pixels
    y1           REAL    NOT NULL,
    x2           REAL    NOT NULL,
    y2           REAL    NOT NULL,
    confidence   REAL    NOT NULL,
    frame_w      INTEGER NOT NULL,
    frame_h      INTEGER NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Indexes: `timestamp_us`, `(track_id, timestamp_us)`, `(label, timestamp_us)`,
`frame_id`.

### 8.2 `crops` table

Populated by `SyncManager.registerCrop()`.  One row per JPEG file.

```sql
CREATE TABLE crops (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file            TEXT    NOT NULL UNIQUE, -- e.g. "20260314_153042/insect_42.jpg"
    path            TEXT    NOT NULL,         -- full filesystem path
    track_id        INTEGER NOT NULL,
    class_id        INTEGER NOT NULL DEFAULT 0,
    label           TEXT    NOT NULL DEFAULT '',
    confidence      REAL    NOT NULL DEFAULT 0,
    timestamp_us    INTEGER NOT NULL DEFAULT 0,
    bytes           INTEGER NOT NULL DEFAULT 0,
    synced          INTEGER NOT NULL DEFAULT 0,  -- 0=new  1=acked  2=deleted
    capture_session TEXT    NOT NULL DEFAULT '',  -- e.g. "20260314_153042"
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Indexes: `(synced, created_at)`, `track_id`, `capture_session`.

**`synced` state machine**

```
0 (new) ──► 1 (acked) ──► 2 (deleted)
                │
                └──► enforceStorageLimit() may also transition 0 → 2
```

### 8.3 Performance Notes

- WAL journal mode: readers and writers do not block each other
- `PRAGMA synchronous=NORMAL`: safe on Pi SD/SSD, much faster than FULL
- 10 MB page cache; 4 KB pages for sequential write throughput
- Detection inserts are batched in a single transaction per wakeup

---

## 9. REST API Reference

Base URL: `http://192.168.5.1:8080`
All responses include `Access-Control-Allow-Origin: *`.

### 9.1 Trap Identity

```
GET /api/trap
→ { "id": "trap_001", "location": "" }
```

### 9.2 System Status

```
GET /api/status
→ {
    "id": "trap_001",
    "location": "",
    "capturing": true,
    "sessionId": "20260314_153042",   // null if not capturing
    "uptime_s": 3601,
    "fps": 28.4,
    "detections": 247,
    "tracks": 31,
    "db_mb": 1.4,
    "sse_clients": 2
  }
```

### 9.3 Capture Control

```
GET  /api/capture
→ { "active": true, "sessionId": "20260314_153042" }

POST /api/capture
Body: { "active": true }          ← starts a new session
Body: { "active": false }         ← closes the current session
→ { "active": true, "sessionId": "20260314_153042" }
```

Starting capture while already active starts a **new** session.

### 9.4 Crops

```
GET /api/crops
→ [
    {
      "file": "20260314_153042/insect_42.jpg",
      "bytes": 18432,
      "trackId": 42,
      "conf": 0.8721,
      "timestampUs": 1741234567890123,
      "label": "insect",
      "session": "20260314_153042"
    },
    ...
  ]

GET /api/crops/{sessionId}/{filename}    ← download JPEG
GET /api/crops/{filename}                ← legacy flat path (backward compat)
```

### 9.5 Configuration

```
POST /api/config/location
Body: { "lat": 51.5074, "lon": -0.1278 }
→ { "status": "ok" }

POST /api/config/threshold
Body: { "value": 0.45 }
→ { "status": "ok" }

POST /api/af/trigger               ← one-shot autofocus
→ { "status": "ok" }
```

### 9.6 Field Sync Protocol

```
POST   /api/sync/session
→ { "sessionId": "a3f9c12b4e7d...", "pending": 11 }

GET    /api/sync/session/{id}
→ {
    "sessionId": "a3f9c12b4e7d...",
    "pending": 11,
    "crops": [{ "file", "bytes", "trackId", "label", "conf", "timestampUs" }, ...]
  }

POST   /api/sync/ack
Body: { "sessionId": "a3f9c12b4e7d...", "files": ["20260314_153042/insect_42.jpg"] }
→ { "acked": 1 }

DELETE /api/sync/session/{id}
→ { "deleted": 11, "bytesFreed": 204800, "notFound": 0 }
```

> **Note:** The sync session ID (random hex token) is separate from the
> capture session ID (timestamp string).  The sync session is a lightweight
> download-coordination mechanism; the capture session namespaces the files.

### 9.7 Events Redirect

```
GET /api/events  →  307 redirect to http://192.168.5.1:8081/api/events
```

---

## 10. Server-Sent Events

Connect to `http://192.168.5.1:8081/api/events` (SSE stream).

### `detection`
Fired once when a track is first confirmed (`age == min_hits`).
```json
{
  "type": "detection",
  "trackId": 42,
  "class": "insect",
  "conf": 0.87,
  "bbox": [120.0, 80.0, 260.0, 190.0],
  "frameId": 1234,
  "ts": 1741234567890
}
```

### `crop_saved`
Fired after each JPEG is written to disk.
```json
{
  "type": "crop_saved",
  "trackId": 42,
  "class": "insect",
  "conf": 0.87,
  "file": "insect_42.jpg",
  "w": 140, "h": 110,
  "ts": 1741234567890
}
```

### `capture`
Fired when detection starts or stops.
```json
{ "type": "capture", "active": true, "ts": 1741234567890 }
```

### `stats`
Pushed every 30 seconds.
```json
{
  "type": "stats",
  "today": 247,
  "uptime_s": 22440,
  "fps": 28.4,
  "tracks": 31,
  "db_mb": 1.4,
  "ts": 1741234567890
}
```

### `health`
Pushed every 30 seconds alongside `stats`.
```json
{
  "type": "health",
  "temp_c": 42.1,
  "af_state": 2,
  "lens_pos": 2.4,
  "ts": 1741234567890
}
```

---

## 11. Configuration Reference

All settings are read from `trap_config.toml` at startup.
GPS location and confidence threshold can be updated at runtime via the API.

### `[trap]`
| Key | Default | Description |
|---|---|---|
| `id` | `"trap_001"` | Trap identifier — embedded in EXIF and API responses |
| `location` | `""` | Human-readable location string |
| `lat` / `lon` | — | GPS decimal degrees (optional; enables EXIF geo-tagging) |
| `alt_m` | `0.0` | Altitude in metres above sea level |

### `[model]`
| Key | Default | Description |
|---|---|---|
| `param` | `"../firmware/models/yolo11n-320/model.ncnn.param"` | NCNN param file |
| `bin` | `"../firmware/models/yolo11n-320/model.ncnn.bin"` | NCNN bin file |
| `width` / `height` | `320` | Model input size (must match export) |
| `num_classes` | `1` | Number of output classes |
| `format` | `"anchor_grid"` | NCNN output format |
| `pre_applied_sigmoid` | `true` | Skip sigmoid in decoder if already applied |

### `[detection]`
| Key | Default | Description |
|---|---|---|
| `conf_threshold` | `0.45` | Minimum score to keep a detection |
| `nms_threshold` | `0.45` | IoU threshold for NMS |
| `min_box_width` | `20` | Reject boxes narrower than this (pixels) |
| `min_box_height` | `20` | Reject boxes shorter than this (pixels) |
| `max_aspect_ratio` | `5.0` | Reject highly elongated boxes |
| `max_box_area_ratio` | `0.15` | Reject boxes covering >15% of frame area |

### `[tracker]`
| Key | Default | Description |
|---|---|---|
| `high_threshold` | `0.50` | Confident detections (pass-1 matching) |
| `low_threshold` | `0.20` | Tentative detections (pass-2 matching) |
| `iou_threshold` | `0.30` | IoU gate for Hungarian algorithm |
| `min_hits` | `3` | Frames to confirm a track |
| `max_missed` | `30` | Frames before deleting a lost track |

### `[camera]`
| Key | Default | Description |
|---|---|---|
| `capture_width` | `2304` | Sensor capture width |
| `capture_height` | `1296` | Sensor capture height |
| `framerate` | `30` | Target frame rate |
| `brightness` | `0.0` | −1.0 to +1.0 |
| `contrast` | `1.0` | 0.0 to 4.0 |
| `saturation` | `1.0` | 0.0 to 4.0 |
| `sharpness` | `1.0` | 0.0 to 16.0 |

### `[autofocus]`
| Key | Default | Description |
|---|---|---|
| `mode` | `2` | 0=Manual, 1=Auto, 2=Continuous |
| `range` | `0` | 0=Normal, 1=Macro, 2=Full |
| `speed` | `0` | 0=Normal, 1=Fast |
| `lens_position` | `2.0` | Dioptres (Manual mode only) |

### `[crops]`
| Key | Default | Description |
|---|---|---|
| `output_dir` | `"crops"` | Base directory; sessions created as subdirs |
| `jpeg_quality` | `90` | JPEG quality 1–100 |
| `min_confidence` | `0.50` | Ignore tracks below this score |
| `min_confidence_delta` | `0.05` | Beat previous best by this to re-save |
| `max_saves_per_track` | `3` | Cap per-track saves |
| `max_queue_depth` | `16` | Async queue depth (excess submits dropped) |

### `[stream]`
| Key | Default | Description |
|---|---|---|
| `port` | `9000` | MJPEG stream port |
| `width` | `640` | Stream width |
| `height` | `480` | Stream height |
| `jpeg_quality` | `75` | Stream JPEG quality |

### `[sse]`
| Key | Default | Description |
|---|---|---|
| `port` | `8081` | SSE server port |
| `max_clients` | `8` | Maximum simultaneous SSE clients |
| `max_queue_depth` | `64` | Per-client event queue depth |

### `[api]`
| Key | Default | Description |
|---|---|---|
| `port` | `8080` | HTTP REST API port |

### `[database]`
| Key | Default | Description |
|---|---|---|
| `path` | `"detections.db"` | SQLite database file path |

---

## 12. Build System

### 12.1 Requirements

| Dependency | Required by |
|---|---|
| `ncnn` | Inference + NV12→RGB conversion |
| `sqlite3` | Detection log and crop manifest |
| `pthreads` | All worker threads |
| `libcamera` | `yolo_libcamera` target only |
| `openmp` | Optional; parallel NMS (ncnn) |

### 12.2 Build Targets

| Option | Target binary | Platform |
|---|---|---|
| `-Dtarget=libcamera` (default) | `yolo_libcamera` | Pi 5 + IMX708 |
| `-Dtarget=v4l2` | `yolo_v4l2` | Luckfox + IMX415 |
| `-Dtarget=all` | both | (requires libcamera) |

### 12.3 Build Instructions

```bash
# Pi 5 (libcamera)
meson setup buildDir -Dtarget=libcamera
ninja -C buildDir

# Luckfox / V4L2
meson setup buildDir -Dtarget=v4l2
ninja -C buildDir

# Run (from the directory containing trap_config.toml and models/)
./buildDir/yolo_libcamera trap_config.toml
```

### 12.4 Compiler Options

- C++ standard: C++17
- Optimisation: `-O2`
- No additional warning flags beyond Meson defaults

### 12.5 Shared Source Files

The following files are compiled into **both** targets:

```
firmware/src/pipeline/decoder.cpp
firmware/src/pipeline/tracker.cpp
firmware/src/pipeline/persistence.cpp
firmware/src/pipeline/crop_saver.cpp
firmware/src/pipeline/exif_writer.cpp
firmware/src/pipeline/sync_manager.cpp
firmware/src/server/mjpeg_streamer.cpp
firmware/src/server/sse_server.cpp
firmware/src/server/http_server.cpp
firmware/src/common/stb_image_write_impl.cpp
```

---

*End of document.*
