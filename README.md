# Cam-slicer-2.0

Thread-safe GRBL sender service providing CNC/laser/3D device integration.

## Overview

This repository contains a production-ready `SenderService` that streams G-code
commands to GRBL-based controllers with strict back-pressure handling,
real-time jog support, and structured event callbacks. The implementation is
fully thread-safe, designed for easy embedding into higher level CAM
orchestration layers.

## Features

- Serial connection lifecycle management with automatic error handling.
- Dedicated RX and worker threads to keep command flow deterministic.
- Queue-based jobs for:
  - Single line commands.
  - File streaming with progress feedback.
  - Relative/absolute jogging with cancel support.
  - Four-step probing macros.
- Real-time control helpers (`hold`, `start`, `reset`, `jog_cancel`).
- Rich status tracking (last 200 RX lines, `<...>` state parsing, position
  extraction).
- Event sink API for RX, state, and job notifications.
- File logging out-of-the-box (`log.txt`).
- Raster probe planner producing structured height maps with bilinear
  interpolation utilities.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from cam_slicer import SenderService

service = SenderService()
service.set_event_sink(lambda evt: print(evt))  # replace with logging sink
success, message = service.open("/dev/ttyUSB0", baud=115200)
if not success:
    raise RuntimeError(message)

service.enqueue_line("G0 X10 Y10")
service.enqueue_file("/path/to/file.gcode")
service.enqueue_jog("rel", dx=5, dy=0, dz=0, feed=1500)
service.enqueue_probe_point(x=10, y=10, z_clear=5, z_probe=-2, feed_probe=150)

# Use real-time helpers as required
service.hold()
service.start()
service.jog_cancel()

print(service.status())
service.close()
```

The event sink receives dictionaries in the form:

- `{"type": "rx", "data": "<Idle|...>"}` for each raw line.
- `{"type": "state", "data": {"service": "RUNNING", "machine": "Idle", ...}}`
  whenever service or machine status changes.
- `{"type": "job", "data": {"id": "...", "event": "progress", "progress": 0.5}}`
  for job lifecycle notifications.

## Probe planning utilities

The `cam_slicer.probe.planner` module converts raster probing runs into
structured `HeightMap` models. Helpers such as `roi_to_grid` and
`probe_grid` build deterministic boustrophedon plans, while `fit_plane` and
`bilinear_interp` provide smooth surface models for auto-leveling or stock
compensation.

## REST API & Orchestrator

The project ships with a pre-wired FastAPI application (`cam_slicer.api.app`) that
exposes the sender, vision, probe, and high-level intent layers. The
`Orchestrator` class builds on top of the sender to run simple vision-guided
workflows while keeping calibration data in a global, lock-protected `AppState`.

### Run the API locally

```bash
uvicorn cam_slicer.api.app:app --reload
```

Swagger/OpenAPI documentation is available at `http://localhost:8000/docs` once
the server is running.

### Available endpoints

- `GET /health` – basic readiness probe.
- `GET /sender/ports`, `POST /sender/open`, `GET /sender/status`, and
  queue/control helpers under `/sender/*` for G-code streaming.
- `POST /vision/calibrate`, `/vision/relock`, `/vision/detect` to manage
  calibration matrices and synthetic detections.
- `POST /probe/grid` to enqueue raster probing jobs.
- Probe planner helpers in `cam_slicer.probe.planner` convert probing results
  into interpolated height maps for downstream compensation workflows.
- `/intent/*` routes mapping UI intents (guide, measure, find edges) to
  orchestrator workflows.

### WebSocket streaming

Connect to `ws://localhost:8000/ws/sender` to receive the same sender events as
the event sink. Every queued line triggers at least the latest RX echo through
the websocket so UIs stay in sync.

## Logging

All internal logs are written to `log.txt` by default. Adjust the Python
logging configuration if you need to route messages elsewhere.

## Testing

Run the bundled unit tests (a fake serial port is used, no hardware required):

```bash
python -m unittest discover -s tests
```

## Project structure

```
cam_slicer/
  __init__.py
  probe/
    __init__.py
    planner.py
  sender/
    __init__.py
    service.py
requirements.txt
log.txt
README.md
```

The code is intentionally modular so it can be dropped into existing CAM or
shop-floor automation stacks without refactoring.
