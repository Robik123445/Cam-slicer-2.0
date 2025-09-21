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
  sender/
    __init__.py
    service.py
requirements.txt
log.txt
README.md
```

The code is intentionally modular so it can be dropped into existing CAM or
shop-floor automation stacks without refactoring.
