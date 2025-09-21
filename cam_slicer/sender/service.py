"""Thread-safe GRBL sender service implementation."""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Deque, Dict, List, Literal, Optional, Union

import serial
from pydantic import BaseModel
from serial import SerialException
from serial.tools import list_ports


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _LOG_PATH = Path(__file__).resolve().parents[2] / "log.txt"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FILE_HANDLER = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        _FILE_HANDLER.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        _LOGGER.addHandler(_FILE_HANDLER)
    except OSError:
        _LOGGER.addHandler(logging.NullHandler())
else:
    _LOGGER.addHandler(logging.NullHandler())


_EVENT_CALLBACK = Callable[[Dict[str, object]], None]


class SenderError(Exception):
    """Base exception for sender service errors."""


class SenderStateError(SenderError):
    """Raised when an operation is invalid for the current state."""


class SenderJobError(SenderError):
    """Raised when a queued job fails during execution."""


class LineJob(BaseModel):
    """Single G-code line job."""

    job_id: str
    type: Literal["LINE"] = "LINE"
    gcode: str


class FileStreamJob(BaseModel):
    """Streaming job that feeds a G-code file line by line."""

    job_id: str
    type: Literal["FILE_STREAM"] = "FILE_STREAM"
    file_path: str
    start_line: int = 0


class JogJob(BaseModel):
    """Jogging job based on GRBL real-time jogging commands."""

    job_id: str
    type: Literal["JOG"] = "JOG"
    mode: Literal["rel", "abs"]
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    feed: float = 1500.0


class ProbePointJob(BaseModel):
    """Probe job executing a standard four-step probing sequence."""

    job_id: str
    type: Literal["PROBE_POINT"] = "PROBE_POINT"
    x: float
    y: float
    z_clear: float
    z_probe: float
    feed_probe: float


Job = Union[LineJob, FileStreamJob, JogJob, ProbePointJob]


class SenderService:
    """Thread-safe GRBL sender service coordinating TX/RX workers."""

    _STATUS_RE = re.compile(r"^<(?P<state>[A-Za-z]+)(?:\|(?P<extra>.*))?>$")
    _M_POS_RE = re.compile(r"MPos:(?P<values>[-0-9.,]+)")
    _W_POS_RE = re.compile(r"WPos:(?P<values>[-0-9.,]+)")

    _REALTIME_COMMANDS = {
        "hold": 0x21,
        "start": 0x24,
        "reset": 0x18,
        "jog_cancel": 0x85,
    }

    def __init__(self) -> None:
        """Initialize the sender service with worker and RX threads."""

        self._serial_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._ack_condition = threading.Condition()
        self._job_queue: "Queue[Job]" = Queue()
        self._rx_lines: Deque[str] = deque(maxlen=200)
        self._event_sink: Optional[_EVENT_CALLBACK] = None

        self._serial: Optional[serial.Serial] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_stop = threading.Event()

        self._awaiting_ack = False
        self._ack_line: Optional[str] = None

        self._service_state = "DISCONNECTED"
        self._machine_state: Optional[str] = None
        self._last_status_line: Optional[str] = None
        self._last_rx_line: str = ""
        self._mpos: Optional[List[float]] = None
        self._wpos: Optional[List[float]] = None
        self._port: Optional[str] = None
        self._baud: Optional[int] = None
        self._serial_error = False

        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="grbl-worker", daemon=True
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_ports(self) -> List[str]:
        """Return a list of available serial ports."""

        return [port.device for port in list_ports.comports()]

    def open(self, port: str, baud: int = 115200) -> tuple[bool, str]:
        """Open a serial connection to the GRBL controller."""

        with self._serial_lock:
            if self._serial and self._serial.is_open:
                return False, "Serial port already open"

            self._serial_error = False
            try:
                self._serial = serial.Serial(
                    port=port,
                    baudrate=baud,
                    timeout=0.1,
                    write_timeout=0.1,
                )
            except SerialException as exc:
                _LOGGER.error("Failed to open serial port %s: %s", port, exc)
                self._update_service_state("ERROR")
                return False, str(exc)

            self._port = port
            self._baud = baud
            try:
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
            except SerialException as exc:
                _LOGGER.warning("Failed to reset buffers: %s", exc)

        self._rx_stop.clear()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="grbl-rx", daemon=True
        )
        self._rx_thread.start()
        self._update_service_state("IDLE")
        return True, "Connected"

    def close(self) -> None:
        """Close the serial connection and stop RX thread."""

        self._rx_stop.set()
        rx_thread = self._rx_thread
        if rx_thread and rx_thread.is_alive():
            rx_thread.join(timeout=1.0)
        self._rx_thread = None

        with self._serial_lock:
            if self._serial:
                try:
                    self._serial.close()
                except SerialException as exc:
                    _LOGGER.warning("Error closing serial port: %s", exc)
                self._serial = None

        with self._ack_condition:
            self._awaiting_ack = False
            self._ack_line = None
            self._ack_condition.notify_all()

        with self._state_lock:
            self._machine_state = None
            self._last_status_line = None
            self._mpos = None
            self._wpos = None
            self._port = None
            self._baud = None
            self._serial_error = False

        self._update_service_state("DISCONNECTED")

    def status(self) -> dict:
        """Return the latest machine and connection status."""

        with self._state_lock:
            machine_state = self._machine_state
            last_line = self._last_rx_line
            mpos = list(self._mpos) if self._mpos else None
            wpos = list(self._wpos) if self._wpos else None
            port = self._port or ""
            service_state = self._service_state

        state_value = machine_state or service_state
        return {
            "state": state_value,
            "last": last_line,
            "mpos": mpos,
            "wpos": wpos,
            "port": port,
        }

    def enqueue_line(self, gcode: str) -> str:
        """Queue a single G-code line for execution."""

        if not gcode or not gcode.strip():
            raise ValueError("G-code line must not be empty")
        job_id = self._new_job_id()
        job = LineJob(job_id=job_id, gcode=gcode.strip())
        self._queue_job(job)
        return job_id

    def enqueue_file(self, file_path: str, start_line: int = 0) -> str:
        """Queue a file streaming job starting at the given line."""

        if start_line < 0:
            raise ValueError("start_line must be non-negative")
        job_id = self._new_job_id()
        job = FileStreamJob(job_id=job_id, file_path=file_path, start_line=start_line)
        self._queue_job(job)
        return job_id

    def enqueue_jog(
        self, mode: str, dx: float, dy: float, dz: float, feed: float
    ) -> str:
        """Queue a jog command in either relative or absolute mode."""

        if mode not in {"rel", "abs"}:
            raise ValueError("Jog mode must be 'rel' or 'abs'")
        if feed <= 0:
            raise ValueError("Feed must be positive")
        job_id = self._new_job_id()
        job = JogJob(job_id=job_id, mode=mode, dx=dx, dy=dy, dz=dz, feed=feed)
        self._queue_job(job)
        return job_id

    def enqueue_probe_point(
        self, x: float, y: float, z_clear: float, z_probe: float, feed_probe: float
    ) -> str:
        """Queue a probing sequence to measure a specific point."""

        if feed_probe <= 0:
            raise ValueError("Probe feed must be positive")
        job_id = self._new_job_id()
        job = ProbePointJob(
            job_id=job_id,
            x=x,
            y=y,
            z_clear=z_clear,
            z_probe=z_probe,
            feed_probe=feed_probe,
        )
        self._queue_job(job)
        return job_id

    def hold(self) -> None:
        """Pause the machine using the GRBL hold command."""

        self._send_realtime("hold")

    def start(self) -> None:
        """Resume the machine after a hold."""

        self._send_realtime("start")

    def reset(self) -> None:
        """Reset the controller by issuing the real-time reset command."""

        self._send_realtime("reset")

    def jog_cancel(self) -> None:
        """Abort an active jog motion."""

        self._send_realtime("jog_cancel")

    def set_event_sink(self, callback: Callable[[Dict[str, object]], None]) -> None:
        """Register a callback for RX, state, and job events."""

        self._event_sink = callback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _queue_job(self, job: Job) -> None:
        """Queue a validated job for the worker thread."""

        self._ensure_ready()
        self._job_queue.put(job)
        _LOGGER.debug("Queued job %s (%s)", job.job_id, job.type)

    def _ensure_ready(self) -> None:
        """Ensure that the serial connection is ready for new jobs."""

        if self._serial_error:
            raise SenderStateError("Serial port is in error state")
        with self._serial_lock:
            if not self._serial or not self._serial.is_open:
                raise SenderStateError("Serial port is not open")

    def _worker_loop(self) -> None:
        """Continuously process queued jobs in sequence."""

        while True:
            try:
                job = self._job_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._process_job(job)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.exception("Unhandled error processing job %s: %s", job.job_id, exc)
            finally:
                self._job_queue.task_done()

    def _process_job(self, job: Job) -> None:
        """Execute a job based on its type."""

        try:
            self._ensure_ready()
        except SenderStateError as exc:
            _LOGGER.error("Skipping job %s: %s", job.job_id, exc)
            self._emit_job_event(job.job_id, "error", {"message": str(exc), "kind": job.type})
            return

        _LOGGER.info("Starting job %s (%s)", job.job_id, job.type)
        self._emit_job_event(job.job_id, "started", {"kind": job.type})

        if isinstance(job, LineJob):
            self._execute_line_job(job)
        elif isinstance(job, FileStreamJob):
            self._execute_file_job(job)
        elif isinstance(job, JogJob):
            self._execute_jog_job(job)
        elif isinstance(job, ProbePointJob):
            self._execute_probe_job(job)
        else:  # pragma: no cover - defensive branch
            _LOGGER.error("Unknown job type: %s", job)

    def _execute_line_job(self, job: LineJob) -> None:
        """Send a single line job to the controller."""

        try:
            self._update_service_state("RUNNING")
            ack = self._send_line(job.gcode)
            if ack and ack.startswith("error"):
                raise SenderJobError(ack)
            self._emit_job_event(job.job_id, "finished", {"kind": job.type})
        except SenderJobError as exc:
            _LOGGER.error("Line job %s failed: %s", job.job_id, exc)
            self._emit_job_event(job.job_id, "error", {"message": str(exc), "kind": job.type})
            self._update_service_state("ERROR")
        finally:
            if self._service_state != "ERROR":
                self._update_service_state("IDLE")

    def _execute_file_job(self, job: FileStreamJob) -> None:
        """Stream a G-code file with back-pressure handling."""

        try:
            path = Path(job.file_path).expanduser()
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _LOGGER.error("Cannot read file %s: %s", job.file_path, exc)
            self._emit_job_event(
                job.job_id,
                "error",
                {"message": f"File read failed: {exc}", "kind": job.type},
            )
            self._update_service_state("ERROR")
            return

        if job.start_line >= len(lines):
            self._emit_job_event(
                job.job_id,
                "error",
                {"message": "start_line beyond file length", "kind": job.type},
            )
            return

        selected = lines[job.start_line :]
        total = len(selected) if selected else 1
        self._update_service_state("RUNNING")
        self._emit_job_event(
            job.job_id,
            "progress",
            {"progress": 0.0, "kind": job.type},
        )
        try:
            for index, raw_line in enumerate(selected, start=1):
                line = raw_line.strip()
                if not line:
                    progress = index / total
                    self._emit_job_event(
                        job.job_id, "progress", {"progress": progress, "kind": job.type}
                    )
                    continue
                try:
                    ack = self._send_line(line)
                except SenderJobError as exc:
                    _LOGGER.error("File job %s aborted: %s", job.job_id, exc)
                    self._emit_job_event(
                        job.job_id,
                        "error",
                        {"message": str(exc), "kind": job.type},
                    )
                    self._update_service_state("ERROR")
                    return
                if ack and ack.startswith("error"):
                    _LOGGER.error("Controller returned error for line: %s", ack)
                    self._emit_job_event(
                        job.job_id,
                        "error",
                        {"message": ack, "kind": job.type},
                    )
                    self._update_service_state("ERROR")
                    return
                progress = index / total
                self._emit_job_event(
                    job.job_id, "progress", {"progress": progress, "kind": job.type}
                )
            self._emit_job_event(job.job_id, "finished", {"kind": job.type})
        finally:
            if self._service_state != "ERROR":
                self._update_service_state("IDLE")

    def _execute_jog_job(self, job: JogJob) -> None:
        """Execute a jog job and keep service responsive."""

        command = ["$J="]
        if job.mode == "rel":
            command.append("G91")
        else:
            command.append("G90")

        axis_parts = []
        for axis, value in (("X", job.dx), ("Y", job.dy), ("Z", job.dz)):
            if job.mode == "abs" or abs(value) > 1e-9:
                axis_parts.append(f"{axis}{self._format_float(value)}")
        axis_parts.append(f"F{self._format_float(job.feed)}")
        command.extend(axis_parts)
        jog_command = " ".join(command)

        try:
            self._update_service_state("JOGGING")
            ack = self._send_line(jog_command)
            if ack and ack.startswith("error"):
                raise SenderJobError(ack)
            self._emit_job_event(job.job_id, "finished", {"kind": job.type})
        except SenderJobError as exc:
            _LOGGER.error("Jog job %s failed: %s", job.job_id, exc)
            self._emit_job_event(job.job_id, "error", {"message": str(exc), "kind": job.type})
            self._update_service_state("ERROR")
        finally:
            if self._service_state != "ERROR":
                self._update_service_state("IDLE")

    def _execute_probe_job(self, job: ProbePointJob) -> None:
        """Execute a probing routine at the specified coordinates."""

        sequence = [
            f"G0 Z{self._format_float(job.z_clear)}",
            f"G0 X{self._format_float(job.x)} Y{self._format_float(job.y)}",
            f"G38.2 Z{self._format_float(job.z_probe)} F{self._format_float(job.feed_probe)}",
            f"G0 Z{self._format_float(job.z_clear)}",
        ]
        self._update_service_state("PROBING")
        try:
            for line in sequence:
                ack = self._send_line(line)
                if ack and ack.startswith("error"):
                    raise SenderJobError(ack)
            self._emit_job_event(job.job_id, "finished", {"kind": job.type})
        except SenderJobError as exc:
            _LOGGER.error("Probe job %s failed: %s", job.job_id, exc)
            self._emit_job_event(job.job_id, "error", {"message": str(exc), "kind": job.type})
            self._update_service_state("ERROR")
        finally:
            if self._service_state != "ERROR":
                self._update_service_state("IDLE")

    def _send_line(self, line: str) -> Optional[str]:
        """Send a line and wait for acknowledgement with retry."""

        clean_line = line.strip()
        if not clean_line:
            return None

        ack: Optional[str] = None
        for attempt in range(2):
            ack = self._transmit_and_wait(clean_line, timeout=2.0)
            if ack:
                break
            if attempt == 0:
                _LOGGER.warning("No ACK for line '%s', retrying", clean_line)
        if not ack:
            _LOGGER.error("No ACK received for line '%s' after retries", clean_line)
        return ack

    def _transmit_and_wait(self, line: str, timeout: float) -> Optional[str]:
        """Write a line to serial and wait for an ACK."""

        with self._ack_condition:
            self._awaiting_ack = True
            self._ack_line = None

        data = (line + "\n").encode("ascii", errors="ignore")
        try:
            with self._serial_lock:
                if not self._serial:
                    raise SenderStateError("Serial port not available")
                self._serial.write(data)
                self._serial.flush()
        except (SerialException, SenderStateError) as exc:
            _LOGGER.error("Failed to write line '%s': %s", line, exc)
            with self._ack_condition:
                self._awaiting_ack = False
                self._ack_condition.notify_all()
            if isinstance(exc, SerialException):
                self._handle_serial_failure(exc)
                raise SenderJobError(str(exc)) from exc
            raise SenderJobError(str(exc)) from exc

        deadline = time.monotonic() + timeout
        with self._ack_condition:
            while self._awaiting_ack and time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ack_condition.wait(timeout=remaining)
            ack = self._ack_line
            self._awaiting_ack = False
            self._ack_line = None
        return ack

    def _send_realtime(self, command: str) -> None:
        """Send a real-time command byte to the controller."""

        if command not in self._REALTIME_COMMANDS:
            raise ValueError(f"Unsupported real-time command: {command}")
        code = self._REALTIME_COMMANDS[command]
        data = bytes([code])
        with self._serial_lock:
            if not self._serial or not self._serial.is_open:
                raise SenderStateError("Serial port is not open")
            try:
                self._serial.write(data)
                self._serial.flush()
            except SerialException as exc:
                _LOGGER.error("Real-time command %s failed: %s", command, exc)
                self._handle_serial_failure(exc)
                raise SenderError(str(exc)) from exc

    def _rx_loop(self) -> None:
        """Continuously read from the serial port and handle responses."""

        while not self._rx_stop.is_set():
            with self._serial_lock:
                serial_ref = self._serial
            if not serial_ref or not serial_ref.is_open:
                time.sleep(0.05)
                continue
            try:
                raw = serial_ref.readline()
            except SerialException as exc:
                _LOGGER.error("Serial read failed: %s", exc)
                self._handle_serial_failure(exc)
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            self._rx_lines.append(line)
            self._last_rx_line = line
            self._emit_event({"type": "rx", "data": line})
            if line.startswith("<") and line.endswith(">"):
                self._handle_status_line(line)
            elif line == "ok" or line.startswith("error"):
                self._register_ack(line)

    def _register_ack(self, line: str) -> None:
        """Register an acknowledgement and notify waiting threads."""

        with self._ack_condition:
            self._ack_line = line
            self._awaiting_ack = False
            self._ack_condition.notify_all()

    def _handle_status_line(self, line: str) -> None:
        """Parse machine status and update cached values."""

        match = self._STATUS_RE.match(line)
        if not match:
            return
        state = match.group("state")
        extra = match.group("extra") or ""

        mpos_match = self._M_POS_RE.search(extra)
        wpos_match = self._W_POS_RE.search(extra)
        mpos = self._parse_position(mpos_match.group("values")) if mpos_match else None
        wpos = self._parse_position(wpos_match.group("values")) if wpos_match else None

        with self._state_lock:
            self._machine_state = state
            self._last_status_line = line
            if mpos is not None:
                self._mpos = mpos
            if wpos is not None:
                self._wpos = wpos

        self._emit_event(
            {
                "type": "state",
                "data": {
                    "service": self._service_state,
                    "machine": state,
                    "mpos": self._mpos,
                    "wpos": self._wpos,
                },
            }
        )

    def _parse_position(self, data: str) -> List[float]:
        """Parse a comma separated position string."""

        return [float(value) for value in data.split(",") if value]

    def _emit_event(self, payload: Dict[str, object]) -> None:
        """Emit an event through the registered callback."""

        callback = self._event_sink
        if not callback:
            return
        try:
            callback(payload)
        except Exception as exc:  # pragma: no cover - best effort logging
            _LOGGER.error("Event sink raised an exception: %s", exc)

    def _emit_job_event(self, job_id: str, event_type: str, data: Dict[str, object]) -> None:
        """Emit a structured job event."""

        payload = {"type": "job", "data": {"id": job_id, "event": event_type}}
        payload["data"].update(data)
        self._emit_event(payload)

    def _update_service_state(self, new_state: str) -> None:
        """Update internal service state and notify listeners."""

        with self._state_lock:
            if self._service_state == new_state:
                return
            self._service_state = new_state
        self._emit_event(
            {
                "type": "state",
                "data": {
                    "service": new_state,
                    "machine": self._machine_state,
                    "mpos": self._mpos,
                    "wpos": self._wpos,
                },
            }
        )

    def _handle_serial_failure(self, exc: Exception) -> None:
        """Handle serial failure scenarios by switching to error state."""

        _LOGGER.error("Serial failure detected: %s", exc)
        with self._serial_lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except SerialException:
                    pass
                self._serial = None
        with self._ack_condition:
            self._awaiting_ack = False
            self._ack_condition.notify_all()
        with self._state_lock:
            self._serial_error = True
        self._update_service_state("ERROR")

    def _format_float(self, value: float) -> str:
        """Format a float value for G-code commands."""

        return ("{:.5f}".format(value)).rstrip("0").rstrip(".") or "0"

    def _new_job_id(self) -> str:
        """Generate a new unique job identifier."""

        return uuid.uuid4().hex


__all__ = ["SenderService", "SenderError", "SenderStateError", "SenderJobError"]
