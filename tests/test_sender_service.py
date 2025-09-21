"""Basic tests for the SenderService implementation."""

from __future__ import annotations

import queue
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from cam_slicer.sender.service import SenderService, SenderStateError


class FakeSerial:
    """In-memory serial port mock to validate sender behaviour."""

    instances: List["FakeSerial"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.get("timeout", 0.1)
        self.write_timeout = kwargs.get("write_timeout", 0.1)
        self.is_open = True
        self.written: List[str] = []
        self._read_queue: "queue.Queue[bytes]" = queue.Queue()
        # Provide initial status update to emulate GRBL start-up feedback.
        self._read_queue.put(
            b"<Idle|MPos:0.000,0.000,0.000|WPos:0.000,0.000,0.000>\n"
        )
        FakeSerial.instances.append(self)

    # Serial API -----------------------------------------------------------------
    def write(self, data: bytes) -> int:
        try:
            decoded = data.decode("ascii").strip()
        except UnicodeDecodeError:
            decoded = repr(data)
        self.written.append(decoded)
        if data.endswith(b"\n"):
            self._read_queue.put(b"ok\n")
        return len(data)

    def readline(self) -> bytes:
        try:
            return self._read_queue.get(timeout=self.timeout)
        except queue.Empty:
            return b""

    def flush(self) -> None:
        return

    def reset_input_buffer(self) -> None:  # pragma: no cover - compatibility shim
        return

    def reset_output_buffer(self) -> None:  # pragma: no cover - compatibility shim
        return

    def close(self) -> None:
        self.is_open = False


class SenderServiceTests(unittest.TestCase):
    """Validate core behaviour of SenderService with a fake serial port."""

    def setUp(self) -> None:
        self.serial_patch = mock.patch(
            "cam_slicer.sender.service.serial.Serial", new=FakeSerial
        )
        self.serial_patch.start()
        self.service = SenderService()

    def tearDown(self) -> None:
        self.service.close()
        self.serial_patch.stop()
        FakeSerial.instances.clear()

    def test_open_and_status(self) -> None:
        """Service should open the fake port and report Idle status."""

        ok, _ = self.service.open("COM1")
        self.assertTrue(ok)
        # Allow RX thread to read the initial status line.
        time.sleep(0.1)
        status = self.service.status()
        self.assertEqual(status["port"], "COM1")
        self.assertTrue(status["state"].lower().startswith("idle"))
        self.assertTrue(status["last"])

    def test_enqueue_line_sends_command(self) -> None:
        """A line job should be written to the serial port."""

        self.service.open("COM1")
        job_id = self.service.enqueue_line("G0 X1")
        self.assertTrue(job_id)
        time.sleep(0.2)
        fake = FakeSerial.instances[-1]
        self.assertIn("G0 X1", " ".join(fake.written))

    def test_file_stream_emits_progress(self) -> None:
        """File streaming should report progress updates."""

        self.service.open("COM1")
        events: List[Dict[str, Any]] = []
        self.service.set_event_sink(events.append)
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("G0 X0\n")
            handle.write("G1 X1 F100\n")
            temp_path = Path(handle.name)
        try:
            job_id = self.service.enqueue_file(str(temp_path))
            time_limit = time.time() + 1.0
            while time.time() < time_limit:
                if any(
                    evt["type"] == "job"
                    and evt["data"].get("id") == job_id
                    and evt["data"].get("event") == "finished"
                    for evt in events
                ):
                    break
                time.sleep(0.05)
            progresses = [
                evt["data"].get("progress")
                for evt in events
                if evt["type"] == "job"
                and evt["data"].get("id") == job_id
                and evt["data"].get("event") == "progress"
            ]
            self.assertTrue(progresses)
            self.assertAlmostEqual(progresses[-1], 1.0, places=2)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_reject_jobs_when_disconnected(self) -> None:
        """Service should reject jobs when no serial port is open."""

        with self.assertRaises(SenderStateError):
            self.service.enqueue_line("G0 X0")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
