"""Incremental CSV/JSONL output for long-running NPU benchmark matrices."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

from .timing import Samples, summarize

Scalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = Scalar | list["JsonValue"] | dict[str, "JsonValue"]


class ResultWriter:
    """Write one summary row and one raw-sample record per operation.

    The destination must not already exist. Files are flushed after every
    operation so a device reset or process failure preserves completed cases.
    """

    def __init__(self, output_dir: Path, metadata: Mapping[str, JsonValue]) -> None:
        output_dir.mkdir(parents=True, exist_ok=False)
        self.output_dir = output_dir
        self._summary_path = output_dir / "summary.csv"
        self._samples_path = output_dir / "samples.jsonl"
        self._metadata_path = output_dir / "metadata.json"
        self._metadata = dict(metadata)
        self._summary_file = self._summary_path.open("w", newline="", encoding="utf-8")
        self._samples_file = self._samples_path.open("w", encoding="utf-8")
        self._writer: csv.DictWriter[str] | None = None
        self._closed = False
        self._write_metadata(dict(metadata), "running")

    def _write_metadata(self, metadata: dict[str, JsonValue], status: str) -> None:
        payload = dict(metadata)
        payload["status"] = status
        temporary = self._metadata_path.with_name("metadata.json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._metadata_path)

    def write(
        self, case_fields: Mapping[str, Scalar], operation: str, samples: Samples
    ) -> None:
        if self._closed:
            raise RuntimeError("result writer is closed")
        if len(samples.device_us) != len(samples.wall_us):
            raise ValueError("device and wall samples must have equal lengths")
        device = summarize(samples.device_us)
        wall = summarize(samples.wall_us)
        row: dict[str, Scalar] = {str(k): v for k, v in case_fields.items()}
        row["operation"] = operation
        for prefix, stats in (("device_us", device), ("wall_us", wall)):
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
        payload_bytes = float(case_fields.get("payload_bytes", 0) or 0)
        if operation == "plan":
            row["logical_payload_gb_per_s"] = ""
        elif payload_bytes == 0:
            row["logical_payload_gb_per_s"] = 0.0
        else:
            row["logical_payload_gb_per_s"] = (
                "" if device["mean"] == 0 else payload_bytes / (device["mean"] * 1000.0)
            )
        if self._writer is None:
            fields = list(row)
            self._writer = csv.DictWriter(self._summary_file, fieldnames=fields)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._summary_file.flush()
        self._samples_file.write(
            json.dumps(
                {
                    "case": dict(case_fields),
                    "operation": operation,
                    "device_us": samples.device_us,
                    "wall_us": samples.wall_us,
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._samples_file.flush()

    def finish(self) -> None:
        if self._closed:
            return
        try:
            self._summary_file.close()
            self._samples_file.close()
        finally:
            self._closed = True
        self._write_metadata(self._metadata, "measurements_finished")

    def fail(self) -> None:
        if self._closed:
            return
        try:
            self._summary_file.close()
            self._samples_file.close()
        finally:
            self._closed = True
        self._write_metadata(self._metadata, "failed")

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.finish()
        else:
            try:
                self.fail()
            except OSError as error:
                try:
                    message = f"could not persist benchmark failure status: {error}\n"
                    os.write(2, message.encode("utf-8", errors="backslashreplace"))
                except OSError:
                    pass  # Preserve the original failure if stderr is also unavailable.
