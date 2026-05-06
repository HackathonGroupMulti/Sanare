from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

async def main() -> None:
    try:
        from temporalio.client import Client
        from temporalio.worker import Worker
        from sanare.temporal_app import AnalyzeNoteWorkflow, analyze_note_activity
    except ImportError as exc:
        raise RuntimeError("Install temporalio to run the Temporal worker") from exc

    client = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "localhost:7233"))
    worker = Worker(
        client,
        task_queue=os.getenv("TEMPORAL_TASK_QUEUE", "clinical-analysis"),
        workflows=[AnalyzeNoteWorkflow],
        activities=[analyze_note_activity],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

