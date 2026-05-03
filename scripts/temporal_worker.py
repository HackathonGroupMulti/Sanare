from __future__ import annotations

import asyncio
import os

from iasis.temporal_app import AnalyzeNoteWorkflow, analyze_note_activity


async def main() -> None:
    try:
        from temporalio.client import Client
        from temporalio.worker import Worker
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
