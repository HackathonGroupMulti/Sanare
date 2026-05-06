from __future__ import annotations

from datetime import timedelta

from sanare.pipeline import ClinicalPipeline

try:
    from temporalio import activity, workflow
except ImportError:
    activity = None
    workflow = None


if activity and workflow:

    @activity.defn
    async def analyze_note_activity(text: str) -> dict:
        return ClinicalPipeline().analyze(text).model_dump(mode="json")


    @workflow.defn
    class AnalyzeNoteWorkflow:
        @workflow.run
        async def run(self, text: str) -> dict:
            return await workflow.execute_activity(
                analyze_note_activity,
                text,
                start_to_close_timeout=timedelta(seconds=45),
            )


