#!/usr/bin/env python3
"""
Automated Model Retraining Scheduler
Runs retraining pipeline on a schedule and provides API endpoints for manual triggering.
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import schedule

from model_retraining import main as run_retraining_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RetrainingStatus(BaseModel):
    """Model for retraining status response."""

    status: str
    last_run: Optional[str] = None
    next_scheduled: Optional[str] = None
    is_running: bool = False
    results: Optional[Dict] = None


class RetrainingRequest(BaseModel):
    """Model for manual retraining request."""

    model_type: Optional[str] = None  # 'housing' or None for both
    force: bool = False  # Force retraining even if performance is good


class RetrainingScheduler:
    """Manages automated model retraining schedule."""

    def __init__(self):
        self.is_running = False
        self.last_run = None
        self.last_results = None
        self.schedule_thread = None
        self.setup_schedule()

    def setup_schedule(self):
        """Setup the retraining schedule."""
        # Schedule retraining every day at 2 AM
        schedule.every().day.at("02:00").do(self.run_scheduled_retraining)

        # Schedule performance check every 6 hours
        schedule.every(6).hours.do(self.check_performance)

        logger.info("Retraining schedule configured:")
        logger.info("- Full retraining: Daily at 2:00 AM")
        logger.info("- Performance check: Every 6 hours")

    def start_scheduler(self):
        """Start the background scheduler thread."""
        if self.schedule_thread is None or not self.schedule_thread.is_alive():
            self.schedule_thread = threading.Thread(
                target=self._run_scheduler, daemon=True
            )
            self.schedule_thread.start()
            logger.info("Retraining scheduler started")

    def _run_scheduler(self):
        """Run the scheduler in a background thread."""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def run_scheduled_retraining(self):
        """Run the scheduled retraining pipeline."""
        if self.is_running:
            logger.warning("Retraining already in progress, skipping scheduled run")
            return

        logger.info("Starting scheduled retraining...")
        self.run_retraining()

    def check_performance(self):
        """Check model performance without retraining."""
        logger.info("Running performance check...")

        try:
            from model_retraining import ModelPerformanceMonitor

            monitor = ModelPerformanceMonitor()

            housing_perf = monitor.evaluate_model_performance("housing")

            # Log performance status
            if housing_perf.get("needs_retraining", False):
                logger.warning(
                    "Housing model performance degraded, retraining recommended"
                )

            # Save performance check results
            performance_results = {
                "check_time": datetime.now().isoformat(),
                "housing": housing_perf,
            }

            with open("performance_check.json", "w") as f:
                json.dump(performance_results, f, indent=2)

        except Exception as e:
            logger.error(f"Performance check failed: {e}")

    def run_retraining(
        self, model_type: Optional[str] = None, force: bool = False
    ) -> Dict:
        """Run the retraining pipeline."""
        if self.is_running:
            return {"status": "error", "message": "Retraining already in progress"}

        self.is_running = True
        start_time = datetime.now()

        try:
            logger.info(
                f"Starting retraining pipeline (model_type={model_type}, force={force})"
            )

            # Run the retraining pipeline
            results = run_retraining_pipeline()

            self.last_run = start_time.isoformat()
            self.last_results = results

            logger.info("Retraining pipeline completed successfully")
            return {
                "status": "success",
                "results": results,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            }
        finally:
            self.is_running = False

    def get_status(self) -> RetrainingStatus:
        """Get current retraining status."""
        next_scheduled = None
        try:
            # Get next scheduled job time
            jobs = schedule.jobs
            if jobs:
                next_job = min(jobs, key=lambda job: job.next_run)
                next_scheduled = next_job.next_run.isoformat()
        except:
            pass

        return RetrainingStatus(
            status="running" if self.is_running else "idle",
            last_run=self.last_run,
            next_scheduled=next_scheduled,
            is_running=self.is_running,
            results=self.last_results,
        )


# Global scheduler instance
scheduler = RetrainingScheduler()

# FastAPI app for retraining management
app = FastAPI(
    title="MLOps Model Retraining Service",
    description="Automated model retraining and performance monitoring service",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Start the scheduler when the app starts."""
    scheduler.start_scheduler()
    logger.info("Retraining service started")


@app.get("/status", response_model=RetrainingStatus)
async def get_retraining_status():
    """Get current retraining status."""
    return scheduler.get_status()


@app.post("/retrain")
async def trigger_retraining(
    request: RetrainingRequest, background_tasks: BackgroundTasks
):
    """Manually trigger model retraining."""
    if scheduler.is_running:
        return {"status": "error", "message": "Retraining already in progress"}

    # Run retraining in background
    background_tasks.add_task(
        scheduler.run_retraining, model_type=request.model_type, force=request.force
    )

    return {
        "status": "started",
        "message": "Retraining pipeline started in background",
        "model_type": request.model_type,
        "force": request.force,
    }


@app.get("/performance")
async def get_performance_metrics():
    """Get latest performance metrics."""
    try:
        # Check if performance check file exists
        if os.path.exists("performance_check.json"):
            with open("performance_check.json", "r") as f:
                return json.load(f)
        else:
            return {"message": "No performance data available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/results")
async def get_retraining_results():
    """Get latest retraining results."""
    try:
        if os.path.exists("retraining_results.json"):
            with open("retraining_results.json", "r") as f:
                return json.load(f)
        else:
            return {"message": "No retraining results available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "retraining-scheduler",
        "timestamp": datetime.now().isoformat(),
        "scheduler_running": scheduler.schedule_thread is not None
        and scheduler.schedule_thread.is_alive(),
    }


if __name__ == "__main__":
    import uvicorn

    # Start the scheduler
    scheduler.start_scheduler()

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8002)
