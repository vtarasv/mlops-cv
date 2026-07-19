"""Trivial DAG proving the Airflow stack schedules and executes tasks (trigger it manually)."""

from __future__ import annotations

import logging

import pendulum
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="smoke_test",
    description="Stack smoke check: one empty operator, one in-process task.",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["smoke"],
)
def smoke_test() -> None:
    @task
    def say_ok() -> str:
        logger.info("airflow stack executes tasks: ok")
        return "ok"

    EmptyOperator(task_id="start") >> say_ok()


smoke_test()
