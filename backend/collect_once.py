#!/usr/bin/env python3
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from database import init_db
from scheduler import ContinuousDataCollector
from rollup_retail import build_retail_rollups
from config import ROLLUP_DAYS


def main():
    mode = os.getenv("COLLECT_MODE", "both").strip().lower()
    init_db()
    collector = ContinuousDataCollector()

    if mode in ("high", "hf", "fast"):
        collector.collect_high_frequency_data()
        return
    if mode in ("comprehensive", "full", "hourly"):
        collector.collect_comprehensive_update()
        build_retail_rollups(days=ROLLUP_DAYS)
        return

    collector.collect_high_frequency_data()
    collector.collect_comprehensive_update()
    build_retail_rollups(days=ROLLUP_DAYS)


if __name__ == "__main__":
    main()
