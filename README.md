# Prediction Market Retail Behavior Dashboard

A research and monitoring system for retail trading patterns in prediction markets, with a focus on Polymarket. It collects market data, builds retail behavior signals, and presents the results in a React dashboard.

## Features

- Retail behavior scoring based on trade size, off-hours activity, and burstiness
- Market lifecycle signals and liquidity/quality gates
- Category-level rollups and trend analytics
- FastAPI backend with a SQLite or PostgreSQL data store
- React dashboard for exploration and diagnostics

## Architecture

- **Collector**: pulls active Polymarket markets, snapshots, and trades on a schedule
- **Backend API**: FastAPI + SQLAlchemy, computes retail signals and serves JSON endpoints
- **Database**: SQLite by default, configurable via `DATABASE_URL`
- **Frontend**: React app that consumes the API

## Requirements

- Python 3.10+
- Node.js 18+

## Setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000`. Update `API_BASE` in `frontend/src/App.jsx` if your API runs elsewhere.

## Data Collection

Run a one-time collection and rollup:

```bash
cd backend
python3 collect_once.py
```

Run the continuous collector (5-minute high-frequency updates and hourly comprehensive updates):

```bash
cd backend
python3 scheduler.py
```

## Configuration

Set environment variables in a `.env` file at the repo root if needed:

```bash
DATABASE_URL=sqlite:///backend/prediction_markets.db
```

Other configuration options live in `backend/config.py`, including:

- `TRACKED_CATEGORIES`
- `MIN_MARKET_VOLUME_24H`
- `MIN_MARKET_LIQUIDITY`
- `RETAIL_MIN_TRADES`

## API Overview

- `GET /markets` List markets with latest stats and retail signals
- `GET /markets/{id}` Detailed market view with retail signals and lifecycle
- `GET /markets/{id}/history` Snapshot time series for charting
- `GET /markets/{id}/trades` Trade feed with retail analysis
- `GET /overview` Aggregate system metrics
- `GET /analytics/retail-index` Category rollups

## Operational Notes

- The collector only ingests active, non-archived Polymarket markets.
- The UI surfaces closed markets and warns when snapshots are stale.
- Data freshness depends on the collector schedule and the configured market filters.

## License

This project is for research and analytical use. Ensure compliance with Polymarket API terms.
