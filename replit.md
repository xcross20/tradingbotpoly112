# Polymarket FastLoop Trader

## Overview
A Python script that trades Polymarket BTC 5-minute fast markets using real-time price momentum from Binance. Uses the Simmer API for market discovery and trade execution.

## Project Structure
- `fastloop_trader.py` - Main trading script (single-file, stdlib only)
- `config.json` - Trading configuration (thresholds, asset, window, etc.)
- `SKILL.md` - Skill documentation and usage instructions

## How to Run
- **Dry run:** `python3 fastloop_trader.py`
- **Live trading:** `python3 fastloop_trader.py --live`
- **With smart sizing:** `python3 fastloop_trader.py --live --smart-sizing --quiet`

## Requirements
- Python 3.11 (stdlib only, no pip packages needed)
- `SIMMER_API_KEY` environment variable (from simmer.markets/dashboard)

## Deployment
- Configured as a scheduled deployment for periodic execution
- Run command: `python3 fastloop_trader.py --live --quiet`
