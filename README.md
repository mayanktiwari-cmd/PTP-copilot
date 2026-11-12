# Real‑Time Fake‑PTP Detection & In‑Call Guidance

[Demo](https://open-iit-navy.vercel.app/overview) – Interactive showcase of the real‑time copilot.

---

## Overview
This repository provides an end‑to‑end solution for detecting risky "Promise‑to‑Pay" (PTP) commitments in real time during a customer call and delivering actionable guidance to agents. It combines a robust data pipeline, feature engineering, LightGBM models, an optional LoRA‑adapted small language model (SLM), and a production‑grade FastAPI + React web application for monitoring and audit.

---

## Architecture
The system is organized into three tightly coupled layers:
1. **Data Ingestion & Ground‑Truth Construction** – CSV extracts are loaded, validated, and the business "kept" label is derived (≥ 90 % of promised amount paid within the grace window).
2. **Feature Engineering & Model Training** – Time‑aware lexical, acoustic, financial, and agent‑integrity features are generated per turn. LightGBM models (pre‑call, in‑call, post‑call) are trained, complemented by a cost‑sensitive multiclass model for the six behavioural categories. An optional Qwen2.5‑0.5B LoRA adapter can be fine‑tuned for enhanced NLP scoring.
3. **Streaming In‑Call Copilot & UI** – `streaming.py` processes each ASR‑finalised utterance, updates cumulative features, runs the models, applies guardrails, and emits a confidence‑gated decision. The FastAPI backend and React frontend visualise predictions, latency, and audit trails in real time.

---

## Deliverables
| Deliverable | Location |
|---|---|
| Python package (`ptp_copilot/`) | `ptp_copilot/` |
| Enriched prediction CSV (3,696 rows) | `outputs/enriched_ptp_predictions.csv` |
| Ground‑truth tables | `outputs/outcome_labels.csv`, `outputs/ground_truth_labels.csv` |
| Evaluation reports (AUC, ECE, calibration, etc.) | `outputs/results.json`, `outputs/pipeline_log.txt`, `outputs/figures/*.png` |
| Live‑call replay markdown | `outputs/streaming_demo.md` |
| Latency benchmarks | `outputs/latency_tabular.json`, `outputs/latency_slm.json` |
| Agent‑integrity dashboard | `outputs/agent_integrity_dashboard.csv` |
| SLM LoRA adapter + merged weights | `outputs/slm_lora/` |
| Full‑stack web app (FastAPI + React) | `webapp/` |

---

## Key Performance Metrics (held‑out test split)
| Metric | Result |
|---|---|
| In‑call keep‑prediction AUC | **0.763** |
| Calibration (ECE) | **0.037** |
| Broken‑PTP flag recall / precision | 88.4 % / 84 % |
| Hardship mis‑classification (cost‑sensitive) | **1.6 %** |
| Six‑way macro‑F1 (cost‑sensitive) | **0.75** |
| CPU latency (p50 / p95) | **60 ms / 78 ms** per turn |
| SLM LoRA latency (RTX 3050) | **47 ms / 57 ms** (≈ 0.74 GB VRAM) |
| Third‑party disclosure | **0 %** (baseline 78 %) |

---

## Installation & Execution
```bash
# Install core dependencies
pip install -r requirements-render.txt

# Run the full pipeline (add --slm for the GPU track)
python run_all.py

# Start the web UI (FastAPI + React)
python webapp/start.py   # then open http://localhost:8000 in a browser
```

---

## Repository Layout
```
.
├─ ptp_copilot/          # Core library (pipeline, models, features)
├─ webapp/               # FastAPI backend + React frontend
├─ outputs/              # Generated artefacts (CSV, JSON, figures, models)
├─ csv_folder/           # Raw data extracts
├─ experiments/          # Evaluation scripts & notebooks
├─ Dockerfile            # Optional container build
└─ README.md             # This document
```

---

## Contribution Guidelines
Contributions are welcome. Please see `CONTRIBUTING.md` for the development workflow, testing procedures, and coding standards.

---

## License
MIT License.

---

*Prepared by Antigravity coding assistant.*