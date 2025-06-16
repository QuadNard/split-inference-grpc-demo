# split-inference-grpc-demo

Splits LSTM inference into two gRPC microservices—“Prefill” (embedding) and “Decode”—with real-time Kafka → Flink → Grafana telemetry. This demo shows how to parallelize and scale the lightweight embedding stage separately from the heavier decode stage, track per-phase latency, and visualize metrics in Grafana.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Layout](#project-layout)
4. [Usage](#usage)
   - [Generate gRPC Stubs](#generate-grpc-stubs)
   - [Run Prefill Service](#run-prefill-service)
   - [Run Decode Service](#run-decode-service)
   - [Run Chained Client Simulator](#run-chained-client-simulator)
   - [Kafka & Flink Pipeline (Optional)](#kafka--flink-pipeline-optional)
   - [Grafana Dashboards (Optional)](#grafana-dashboards-optional)
5. [Architecture Diagram](#architecture-diagram)
6. [Development Notes](#development-notes)
7. [Contributing & Issues](#contributing--issues)
8. [License](#license)

---

## Prerequisites

- **Operating System**: Linux, macOS, or Windows Subsystem for Linux
- **Git** (for cloning the repo)
- **Python 3.8+**
  - `venv` or `virtualenv` for isolated environments
- **Docker & Docker Compose** (if you choose to containerize Kafka, Flink, and Grafana)
- **(Optional) Kafka & Flink CLI** for local pipelines
- **Grafana** (for dashboard visualization)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:<your-username>/split-inference-grpc-demo.git
   cd split-inference-grpc-demo
