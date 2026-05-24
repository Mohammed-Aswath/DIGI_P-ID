<!-- Keywords: DigiP&ID, P&ID digitization, AI engineering, piping and instrumentation diagram, OCR, YOLO, DXF, semantic graph, digital twin, industrial AI, computer vision engineering -->

<div align="center">

```
██████╗ ██╗ ██████╗ ██╗██████╗ ██╗██████╗
██╔══██╗██║██╔════╝ ██║██╔══██╗╚═╝██╔══██╗
██║  ██║██║██║  ███╗██║██████╔╝   ██║  ██║
██║  ██║██║██║   ██║██║██╔═══╝    ██║  ██║
██████╔╝██║╚██████╔╝██║██║    ██╗ ██████╔╝
╚═════╝ ╚═╝ ╚═════╝ ╚═╝╚═╝    ╚═╝ ╚═════╝
```

### AI-Powered P&ID Semantic Digitization & Engineering Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![NetworkX](https://img.shields.io/badge/NetworkX-FF6600?style=for-the-badge&logo=python&logoColor=white)](https://networkx.org)

> Transforming static Piping & Instrumentation Diagrams (P&IDs) into intelligent, editable, machine-readable digital engineering models using AI, Computer Vision, and Graph Reconstruction.

</div>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
  - [Symbol Detection](#symbol-detection)
  - [Multi-Layer OCR System](#multi-layer-ocr-system)
  - [Metadata Extraction Engine](#metadata-extraction-engine)
  - [Engineering Line Detection](#engineering-line-detection)
  - [Semantic Graph Construction](#semantic-graph-construction)
  - [DXF Reconstruction](#dxf-reconstruction)
  - [Agentic AI Engineering Editor](#agentic-ai-engineering-editor)
- [System Workflow](#system-workflow)
- [Architecture](#architecture)
- [Engineering Graph Reconstruction](#engineering-graph-reconstruction)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Current Features](#current-features)
- [Future Roadmap](#future-roadmap)
  - [Engineering Intelligence](#engineering-intelligence)
  - [CAD \& Reconstruction](#cad--reconstruction)
  - [AI Engineering Systems](#ai-engineering-systems)
  - [Digital Twin Enablement](#digital-twin-enablement)
- [Project Vision](#project-vision)

---

## Overview

Industrial plants rely heavily on:

- Scanned P&IDs and PDF engineering drawings
- Raster process diagrams and CAD exports
- Dense engineering documentation

These documents contain critical engineering information but are inherently difficult to analyze, edit, search, and integrate into modern intelligent systems.

**DigiP&ID** resolves this gap by converting static engineering drawings into:

| Output | Description |
|---|---|
| **Structured Engineering Graphs** | Topology-aware digital representations of process connectivity |
| **Editable Digital Diagrams** | CAD-compatible, modifiable engineering drawings |
| **Machine-Readable Models** | JSON/XML-structured engineering datasets |
| **Intelligent Engineering Assets** | Semantically enriched process models for AI consumption |

The system reconstructs **engineering symbols**, **piping topology**, **semantic relationships**, **metadata**, **process connectivity**, and **editable CAD structures** directly from raw static engineering diagrams.

---

## Core Capabilities

### Symbol Detection

DigiP&ID uses a **YOLO-based object detection pipeline** trained on industrial P&ID datasets to identify:

- Valves, pumps, and compressors
- Instruments and process equipment
- Engineering symbols and annotations

Each detected symbol is returned with a **class label**, **confidence score**, **bounding box**, and **spatial coordinates**.

---

### Multi-Layer OCR System

DigiP&ID implements a **hybrid OCR fusion architecture** for robust engineering tag extraction.

| OCR Engine | Role |
|---|---|
| **DocTR** | Deep learning-based document text recognition |
| **PaddleOCR Microservice** | High-performance multilingual OCR |
| **Tesseract** | Classical OCR fallback layer |
| **Symbol-Guided OCR** | Context-aware tag extraction near symbols |
| **Region-Guided OCR** | Spatially constrained text localization |

Outputs from all engines are **cross-validated and fused** to improve tag accuracy, reduce ambiguity, and recover difficult engineering annotations.

---

### Metadata Extraction Engine

The metadata pipeline extracts structured information from:

- Title blocks and revision tables
- Engineering notes and specifications
- Drawing-level metadata

Outputs are generated in **JSON**, **Markdown**, and **structured engineering summaries**.

---

### Engineering Line Detection

The line extraction pipeline applies:

```
Thresholding → Skeletonization → Edge Detection
      ↓
  Hough Transform → Line Merging → Junction Analysis
```

This enables full **piping extraction**, **topology recovery**, **line classification**, and **process connectivity reconstruction**.

---

### Semantic Graph Construction

One of the **core innovations** of DigiP&ID is its topology-aware semantic graph engine.

Engineering drawings are converted into structured graph models using **NetworkX**, **Shapely**, **junction clustering**, and **graph traversal algorithms**.

```
Symbols       →  Graph Nodes
Pipelines     →  Graph Edges
Relationships →  Semantic Attributes
```

This graph becomes the foundation for intelligent editing, semantic querying, and engineering analysis.

---

### DXF Reconstruction

The DXF reconstruction engine regenerates **editable CAD engineering drawings** from digitized data, supporting:

- Piping layouts and engineering topology
- Process connectivity and symbol-aware placement
- Geometry-aware CAD reconstruction
- Enterprise-compatible editable structures

The reconstruction pipeline is designed to **preserve engineering semantics** while producing fully editable engineering drawings.

---

### Agentic AI Engineering Editor

DigiP&ID includes an **AI-assisted engineering editor** capable of:

- Full **CRUD operations** on engineering diagrams
- Semantic modifications and topology-aware editing
- Intelligent engineering corrections and updates

This elevates the platform from a static digitization tool to a fully **interactive engineering intelligence system**.

---

## System Workflow

```
┌─────────────────────────────────┐
│       Input PDF / Image         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Image Preprocessing       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      YOLO Symbol Detection      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       OCR Fusion Pipeline       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Line Detection & Skeletonization│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Topology Reconstruction     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Semantic Graph Generation    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Metadata Extraction        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       DXF Reconstruction        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   AI-Assisted Engineering Edit  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Structured Digital Eng. Model  │
└─────────────────────────────────┘
```

---

## Architecture

DigiP&ID is organized into five layered processing tiers:

```
╔══════════════════════════════════════╗
║         AI INTERACTION LAYER         ║  ← Intelligent editing, topology-aware updates
╠══════════════════════════════════════╣
║         CAD GENERATION LAYER         ║  ← DXF reconstruction, geometry regeneration
╠══════════════════════════════════════╣
║           SEMANTIC LAYER             ║  ← Engineering relationships, graph intelligence
╠══════════════════════════════════════╣
║        RECONSTRUCTION LAYER          ║  ← Line extraction, junction analysis, graphs
╠══════════════════════════════════════╣
║          DETECTION LAYER             ║  ← Symbol detection, OCR, tag recognition
╚══════════════════════════════════════╝
```

| Layer | Responsibilities |
|---|---|
| **Detection** | Symbol detection · Text localization · OCR extraction · Engineering tag recognition |
| **Reconstruction** | Line extraction · Junction analysis · Graph generation · Connectivity propagation |
| **Semantic** | Engineering relationships · Metadata structuring · Topology awareness · Graph intelligence |
| **CAD Generation** | DXF reconstruction · Geometry regeneration · Editable engineering layouts |
| **AI Interaction** | Intelligent editing · Engineering operations · Topology-aware updates · Semantic modifications |

---

## Engineering Graph Reconstruction

The graph generation pipeline reconstructs full engineering topology from extracted line structures through the following sequence:

1. **Detect and filter** engineering lines from the drawing
2. **Remove** overlapping symbol and text regions
3. **Convert** line segments into graph edges
4. **Cluster** junction endpoints spatially
5. **Merge** nearby graph vertices
6. **Reconstruct** engineering topology
7. **Assign** semantic relationships to graph elements
8. **Propagate** engineering connectivity across the graph

This enables:

- Nearest-neighbor engineering mapping
- Junction-aware graph traversal
- Process flow reconstruction
- Topology intelligence and semantic connectivity analysis

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Backend** | Python · Flask |
| **Computer Vision & AI** | YOLO · OpenCV · DocTR · PaddleOCR · Tesseract |
| **Graph & Geometry** | NetworkX · Shapely · Skeletonization Algorithms |
| **CAD & Reconstruction** | DXF Generation Pipeline · Geometry Reconstruction Engine |
| **Frontend** | Plotly Interactive Editor · HTML · CSS · JavaScript |
| **Data Formats** | JSON · XML · ISO-15926-style Structures · DXF |

---

## Repository Structure

```
project/
│
├── app.py                    # Main Flask application entry point
├── pid_to_dxf.py             # DXF reconstruction pipeline
├── metadata_extractor.py     # Engineering metadata extraction engine
├── table_detector.py         # Table and title block detector
├── paddle_ocr_client.py      # PaddleOCR client interface
├── paddle_ocr_service.py     # PaddleOCR microservice
│
├── models/                   # Trained YOLO and AI models
├── static/                   # Static frontend assets
├── templates/                # HTML templates
├── tests/                    # Unit and integration tests
├── feedback_data/            # Feedback-driven correction data
├── outputs/                  # Generated outputs and exports
└── symbol_templates/         # Engineering symbol template library
```

---

## Current Features

- [x] AI-based P&ID symbol detection (YOLO)
- [x] Multi-engine OCR fusion pipeline
- [x] Engineering metadata extraction
- [x] Line and junction detection
- [x] Semantic graph construction
- [x] DXF CAD generation
- [x] Interactive engineering editor (Plotly)
- [x] AI-assisted editing workflows
- [x] Feedback-driven model correction pipeline

---

## Future Roadmap

### Engineering Intelligence
- Enhanced semantic analysis and advanced topology reasoning
- Automated engineering validation
- Intelligent process understanding

### CAD & Reconstruction
- Advanced vector-level symbol rendering
- Reusable engineering symbol template libraries
- High-fidelity CAD regeneration
- Enterprise CAD interoperability

### AI Engineering Systems
- Engineering copilots and intelligent modification agents
- Process optimization assistance
- Automated engineering workflows

### Digital Twin Enablement
- Real-time synchronization with operational systems
- Industrial telemetry integration
- Operational state modeling
- Intelligent plant representation

---

## Project Vision

```
        Static Engineering Diagram
                    │
                    ▼
          Semantic Reconstruction
                    │
                    ▼
       Engineering Intelligence Layer
                    │
                    ▼
      Editable Digital Engineering Model
                    │
                    ▼
        Intelligent Industrial Systems
```

DigiP&ID establishes a structured **engineering intelligence layer** capable of supporting intelligent engineering operations, semantic process analysis, editable engineering workflows, next-generation industrial systems, and future **Digital Twin ecosystems**.

---

<div align="center">

**DigiP&ID** &nbsp;|&nbsp; AI-Powered P&ID Semantic Digitization and Engineering Intelligence Platform

</div>