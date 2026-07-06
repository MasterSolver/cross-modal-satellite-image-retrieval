# Cross-Modal Satellite Image Retrieval

An AI-powered satellite image retrieval system developed for the **ISRO Bharatiya Antariksh Hackathon 2026**.

## Overview

This project retrieves visually and semantically similar satellite images using deep learning and vector similarity search.

The system extracts feature embeddings from satellite imagery using **DINOv2**, stores them in a **FAISS** vector database, and retrieves the most similar images in real time.

The long-term objective is to support **cross-modal retrieval** between Sentinel-1 (SAR) and Sentinel-2 (Optical) imagery.

---

## Features

- Semantic satellite image retrieval
- DINOv2 feature extraction
- FAISS similarity search
- Fast image retrieval
- Streamlit web interface
- Cross-modal retrieval support (Work in Progress)

---

## Tech Stack

- Python
- PyTorch
- DINOv2
- FAISS
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## Project Structure

```
cross-modal-satellite-image-retrieval/
│
├── app/
├── data/
├── models/
├── outputs/
├── utils/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

Dataset

↓

Image Preprocessing

↓

Feature Extraction (DINOv2)

↓

Embedding Generation

↓

FAISS Index

↓

Similarity Search

↓

Retrieved Images

---

## Team

**Agentic Minds**

- Shashwat Kumar
- Lakshaya Kushwaha
- Tanishka Bhatia

---

## Status

🚧 Currently under development.