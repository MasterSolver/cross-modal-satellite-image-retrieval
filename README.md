# 🛰️ Cross-Modal Satellite Image Retrieval using DINOv2 & FAISS

A deep learning-based image retrieval system that performs **Same-Modal** and **Cross-Modal** satellite image retrieval using **DINOv2** embeddings and **FAISS** similarity search. The project provides an interactive **Streamlit** web application for retrieving visually and semantically similar satellite images from the BEN-14K dataset.

---

## 📌 Overview

Satellite images are collected using different sensors such as Sentinel-1 (SAR) and Sentinel-2 (Optical). Retrieving similar images across different modalities is a challenging computer vision task.

This project extracts feature embeddings using **Meta's DINOv2 Vision Transformer**, indexes them using **FAISS**, and retrieves the most similar images based on cosine similarity (L2 distance in normalized embedding space).

The application supports:

- Same-Modal Retrieval
- Cross-Modal Retrieval
- Interactive Streamlit Interface
- Fast Similarity Search using FAISS
- Top-K Image Retrieval

---

## 🚀 Features

- 🛰️ Cross-Modal Satellite Image Retrieval
- 🌍 Same-Modal Image Retrieval
- 🤖 DINOv2 Feature Extraction
- ⚡ FAISS Vector Search
- 🖥️ Interactive Streamlit UI
- 📂 Automatic Sentinel-1 & Sentinel-2 Image Loading
- 🎯 Top-K Similar Image Search
- 🧩 Modular Project Structure

---

## 🏗️ Project Architecture

```
                   Query Image
                        │
                        ▼
             Image Preprocessing
                        │
                        ▼
              DINOv2 Feature Extractor
                        │
                        ▼
                 Feature Embedding
                        │
                        ▼
                  FAISS Index Search
                        │
                        ▼
          Top-K Similar Satellite Images
```

---

## 📁 Project Structure

```
Cross-Modal-Satellite-Retrieval/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│   ├── dinov2.py
│   └── feature_extractor.py
│
├── retrieval/
│   ├── faiss_index.py
│   ├── metrics.py
│   └── search.py
│
├── utils/
│   ├── preprocessing.py
│   └── embedding_generator.py
│
├── outputs/
│   ├── embeddings/
│   └── faiss_index/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Vision Model | DINOv2 |
| Similarity Search | FAISS |
| Web App | Streamlit |
| Image Processing | Rasterio, NumPy |
| Dataset | BEN-14K |

---

## 📂 Dataset

This project uses the **BEN-14K (BigEarthNet-14K)** dataset.

**Dataset Download**

https://drive.google.com/file/d/1dl81O9Qm-8HUt59mNHCDrYkgTgwNgDi2/view?usp=drive_link

After downloading, place the dataset inside the `data/` directory.

Example:

```
data/
└── BEN-14K/
    ├── s1/
    ├── s2/
    ├── benv1_14k_dataset.parquet
    └── benv1_14k_dataset_master_labels.csv
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/mastersolver/Cross-Modal-Satellite-Retrieval.git

cd Cross-Modal-Satellite-Retrieval
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

---

## 📖 Usage

1. Select Retrieval Mode
   - Same Modal
   - Cross Modal

2. Select Query Modality
   - Sentinel-1
   - Sentinel-2

3. Enter the sample folder path.

4. Select Top-K results.

5. Click **Retrieve**.

The application will display the most similar satellite images along with their similarity distances.

---

## 📷 Sample Output

### Query Image

(Add Screenshot Here)

### Retrieved Images

(Add Screenshot Here)

---

## 🔍 Retrieval Pipeline

```
Query Folder
      │
      ▼
Preprocessing
      │
      ▼
DINOv2 Feature Extraction
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top-K Retrieved Images
```

---

## 📈 Future Improvements

- Evaluation Metrics (Precision@K, Recall@K, F1@K)
- Deploy using Hugging Face Spaces
- Image Upload Support
- GPU Acceleration
- Better Visualization
- Support for Additional Remote Sensing Datasets

---

## 📚 References

- DINOv2
- FAISS
- BigEarthNet
- Streamlit
- PyTorch

---

## 👨‍💻 Author

**Shashwat Kumar**

GitHub: https://github.com/MasterSolver

LinkedIn: https://www.linkedin.com/in/mastersolver

---

## 📄 License

This project is licensed under the MIT License.
