# 🦉 Smart Dyslexia Detector

An AI-powered web application for early dyslexia screening through handwriting analysis.

## 🎯 Overview

The Smart Dyslexia Detector uses YOLOv11 to analyse children's handwriting and detect potential dyslexia indicators (letter reversals). It provides real-time feedback, progress tracking, and gamified practice activities.

## ✨ Features

- **Image Upload** – Upload handwriting photos for analysis
- **Canvas Writing** – Write directly on screen
- **Real-time Analysis** – Instant feedback with per-letter breakdown
- **Progress Tracking** – Monitor improvement over time
- **Spelling Bee Game** – Interactive practice
- **PDF Worksheets** – Printable practice materials
- **History** – View past screenings

## 📊 Performance

| Metric | Score |
|--------|-------|
| Precision | 99.92% |
| Recall | 99.88% |
| F1-Score | 99.90% |
| mAP@0.5 | 99.50% |

## 🚀 Installation

### Prerequisites

- Python 3.11+
- MySQL (XAMPP recommended)
- Tesseract OCR

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/smartDyslexiaDetector_v2.git
cd smartDyslexiaDetector_v2