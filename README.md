# 📱 Implementasi MobileNetV2 dengan PyTorch

Proyek machine learning untuk image classification menggunakan MobileNetV2 dengan PyTorch. MobileNetV2 adalah model yang efisien dan ringan, cocok untuk aplikasi mobile dan edge devices.

## 🚀 Quick Start

### 1️⃣ Instalasi Dependencies

```bash
uv sync
```

### 2️⃣ Training Model

```bash
python train.py
```

### 3️⃣ Export ke ONNX

Konversi model PyTorch ke format ONNX untuk deployment:

```bash
python export.py <MODEL_PATH> <MODEL_OUT>
```

**Contoh:**

```bash
python ./export.py ./model/mobilenet.pth model/mobilenet.onnx
```

## 📁 Project Structure

- `train.py` - Script training model
- `export.py` - Script untuk export model ke ONNX
- `model/` - Direktori menyimpan trained models
- `dataset/` - Rekomendasi untuk menyimpan dataset

## 📝 Notes

- Model akan disimpan di folder `model/`
- Pastikan dataset sudah tersedia sebelum training
- Format hasil export: `.onnx` untuk deployment di berbagai platform
