import torch
import torch.nn as nn
from torchvision import models
import argparse
import os

def export():
    # 1. Setup CLI Arguments
    parser = argparse.ArgumentParser(description="Export PyTorch MobileNetV2 to ONNX")
    parser.add_argument("input", help="Path ke file .pth (state_dict)")
    parser.add_argument("output", help="Path untuk menyimpan file .onnx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Deteksi Jumlah Kelas secara Otomatis
    print(f"Checking model: {args.input}")
    try:
        state_dict = torch.load(args.input, map_location=device)
        
        # Pada MobileNetV2, layer terakhir namanya 'classifier.1.weight'
        # Kita ambil ukuran dimensi pertamanya (out_features)
        if 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].shape[0]
            print(f"Detected classes: {num_classes}")
        else:
            raise KeyError("Tidak bisa menemukan layer classifier.1.weight. Pastikan ini model MobileNetV2.")
            
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    # 3. Build Arsitektur Model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # 4. Load Bobot
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Export ke ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        args.output, 
        export_params=True, 
        opset_version=19,
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        external_data=False
    )
    
    if os.path.exists(args.output):
        print(f"Success! Model saved at: {args.output}")
    else:
        print("Export failed.")

if __name__ == "__main__":
    export()