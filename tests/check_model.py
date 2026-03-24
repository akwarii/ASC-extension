import torch
import json
import os

def test_load_example_model():
    ckpt_file = "examples/painn.pt2"
    if not os.path.exists(ckpt_file):
        return
        
    extra_files = {"metadata.json": ""}
    try:
        program = torch.export.load(ckpt_file, extra_files=extra_files)
        print("Model loaded successfully")
        metadata = json.loads(extra_files["metadata.json"])
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    test_load_example_model()
