# ==============================================================================
# SCRIPT TO FINE-TUNE THE MODEL ON USER CORRECTIONS (v4 - Final Bug Fix)
# ==============================================================================
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
import traceback
import random
from collections import defaultdict
import time
import torch

print("--- Starting Fine-Tuning Process ---")

# --- 1. Define Paths ---
MODEL_PATH = Path("best.pt")
FEEDBACK_DIR = Path("feedback_data")
FINETUNE_YAML_PATH = FEEDBACK_DIR / "finetune_data.yaml"
RESTART_FLAG_FILE = Path("_RESTART_REQUIRED_")
NEW_CLASS_FILE = FEEDBACK_DIR / "_new_class_name.txt"

try:
    # --- 2. Check for Feedback Data ---
    labels_dir = FEEDBACK_DIR / "labels"
    images_dir = FEEDBACK_DIR / "images"
    if not labels_dir.exists() or not any(labels_dir.iterdir()):
        print("No new feedback data found. Exiting.")
        exit()
    
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Could not find 'best.pt' model to fine-tune.")
    
    # --- 3. Prepare the data.yaml file for fine-tuning ---
    # --- THIS IS THE FIX ---
    # Load the model into a single, consistent variable name: 'model'
    model = YOLO(MODEL_PATH)
    class_names = list(model.names.values())
    
    if NEW_CLASS_FILE.exists():
        with open(NEW_CLASS_FILE, 'r') as f:
            for line in f:
                new_class = line.strip()
                if new_class and new_class not in class_names:
                    class_names.append(new_class)
        os.remove(NEW_CLASS_FILE)
    
    data_yaml = {
        'path': str(FEEDBACK_DIR.resolve()),
        'train': 'images',
        'val': 'images',
        'nc': len(class_names),
        'names': class_names
    }
    with open(FINETUNE_YAML_PATH, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print("✅ Created temporary YAML for fine-tuning.")
    
    # --- 4. Intelligent Oversampling for Feedback Data ---
    class_counts = defaultdict(int)
    class_to_images = defaultdict(list)
    all_feedback_labels = list(labels_dir.glob("*.txt"))
    for label_file in all_feedback_labels:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                if label_file.stem not in class_to_images[class_id]:
                    class_to_images[class_id].append(label_file.stem)
    
    target_count = max(30, max(class_counts.values()) if class_counts else 30)
    print(f"Balancing classes to a target of ~{target_count} instances...")
    for class_id, count in class_counts.items():
        if count < target_count:
            num_to_add = target_count - count
            image_pool = class_to_images[class_id]
            if not image_pool: continue
            for i in range(num_to_add):
                stem = random.choice(image_pool)
                original_img_path = next((p for p in images_dir.glob(f"{stem}.*") if p.is_file()), None)
                if original_img_path:
                    shutil.copy(original_img_path, images_dir / f"{stem}_aug_{i}{original_img_path.suffix}")
                    shutil.copy(labels_dir / f"{stem}.txt", labels_dir / f"{stem}_aug_{i}.txt")

    # --- 5. Run the Fine-Tuning ---
    print("\n--- Starting fine-tuning for a few epochs... ---")
    
    # Automatically detect if a GPU is available
    device_to_use = 0 if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {'GPU (cuda:0)' if device_to_use == 0 else 'CPU'}")
    
    # --- THIS IS THE FIX ---
    # We now call .train() on the same 'model' variable we defined earlier
    results = model.train(
        data=str(FINETUNE_YAML_PATH),
        epochs=50,
        imgsz=640,
        batch=4,
        patience=15,
        name='finetuned_from_feedback',
        device=device_to_use,
        exist_ok=True
    )

    # --- 6. Backup the old model and deploy the new one ---
    new_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if new_model_path.is_file():
        backup_path = MODEL_PATH.with_suffix(f'_backup_{int(time.time())}.pt')
        MODEL_PATH.rename(backup_path)
        print(f"✅ Old model backed up to: {backup_path}")
        
        os.replace(new_model_path, MODEL_PATH)
        print(f"\n🎉🎉🎉 FINE-TUNING COMPLETE! The main model 'best.pt' has been updated. 🎉🎉🎉")
        
        RESTART_FLAG_FILE.touch()
        print("Server restart has been requested.")
        shutil.rmtree(FEEDBACK_DIR)
        
    else:
        print("\n Fine-tuning did not produce a new model file.")

except Exception as e:
    print(f"An error occurred during fine-tuning: {e}")
    traceback.print_exc()

