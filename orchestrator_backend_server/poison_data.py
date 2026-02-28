import os
import argparse
import shutil
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path
import random
import json
from typing import List
import pickle

def add_gaussian_noise(image, std_dev=0.1):
    """Adaugă zgomot Gaussian adaptat la dimensiunea imaginii."""
    img_array = np.array(image)
    noise = np.random.normal(0, std_dev * 255, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def label_flip(class_names, current_class):
    """Flip label la o clasă aleatoare diferită."""
    new_class = random.choice([c for c in class_names if c != current_class])
    return new_class

def insert_backdoor(image, trigger_size=10, color=(255, 255, 255)):
    """Adaugă un pătrat backdoor în colț."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle([width - trigger_size, height - trigger_size, width, height], fill=color)
    return image

def write_attack_info_json(output_dir, test_file,nn_name, operation, intensity, percentage):
    
    from datetime import datetime, timezone
    attack_info = {
        "attack_type": "data_poisoning",
        "nn_name": nn_name,
        "method": operation,
        "intensity": intensity,
        "percentage": float(percentage),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(attack_info, f, ensure_ascii=False, indent=2)

    print(f"Created attack info file: {test_file}")
    print(f"Poisoned data saved to {output_dir}")


def extract_labels(input_dir: str) -> List[str]:
    """
    Extrage etichetele (labels/class names) din datele unui model NN/ML.
    
    Suportă formate:
    - Pickled data (train_data.pkl, test_data.pkl cu (images, labels))
    - Structură de directoare (train/ și test/ cu subdirectoare per clasă)
    - metadata.json (cu 'class_names' sau 'num_classes')
    
    Args:
        input_dir: Calea către directorul cu date (e.g., clean_data)
    
    Returns:
        Lista cu nume de clase (str) sau indici (ca str dacă nu sunt nume)
    """
    labels = set()
    
    # 1. Încarcă din pickled data dacă există
    train_pkl = os.path.join(input_dir, 'train_data.pkl')
    test_pkl = os.path.join(input_dir, 'test_data.pkl')
    
    if os.path.exists(train_pkl):
        with open(train_pkl, 'rb') as f:
            _, train_labels = pickle.load(f)
        labels.update(train_labels.flatten() if hasattr(train_labels, 'flatten') else train_labels)
    
    if os.path.exists(test_pkl):
        with open(test_pkl, 'rb') as f:
            _, test_labels = pickle.load(f)
        labels.update(test_labels.flatten() if hasattr(test_labels, 'flatten') else test_labels)
    
    # 2. Dacă nu sunt labels din pickle, încearcă din structură de directoare
    if not labels:
        class_names = set()
        for subset in ['train', 'test']:
            subset_dir = os.path.join(input_dir, subset)
            if os.path.exists(subset_dir):
                class_names.update([d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))])
        labels = class_names
    
    # 3. Suprascrie cu metadata dacă există
    metadata_path = os.path.join(input_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if 'class_names' in metadata:
            return sorted(metadata['class_names'])
        elif 'num_classes' in metadata:
            return [str(i) for i in range(metadata['num_classes'])]
    
    # Return sorted unique labels as str
    return sorted([str(label) for label in labels])


def apply_poisoning(test_file,nn_name, input_dir, output_dir, operation='noise', intensity=0.1, percentage=0.2):
    """Aplică poisoning pe un procent din imagini."""
    print(f"Poisoning {input_dir} with {operation} (intensity={intensity}, percentage={percentage})")
    
    # Copiază structura originală
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
    
    # Încarcă metadata pentru clase
    # metadata_path = os.path.join(input_dir, 'metadata.json')
    # if os.path.exists(metadata_path):
    #     with open(metadata_path, 'r') as f:
    #         metadata = json.load(f)
    #     class_names = metadata['class_names']
    # else:
    #     class_names = [d for d in os.listdir(os.path.join(input_dir, 'train')) if os.path.isdir(os.path.join(input_dir, 'train', d))]
    class_names = extract_labels(input_dir)
    
    # Iterează peste train/ și test/
    for subset in ['train', 'test']:
        subset_dir = os.path.join(output_dir, subset)
        for class_name in class_names:
            class_dir = os.path.join(subset_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png')]
            
            # Selectează un procent de imagini pentru poisoning
            num_poison = int(len(images) * percentage)
            poison_images = random.sample(images, num_poison)
            
            for img_file in poison_images:
                img_path = os.path.join(class_dir, img_file)
                image = Image.open(img_path)
                
                if operation == 'noise':
                    image = add_gaussian_noise(image, std_dev=intensity)
                elif operation == 'label_flip':
                    new_class = label_flip(class_names, class_name)
                    new_path = os.path.join(subset_dir, new_class, img_file)
                    shutil.move(img_path, new_path)
                    continue  # Nu salva imaginea încă, deoarece mutăm
                elif operation == 'backdoor':
                    image = insert_backdoor(image, trigger_size=int(intensity * min(image.size)))
                    # Optional: flip label pentru backdoor
                    if random.random() > 0.5:
                        new_class = label_flip(class_names, class_name)
                        new_path = os.path.join(subset_dir, new_class, img_file)
                        shutil.move(img_path, new_path)
                        continue
                
                # Salvează imaginea poisoned
                image.save(img_path)
    
    print(f"Poisoned data saved to {output_dir}")

    write_attack_info_json(output_dir, test_file,nn_name, operation, intensity, percentage)

    print(f"Created attack info file: {test_file}")
    print(f"Poisoned data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply data poisoning to dataset directory.")
    parser.add_argument("test_file", type=str, help="Name of the test file for this simulation test")
    parser.add_argument("nn_name", type=str, help="Name of the test file for this simulation test")
    parser.add_argument("dir_name", type=str, help="Name of the data directory (e.g., data_resnet50_tf_flowers)")
    parser.add_argument("--operation", type=str, default="noise", choices=["noise", "label_flip", "backdoor"], help="Poisoning operation")
    parser.add_argument("--intensity", type=float, default=0.1, help="Intensity of the operation (e.g., std_dev for noise)")
    parser.add_argument("--percentage", type=float, default=0.2, help="Percentage of images to poison (0-1)")
    args = parser.parse_args()
    
    input_dir = Path(args.dir_name)
    output_dir = input_dir.parent / f"{input_dir.name}_poisoned"
    
    apply_poisoning(args.test_file,args.nn_name,  str(input_dir), str(output_dir), args.operation, args.intensity, args.percentage)
