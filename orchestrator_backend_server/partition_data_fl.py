"""
Script de partiționare Non-IID a datelor pentru Federated Learning.

Distribuie datele de antrenare între N clienți folosind strategia:
  - 20% date distribuite uniform (shared pool)
  - 80% date din clasa dominantă per client (disjoint pool)

Utilizare:
  python partition_data_fl.py --data_dir /path/to/clean_data --num_clients 10
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def partition_data(data_dir: Path, num_clients: int, seed: int = 42):
    """
    Partiționează datele din data_dir/train în N directoare client_0..client_{N-1},
    fiecare cu un subdirector train/ și un subdirector test/ (copie completă).
    """
    random.seed(seed)

    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    # Detectează clasele din structura de directoare
    class_dirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)
    print(f'Detected {num_classes} classes: {class_dirs}')

    # Colectează imaginile per clasă
    class_images = {}
    for class_name in class_dirs:
        class_dir = train_dir / class_name
        imgs = sorted([f.name for f in class_dir.glob('*') if f.is_file()])
        random.shuffle(imgs)
        class_images[class_name] = imgs

    # Creează directoare per client
    for i in range(num_clients):
        client_dir = data_dir / f'client_{i}'
        client_train = client_dir / 'train'
        client_test = client_dir / 'test'

        if client_test.exists():
            shutil.rmtree(client_test)
        if client_dir.exists():
            shutil.rmtree(client_dir)
        shutil.copytree(str(test_dir), str(client_test))

        for class_name in class_dirs:
            (client_train / class_name).mkdir(parents=True, exist_ok=True)

    # Setări: 20% shared, 80% disjoint (conform Secțiunea 6.3)
    shared_ratio = 0.2
    disjoint_ratio = 0.8

    # Asociere client → clasă dominantă
    client_dominant_class = {i: class_dirs[i % num_classes] for i in range(num_clients)}

    # Împarte imaginile în pool-uri disjoint și shared
    disjoint_pools = {}
    shared_pools = {}

    for c in class_dirs:
        imgs = class_images[c]
        split_idx = int(len(imgs) * disjoint_ratio)
        disjoint_pools[c] = imgs[:split_idx]
        shared_pools[c] = imgs[split_idx:]

    client_allocations = defaultdict(list)

    # Alocă datele disjoint (dominante)
    for c in class_dirs:
        clients_with_c_dom = [i for i, dom_c in client_dominant_class.items() if dom_c == c]

        if not clients_with_c_dom:
            shared_pools[c].extend(disjoint_pools[c])
            continue

        chunk_size = len(disjoint_pools[c]) // len(clients_with_c_dom)
        for idx, client_id in enumerate(clients_with_c_dom):
            start = idx * chunk_size
            end = start + chunk_size if idx < len(clients_with_c_dom) - 1 else len(disjoint_pools[c])
            client_allocations[client_id].extend([(c, img) for img in disjoint_pools[c][start:end]])

    # Alocă datele shared (uniforme)
    for c in class_dirs:
        chunk_size = len(shared_pools[c]) // num_clients
        for client_id in range(num_clients):
            start = client_id * chunk_size
            end = start + chunk_size if client_id < num_clients - 1 else len(shared_pools[c])
            client_allocations[client_id].extend([(c, img) for img in shared_pools[c][start:end]])

    # Copierea efectivă a fișierelor pe disc
    print(f'Partitioning data into {num_clients} clients...')
    for client_id, allocations in client_allocations.items():
        client_train = data_dir / f'client_{client_id}' / 'train'
        for c, img_name in allocations:
            src = train_dir / c / img_name
            dst = client_train / c / img_name
            shutil.copy2(str(src), str(dst))

    for i in range(num_clients):
        client_train = data_dir / f'client_{i}' / 'train'
        dominant_class = client_dominant_class[i]
        per_class = {
            c: len(list((client_train / c).glob('*'))) if (client_train / c).exists() else 0
            for c in class_dirs
        }
        total = sum(per_class.values())
        dom_pct = per_class[dominant_class] / total * 100 if total > 0 else 0
        print(f'  Client {i}: {total} imgs, dominant={dominant_class} ({dom_pct:.0f}%)')


def main():
    parser = argparse.ArgumentParser(description='Partition data for Federated Learning (Non-IID)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to clean_data directory')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of FL clients (N)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    partition_data(Path(args.data_dir), args.num_clients, args.seed)


if __name__ == '__main__':
    main()
