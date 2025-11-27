import argparse
import sys
import importlib.util
import json
from pathlib import Path

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

def detect_framework(template_module):
    if hasattr(template_module, 'tf') or 'tensorflow' in dir(template_module) or TENSORFLOW_AVAILABLE:
        return 'tensorflow'
    elif hasattr(template_module, 'torch') or 'nn' in dir(template_module) or PYTORCH_AVAILABLE:
        return 'pytorch'
    raise ValueError("Could not detect framework from template")

def main():
    parser = argparse.ArgumentParser(description="Train any model using template functions (framework-agnostic)")
    parser.add_argument('template_path', type=str, help="Path to template_code.py")
    parser.add_argument('output_model_path', type=str, help="Path to save the trained model")
    args = parser.parse_args()

    # Load template as module
    spec = importlib.util.spec_from_file_location("template", args.template_path)
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)

    # Detect framework dynamically from template
    framework = detect_framework(template)
    print(f"Detected framework: {framework}")

    # Load data (common to both)
    train_dataset, test_dataset = template.load_train_test_data()

    # Preprocess if function exists
    if hasattr(template, 'preprocess_loaded_data'):
        train_dataset, test_dataset = template.preprocess_loaded_data(train_dataset, test_dataset)
    elif hasattr(template, 'preprocess_transform'):
        # For PyTorch, apply transforms if needed (assume DataLoader handles it)
        pass  # Skip or implement if necessary

    # Create model using template's creation logic
    # Assume template has a 'create_model' function; if not, fallback to example or raise error
    if hasattr(template, 'create_model'):
        model = template.create_model()
    else:
        # Fallback: Use example from template __main__, but make it agnostic
        raise ValueError("Template must define 'create_model()' for custom architectures")

    # Compile model if function exists
    if hasattr(template, '_model_compile'):
        model = template._model_compile(model)

    # Train the model (agnostic call)
    history = template.train_neural_network(
        model=model,
        train_dataset=train_dataset if framework == 'tensorflow' else train_dataset,  # Adjust if PyTorch uses DataLoader
        validation_dataset=test_dataset if framework == 'tensorflow' else test_dataset,
        epochs=2,  # Can be parameterized
        verbose=1
    )

    # Save the model
    template.save_model_config(model, args.output_model_path)

if __name__ == "__main__":
    main()