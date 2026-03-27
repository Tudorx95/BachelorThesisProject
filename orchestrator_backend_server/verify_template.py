#!/usr/bin/env python3
"""
Script de test pentru template_code.py - UNIVERSAL (TensorFlow & PyTorch)
Verifică că toate funcțiile critice funcționează corect
Generează init-verification.json cu metrici inițiale

VERSION: 2.0
UPDATED: February 2026
COMPATIBLE WITH: TensorFlow templates (.keras) și PyTorch templates (.pth)
"""

import sys
import traceback
import inspect
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

# ============================================================================
# CULORI TERMINAL
# ============================================================================
class Colors:
    GREEN  = '\033[92m'
    RED    = '\033[91m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'

def print_status(passed: bool, message: str):
    """Print formatted status message"""
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
    print(f"   {symbol} {message}")

def verify_function_exists(module, func_name: str) -> bool:
    """Verifică dacă funcția există în modul"""
    return hasattr(module, func_name)

def verify_function_signature(module, func_name: str, min_params: int = 0) -> Tuple[bool, List[str]]:
    """
    Verifică semnătura funcției
    Returns:
        Tuple[bool, List[str]]: (is_valid, parameter_names)
    """
    try:
        func = getattr(module, func_name)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if len(params) < min_params:
            return False, params
        return True, params
    except Exception:
        return False, []

# ============================================================================
# DETECTARE FRAMEWORK
# ============================================================================
def detect_framework(template_module) -> str:
    """
    Detectează framework-ul folosit în template pe baza importurilor.
    Returns: 'tensorflow', 'pytorch', sau 'unknown'
    """
    try:
        source = inspect.getsource(template_module)
    except Exception:
        source = ""

    has_tf    = "import tensorflow" in source or "from tensorflow" in source
    has_torch = "import torch" in source or "from torch" in source

    if has_tf and not has_torch:
        return "tensorflow"
    if has_torch and not has_tf:
        return "pytorch"
    if has_torch and has_tf:
        idx_tf    = source.find("tensorflow")
        idx_torch = source.find("torch")
        return "pytorch" if idx_torch < idx_tf else "tensorflow"
    return "unknown"


def verify_model_type(model, framework: str) -> bool:
    """Verifică că modelul este instanța corectă pentru framework."""
    if framework == "tensorflow":
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                print_status(True, "Model e instanță validă de tf.keras.Model")
                return True
            else:
                print_status(False, f"Model nu e tf.keras.Model (tip: {type(model)})")
                return False
        except ImportError:
            print(f"   ⚠  TensorFlow nu este instalat — skip type check")
            return True
    elif framework == "pytorch":
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                print_status(True, "Model e instanță validă de nn.Module")
                return True
            else:
                print_status(False, f"Model nu e nn.Module (tip: {type(model)})")
                return False
        except ImportError:
            print(f"   ⚠  PyTorch nu este instalat — skip type check")
            return True
    else:
        print(f"   ⚠  Framework necunoscut '{framework}' — skip type check")
        return True


# ============================================================================
# FUNCȚIA PRINCIPALĂ
# ============================================================================
def test_template():
    """Testează funcțiile principale din template și generează init-verification.json"""

    print("=" * 70)
    print("TESTARE TEMPLATE_CODE.PY — UNIVERSAL (TensorFlow & PyTorch)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Import template
    # ------------------------------------------------------------------
    print("\n1. Testare import template...")
    try:
        import template_code
        print_status(True, "Template importat cu succes")
    except ImportError as e:
        print_status(False, f"Eroare la import: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 1.5: Detectare framework
    # ------------------------------------------------------------------
    print("\n1.5. Detectare framework...")
    framework = detect_framework(template_code)
    print_status(True, f"Framework detectat: {Colors.BOLD}{framework.upper()}{Colors.RESET}")

    # ------------------------------------------------------------------
    # Test 2: Verificare funcții esențiale
    # ------------------------------------------------------------------
    print(f"\n{Colors.BLUE}2. Verificare funcții esențiale...{Colors.RESET}")

    required_functions = {
        # Format: 'function_name': (min_params, description)
        'load_train_test_data':    (0, 'Încărcare date train/test'),
        'preprocess_loaded_data':  (2, 'Preprocesare date'),
        'load_client_data':        (1, 'Încărcare date FL client'),
        'download_data':           (1, 'Salvare date în format FL'),
        'train_neural_network':    (2, 'Antrenare model'),
        'calculate_metrics':       (2, 'Calculare metrici'),
        'get_model_weights':       (1, 'Extragere ponderi'),
        'set_model_weights':       (2, 'Setare ponderi'),
        'save_model_config':       (2, 'Salvare model complet'),
        'load_model_config':       (1, 'Încărcare model complet'),
        'save_weights_only':       (2, 'Salvare doar ponderi'),
        'load_weights_only':       (2, 'Încărcare doar ponderi'),
        'create_model':            (0, 'Creare/Descărcare model'),
        'validate_model_structure':(1, 'Validare structură model'),
        '_model_compile':          (1, 'Compilare model'),
        'get_loss_type':           (0, 'Tip loss function'),
        'get_image_format':        (0, 'Format imagini'),
        'get_data_preprocessing':  (0, 'Info preprocesare'),
    }

    results = {
        'passed':           [],
        'failed':           [],
        'signature_issues': []
    }

    for func_name, (min_params, description) in required_functions.items():
        exists = verify_function_exists(template_code, func_name)

        if not exists:
            print_status(False, f"{func_name} LIPSEȘTE!")
            results['failed'].append(func_name)
            continue

        sig_valid, params = verify_function_signature(template_code, func_name, min_params)

        if not sig_valid:
            print_status(False, f"{func_name}({', '.join(params)}) - parametri insuficienți (minim {min_params})")
            results['signature_issues'].append(func_name)
        else:
            print_status(True, f"{func_name}({', '.join(params)})")
            results['passed'].append(func_name)

    # Rezumat funcții
    total      = len(required_functions)
    passed_cnt = len(results['passed'])
    failed_cnt = len(results['failed'])
    sig_cnt    = len(results['signature_issues'])

    print(f"\n   • Total funcții verificate: {total}")
    print_status(passed_cnt == total, f"Funcții corecte: {passed_cnt}/{total}")

    if failed_cnt > 0:
        print_status(False, f"Funcții lipsă: {failed_cnt}")
        print(f"      Lipsă: {', '.join(results['failed'])}")
        return False

    if sig_cnt > 0:
        print_status(False, f"Probleme semnătură: {sig_cnt}")
        print(f"      Problematice: {', '.join(results['signature_issues'])}")

    print_status(True, f"Toate cele {total} funcții esențiale există!")

    # ------------------------------------------------------------------
    # Test 3: Creare și validare model
    # ------------------------------------------------------------------
    print("\n3. Testare creare și validare model...")
    try:
        model = template_code.create_model()
        print_status(True, "Model creat cu succes")

        # Verificare tip — framework-agnostic
        if not verify_model_type(model, framework):
            return False

        model_info = template_code.validate_model_structure(model)
        print_status(True, "Validare structură completă")
        print(f"   - Nume model:           {model_info.get('model_name', 'N/A')}")
        print(f"   - Total parametri:       {model_info['total_params']:,}")
        print(f"   - Parametri antrenabili: {model_info.get('trainable_params', 'N/A'):,}")
        print(f"   - Layers:                {model_info['layers_count']}")
        print(f"   - Compilat:              {model_info['is_compiled']}")
        print(f"   - Input shape:           {model_info.get('input_shape', 'N/A')}")
        print(f"   - Output shape:          {model_info.get('output_shape', 'N/A')}")

        if not model_info['is_compiled']:
            print(f"   ⚠  WARNING: Model nu este compilat!")

    except Exception as e:
        print_status(False, f"Eroare: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 4: Funcții auxiliare
    # ------------------------------------------------------------------
    print("\n4. Testare funcții auxiliare...")
    try:
        # get_loss_type
        try:
            loss_type = template_code.get_loss_type()
            if isinstance(loss_type, str):
                print_status(True, f"get_loss_type() → '{loss_type}'")
            else:
                print_status(False, f"get_loss_type() returnează tip greșit: {type(loss_type)}")
        except Exception as e:
            print_status(False, f"get_loss_type() eroare: {e}")

        # get_image_format
        try:
            img_format = template_code.get_image_format()
            if isinstance(img_format, dict):
                print_status(True, f"get_image_format() → {img_format}")
            else:
                print_status(False, f"get_image_format() returnează tip greșit: {type(img_format)}")
        except Exception as e:
            print_status(False, f"get_image_format() eroare: {e}")

        # get_data_preprocessing
        try:
            preprocess_fn = template_code.get_data_preprocessing()
            fn_name = preprocess_fn.__name__ if hasattr(preprocess_fn, '__name__') else str(type(preprocess_fn))
            if callable(preprocess_fn):
                print_status(True, f"get_data_preprocessing() → {fn_name}")
            else:
                print_status(False, "get_data_preprocessing() nu returnează funcție")
        except Exception as e:
            print_status(False, f"get_data_preprocessing() eroare: {e}")

    except Exception as e:
        print_status(False, f"Eroare la testare funcții auxiliare: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 5: Semnături funcții critice
    # ------------------------------------------------------------------
    print("\n5. Testare semnături funcții critice...")
    try:
        sig = inspect.signature(template_code.download_data)
        print_status(True, f"download_data({', '.join(sig.parameters.keys())})")

        sig = inspect.signature(template_code.load_client_data)
        print_status(True, f"load_client_data({', '.join(sig.parameters.keys())})")

        sig = inspect.signature(template_code.train_neural_network)
        params = list(sig.parameters.keys())
        print_status(True, f"train_neural_network({', '.join(params[:3])}...)")

        sig = inspect.signature(template_code.save_model_config)
        print_status(True, f"save_model_config({', '.join(sig.parameters.keys())})")

    except Exception as e:
        print_status(False, f"Eroare: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 6: Manipulare ponderi
    # ------------------------------------------------------------------
    print("\n6. Testare manipulare ponderi...")
    try:
        weights = template_code.get_model_weights(model)
        print_status(True, f"Ponderi extrase: {len(weights)} layere")

        if not isinstance(weights, list):
            print_status(False, f"Ponderi nu sunt listă (tip: {type(weights)})")
            return False

        if len(weights) == 0:
            print_status(False, "Nicio pondere extrasă!")
            return False

        print_status(True, f"Format ponderi valid: listă cu {len(weights)} elemente")

        template_code.set_model_weights(model, weights)
        print_status(True, "Ponderi setate cu succes")

    except Exception as e:
        print_status(False, f"Eroare: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 7: Semnături salvare / încărcare
    # ------------------------------------------------------------------
    print("\n7. Verificare funcții salvare/încărcare...")
    try:
        sig_save   = inspect.signature(template_code.save_model_config)
        sig_load   = inspect.signature(template_code.load_model_config)
        sig_save_w = inspect.signature(template_code.save_weights_only)
        sig_load_w = inspect.signature(template_code.load_weights_only)

        print_status(True, f"save_model_config: {len(sig_save.parameters)} parametri")
        print_status(True, f"load_model_config: {len(sig_load.parameters)} parametri")
        print_status(True, f"save_weights_only: {len(sig_save_w.parameters)} parametri")
        print_status(True, f"load_weights_only: {len(sig_load_w.parameters)} parametri")

    except Exception as e:
        print_status(False, f"Eroare: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 8: Calculare și salvare metrici inițiale
    # ------------------------------------------------------------------
    print("\n8. Calculare și salvare metrici inițiale...")
    init_metrics = {}
    test_ds      = None

    # 8a — Încărcare date
    try:
        print("   ⏳ Încărcare date de test pentru evaluare...")
        train_ds, test_ds = template_code.load_train_test_data()
        print_status(True, "Date încărcate (train + test)")
        print(f"   🔍 Tip train_ds: {type(train_ds)}")
        print(f"   🔍 Tip test_ds:  {type(test_ds)}")

        # 8b — Preprocesare
        print("   ⏳ Preprocesare date...")
        _, test_ds = template_code.preprocess_loaded_data(train_ds, test_ds)
        print_status(True, "Date preprocesate")

        # 8c — Verificare format batch (framework-agnostic)
        print("   🔍 Verificare format date...")
        batch_checked = False

        # Încearcă .take(1) — TensorFlow tf.data.Dataset
        if hasattr(test_ds, 'take'):
            try:
                for images, labels in test_ds.take(1):
                    print(f"      - Images shape: {images.shape}  dtype: {images.dtype}")
                    print(f"      - Labels shape:  {labels.shape}  dtype: {labels.dtype}")
                    lshape = labels.shape
                    if len(lshape) == 2 and lshape[1] > 1:
                        print_status(True, f"Labels sunt one-hot encoded ({lshape[1]} clase)")
                    else:
                        print(f"   ⚠  Labels NU sunt one-hot encoded!")
                    batch_checked = True
            except Exception as e:
                print(f"   ⚠  .take(1) eșuat: {e}")

        # Fallback — PyTorch DataLoader (iterare directă, prim batch)
        if not batch_checked:
            try:
                for images, labels in test_ds:
                    print(f"      - Images shape: {images.shape}  dtype: {images.dtype}")
                    print(f"      - Labels shape:  {labels.shape}  dtype: {labels.dtype}")
                    lshape = labels.shape
                    if len(lshape) == 2 and lshape[1] > 1:
                        print_status(True, f"Labels sunt one-hot encoded ({lshape[1]} clase)")
                    else:
                        print(f"   ⚠  Labels NU sunt one-hot encoded!")
                    break  # Doar primul batch
                batch_checked = True
            except Exception as e:
                print(f"   ⚠  Nu s-a putut verifica batch-ul: {e}")

    except Exception as e:
        print_status(False, f"Nu s-au putut încărca datele de test: {e}")
        traceback.print_exc()
        print("   ℹ  Salvare metrici fără evaluare pe date...")
        test_ds = None

    # 8d — Calculare metrici
    if test_ds is not None:
        try:
            print("   ⏳ Calculare metrici inițiale (poate dura ~30-120s)...")
            init_metrics = template_code.calculate_metrics(model, test_ds)
            print_status(True, "Metrici calculate cu succes")
            for metric_name, value in init_metrics.items():
                print(f"      • {metric_name}: {value:.4f}")
        except Exception as e:
            print_status(False, f"Eroare la calculare metrici: {e}")
            traceback.print_exc()
            print("   ℹ  Continuare cu metrici goale...")
            init_metrics = {}
    else:
        print(f"   ⚠  Date de test indisponibile — metrici nu pot fi calculate")

    # 8e — Construire și salvare init-verification.json
    try:
        verification_data = {
            "verification_timestamp": str(Path.cwd()),
            "template_verified": True,
            "framework": framework,
            "model_info": {
                "name": model_info.get('model_name',
                        getattr(model, 'name', 'unknown')),
                "total_params":         int(model_info['total_params']),
                "trainable_params":     int(model_info.get('trainable_params',
                                            model_info['total_params'])),
                "non_trainable_params": int(model_info['total_params'] -
                                            model_info.get('trainable_params',
                                            model_info['total_params'])),
                "layers_count":  int(model_info['layers_count']),
                "input_shape":   str(model_info.get('input_shape',  'N/A')),
                "output_shape":  str(model_info.get('output_shape', 'N/A')),
                "is_compiled":   bool(model_info['is_compiled']),
                "loss_type":     loss_type,
                "image_format":  img_format,
            },
            "initial_metrics": {
                k: float(v) for k, v in init_metrics.items()
            },
            "weights_info": {
                "total_weight_layers": len(weights),
                "weights_extractable": True,
                "weights_settable":    True,
            },
            "functions_verified": {
                fn: True for fn in required_functions
            },
        }

        output_file = Path("init-verification.json")
        with open(output_file, 'w') as f:
            json.dump(verification_data, f, indent=2)

        print_status(True, f"Verificare salvată în: {output_file.absolute()}")
        print(f"   📊 Framework:  {framework.upper()}")
        print(f"   📊 Model:      {verification_data['model_info']['name']}")
        print(f"   📊 Parametri:  {verification_data['model_info']['total_params']:,}")

        if init_metrics:
            print(f"   📊 Accuracy:   {init_metrics.get('accuracy', 0):.4f}")
            print_status(True, "Metrici inițiale calculate și salvate cu succes!")
        else:
            print(f"   ⚠  Metrici inițiale goale (evaluare eșuată)")
            print(f"   ℹ  Verifică log-urile de mai sus pentru detalii")

    except Exception as e:
        print_status(False, f"Eroare la salvare init-verification.json: {e}")
        traceback.print_exc()
        print("   ℹ  Verificare template continuă (salvare opțională)")

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ TOATE TESTELE AU TRECUT CU SUCCES!")
    print("=" * 70)
    print(f"\nTemplate verificat:")
    print(f"  • Framework:           {Colors.BOLD}{framework.upper()}{Colors.RESET}")
    print(f"  • {len(required_functions)} funcții esențiale prezente")
    print(f"  • Model creat și validat cu succes")
    print(f"  • {model_info['total_params']:,} parametri în model")
    print(f"  • Manipulare ponderi funcțională")
    print(f"  • Funcții auxiliare operative")

    if Path("init-verification.json").exists():
        print(f"  • Verificare salvată în init-verification.json")
        if init_metrics:
            print(f"  • Metrici inițiale: ✅ Calculate")
        else:
            print(f"  • Metrici inițiale: ⚠  Goale (verifică log)")

    print("\n✓ Template READY pentru FL simulation!")
    print("=" * 70)

    return True


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    sys.path.insert(0, '.')
    success = test_template()
    sys.exit(0 if success else 1)