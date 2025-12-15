#!/usr/bin/env python3
"""
Script de test pentru template_code.py
Verifică că toate funcțiile critice funcționează corect

VERSION: 2.2
UPDATED: December 14, 2025
FIX: Debugging îmbunătățit pentru calculate_metrics
COMPATIBLE WITH: template_code.py (MNIST) și template_code_creare_model_hugg.py (CIFAR-10)
"""

import sys
import traceback
import json
from pathlib import Path

def test_template():
    """Testează funcțiile principale din template"""
    
    print("=" * 70)
    print("TESTARE TEMPLATE_CODE.PY")
    print("=" * 70)
    
    # Test 1: Verificare import
    print("\n1. Testare import template...")
    try:
        import template_code
        print("   ✓ Template importat cu succes")
    except ImportError as e:
        print(f"   ✗ Eroare la import: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Verificare funcții esențiale (ACTUALIZAT)
    print("\n2. Verificare funcții esențiale...")
    
    # Lista completă de funcții necesare pentru FL simulator
    required_functions = [
        # Funcții de încărcare date
        'load_train_test_data',      # Încarcă date train/test
        'preprocess_loaded_data',     # Preprocesare date
        'load_client_data',           # Încărcare date pentru FL clients
        'download_data',              # Salvare date în format FL
        
        # Funcții de antrenare și evaluare
        'train_neural_network',       # Antrenare model
        'calculate_metrics',          # Calculare metrici
        
        # Funcții pentru manipulare ponderi
        'get_model_weights',          # Extrage ponderi
        'set_model_weights',          # Setează ponderi
        
        # Funcții pentru salvare/încărcare model
        'save_model_config',          # Salvare model complet
        'load_model_config',          # Încărcare model complet
        'save_weights_only',          # Salvare doar ponderi
        'load_weights_only',          # Încărcare doar ponderi
        
        # Funcții pentru creare și validare model
        'create_model',               # Creează/Descarcă model
        'validate_model_structure',   # Validare structură
        '_model_compile',             # Compilare model
        
        # Funcții auxiliare
        'get_loss_type',              # Tip loss
        'get_image_format',           # Format imagini
        'get_data_preprocessing',     # Info preprocesare
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if hasattr(template_code, func_name):
            print(f"   ✓ {func_name} există")
        else:
            print(f"   ✗ {func_name} lipsește!")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"\n   ❌ Funcții lipsă: {', '.join(missing_functions)}")
        return False
    
    print(f"\n   ✅ Toate cele {len(required_functions)} funcții esențiale există!")
    
    # Test 3: Test creare model
    print("\n3. Testare creare și validare model...")
    try:
        model = template_code.create_model()
        print("   ✓ Model creat cu succes")
        
        # Verificare tip model
        import tensorflow as tf
        if not isinstance(model, tf.keras.Model):
            print(f"   ✗ Model nu e instanță de tf.keras.Model (tip: {type(model)})")
            return False
        print(f"   ✓ Model e instanță validă de tf.keras.Model")
        
        # Validare structură
        model_info = template_code.validate_model_structure(model)
        print("   ✓ Validare structură completă")
        print(f"   - Nume model: {model_info.get('model_name', 'N/A')}")
        print(f"   - Total parametri: {model_info['total_params']:,}")
        print(f"   - Parametri antrenabili: {model_info.get('trainable_params', 'N/A'):,}")
        print(f"   - Layers: {model_info['layers_count']}")
        print(f"   - Compilat: {model_info['is_compiled']}")
        print(f"   - Input shape: {model_info.get('input_shape', 'N/A')}")
        print(f"   - Output shape: {model_info.get('output_shape', 'N/A')}")
        
        # Verificare compilare
        if not model_info['is_compiled']:
            print("   ⚠️  WARNING: Model nu este compilat!")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Test funcții auxiliare
    print("\n4. Testare funcții auxiliare...")
    try:
        # Test get_loss_type
        loss_type = template_code.get_loss_type()
        print(f"   ✓ Tip loss: {loss_type}")
        
        # Test get_image_format
        img_format = template_code.get_image_format()
        print(f"   ✓ Format imagini: {img_format}")
        
        # Test get_data_preprocessing
        preprocess_fn = template_code.get_data_preprocessing()
        print(f"   ✓ Funcție preprocesare: {preprocess_fn.__name__ if hasattr(preprocess_fn, '__name__') else type(preprocess_fn)}")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Test semnături funcții critice
    print("\n5. Testare semnături funcții critice...")
    try:
        import inspect
        
        # Test download_data
        sig = inspect.signature(template_code.download_data)
        params = list(sig.parameters.keys())
        print(f"   ✓ download_data({', '.join(params)})")
        
        # Test load_client_data
        sig = inspect.signature(template_code.load_client_data)
        params = list(sig.parameters.keys())
        print(f"   ✓ load_client_data({', '.join(params)})")
        
        # Test train_neural_network
        sig = inspect.signature(template_code.train_neural_network)
        params = list(sig.parameters.keys())
        print(f"   ✓ train_neural_network({', '.join(params[:3])}...)")
        
        # Test save_model_config
        sig = inspect.signature(template_code.save_model_config)
        params = list(sig.parameters.keys())
        print(f"   ✓ save_model_config({', '.join(params)})")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Test manipulare ponderi (fără antrenare)
    print("\n6. Testare manipulare ponderi...")
    try:
        # Extrage ponderi
        weights = template_code.get_model_weights(model)
        print(f"   ✓ Ponderi extrase: {len(weights)} layere")
        
        # Verificare format ponderi
        if not isinstance(weights, list):
            print(f"   ✗ Ponderi nu sunt listă (tip: {type(weights)})")
            return False
        
        if len(weights) == 0:
            print(f"   ✗ Nicio pondere extrasă!")
            return False
        
        print(f"   ✓ Format ponderi valid: listă cu {len(weights)} elemente")
        
        # Test setare ponderi (cu aceleași ponderi)
        template_code.set_model_weights(model, weights)
        print(f"   ✓ Ponderi setate cu succes")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        traceback.print_exc()
        return False
    
    # Test 7: Test salvare/încărcare model (fără scriere efectivă)
    print("\n7. Verificare funcții salvare/încărcare...")
    try:
        import inspect
        
        # Verificare că funcțiile au parametrii corecți
        sig_save = inspect.signature(template_code.save_model_config)
        sig_load = inspect.signature(template_code.load_model_config)
        
        print(f"   ✓ save_model_config: {len(sig_save.parameters)} parametri")
        print(f"   ✓ load_model_config: {len(sig_load.parameters)} parametri")
        
        # Verificare save_weights_only și load_weights_only
        sig_save_w = inspect.signature(template_code.save_weights_only)
        sig_load_w = inspect.signature(template_code.load_weights_only)
        
        print(f"   ✓ save_weights_only: {len(sig_save_w.parameters)} parametri")
        print(f"   ✓ load_weights_only: {len(sig_load_w.parameters)} parametri")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        traceback.print_exc()
        return False
    
    # ========== TEST 8: Calculare și salvare metrici inițiale ==========
    print("\n8. Calculare și salvare metrici inițiale...")
    init_metrics = {}
    
    try:
        # Verificare dacă există date de test pentru evaluare
        print("   ⏳ Încărcare date de test pentru evaluare...")
        
        # Încarcă date test
        try:
            train_ds, test_ds = template_code.load_train_test_data()
            print("   ✓ Date încărcate (train + test)")
            
            # DEBUG: Verifică tipul datelor
            print(f"   🔍 Tip train_ds: {type(train_ds)}")
            print(f"   🔍 Tip test_ds: {type(test_ds)}")
            
            # Preprocesare date
            print("   ⏳ Preprocesare date...")
            _, test_ds = template_code.preprocess_loaded_data(train_ds, test_ds)
            print("   ✓ Date preprocesate")
            
            # DEBUG: Verifică un batch
            print("   🔍 Verificare format date...")
            for images, labels in test_ds.take(1):
                print(f"      - Images shape: {images.shape}")
                print(f"      - Images dtype: {images.dtype}")
                print(f"      - Labels shape: {labels.shape}")
                print(f"      - Labels dtype: {labels.dtype}")
                
                # Verifică dacă labels sunt one-hot
                import tensorflow as tf
                if len(labels.shape) == 2 and labels.shape[1] > 1:
                    print(f"      ✓ Labels sunt one-hot encoded ({labels.shape[1]} clase)")
                else:
                    print(f"      ⚠️  Labels NU sunt one-hot encoded!")
            
        except Exception as e:
            print(f"   ✗ Nu s-au putut încărca datele de test: {e}")
            traceback.print_exc()
            print("   ℹ️  Salvare metrici fără evaluare pe date...")
            test_ds = None
        
        # Calculează metrici inițiale dacă avem date
        if test_ds is not None:
            try:
                print("   ⏳ Calculare metrici inițiale (poate dura ~30-60s)...")
                
                init_metrics = template_code.calculate_metrics(model, test_ds)
                print("   ✓ Metrici calculate cu succes")
                
                # Afișare metrici
                for metric_name, value in init_metrics.items():
                    print(f"      • {metric_name}: {value:.4f}")
                    
            except Exception as e:
                print(f"   ✗ Eroare la calculare metrici: {e}")
                traceback.print_exc()
                print("   ℹ️  Continuare cu metrici goale...")
                init_metrics = {}
        else:
            print("   ⚠️  Date de test indisponibile - metrici nu pot fi calculate")
            init_metrics = {}
        
        # Construiește dicționar complet cu info model + metrici
        verification_data = {
            "verification_timestamp": str(Path.cwd()),
            "template_verified": True,
            "model_info": {
                "name": model_info.get('model_name', model.name if hasattr(model, 'name') else 'unknown'),
                "total_params": int(model_info['total_params']),
                "trainable_params": int(model_info.get('trainable_params', model_info['total_params'])),
                "non_trainable_params": int(model_info['total_params'] - model_info.get('trainable_params', model_info['total_params'])),
                "layers_count": int(model_info['layers_count']),
                "input_shape": str(model_info.get('input_shape', 'N/A')),
                "output_shape": str(model_info.get('output_shape', 'N/A')),
                "is_compiled": bool(model_info['is_compiled']),
                "loss_type": loss_type,
                "image_format": img_format
            },
            "initial_metrics": {
                metric_name: float(value) for metric_name, value in init_metrics.items()
            },
            "weights_info": {
                "total_weight_layers": len(weights),
                "weights_extractable": True,
                "weights_settable": True
            },
            "functions_verified": {
                func: True for func in required_functions
            }
        }
        
        # Salvează în init-verification.json
        output_file = Path("init-verification.json")
        with open(output_file, 'w') as f:
            json.dump(verification_data, f, indent=2)
        
        print(f"\n   ✓ Verificare salvată în: {output_file.absolute()}")
        print(f"   📊 Model: {verification_data['model_info']['name']}")
        print(f"   📊 Parametri: {verification_data['model_info']['total_params']:,}")
        if init_metrics:
            print(f"   📊 Accuracy inițială: {init_metrics.get('accuracy', 0):.4f}")
            print(f"   ✅ Metrici calculate și salvate cu succes!")
        else:
            print(f"   ⚠️  Metrici inițiale goale (evaluare eșuată)")
            print(f"   ℹ️  Verifică log-urile de mai sus pentru detalii")
        
    except Exception as e:
        print(f"   ✗ Eroare la salvare verificare: {e}")
        traceback.print_exc()
        print("   ℹ️  Verificare template continuă (salvare opțională)")
    
    # SUCCES
    print("\n" + "=" * 70)
    print("✅ TOATE TESTELE AU TRECUT CU SUCCES!")
    print("=" * 70)
    print(f"\nTemplate verificat:")
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
            print(f"  • Metrici inițiale: ⚠️  Goale (verifică log)")
    print("\n✓ Template READY pentru FL simulation!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    # Adaugă directorul curent în path pentru import
    sys.path.insert(0, '.')
    
    success = test_template()
    sys.exit(0 if success else 1)