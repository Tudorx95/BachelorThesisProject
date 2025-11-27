#!/usr/bin/env python3
"""
Script de test pentru template_code.py
Verifică că toate funcțiile critice funcționează corect
"""

import sys
import traceback

def test_template():
    """Testează funcțiile principale din template"""
    
    print("=" * 60)
    print("TESTARE TEMPLATE_CODE.PY")
    print("=" * 60)
    
    # Test 1: Verificare import
    print("\n1. Testare import template...")
    try:
        import template_code
        print("   ✓ Template importat cu succes")
    except ImportError as e:
        print(f"   ✗ Eroare la import: {e}")
        return False
    
    # Test 2: Verificare funcții esențiale
    print("\n2. Verificare funcții esențiale...")
    required_functions = [
        'load_train_test_data',
        'download_data',
        'preprocess_loaded_data', 
        'train_model',
        'evaluate_model',
        'create_model',
        'validate_model_structure',
        'get_model_config',
        'get_loss_type',
        'get_data_preprocessing'
    ]
    
    for func_name in required_functions:
        if hasattr(template_code, func_name):
            print(f"   ✓ {func_name} există")
        else:
            print(f"   ✗ {func_name} lipsește!")
            return False
    
    # Test 3: Test creare model
    print("\n3. Testare creare și validare model...")
    try:
        model = template_code.create_model()
        print("   ✓ Model creat cu succes")
        
        # Validare structură
        model_info = template_code.validate_model_structure(model)
        print("   ✓ Validare structură completă")
        print(f"   - Total parametri: {model_info['total_params']}")
        print(f"   - Layers: {model_info['layers_count']}")
        print(f"   - Compilat: {model_info['is_compiled']}")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Test funcții auxiliare
    print("\n4. Testare funcții auxiliare...")
    try:
        loss_type = template_code.get_loss_type()
        print(f"   ✓ Tip loss: {loss_type}")
        
        config = template_code.get_model_config()
        print(f"   ✓ Configurație model obținută")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        return False
    
    # Test 5: Test download_data (doar verificare că nu dă eroare)
    print("\n5. Testare funcție download_data...")
    try:
        # Verificăm doar că funcția există și poate fi apelată
        import inspect
        sig = inspect.signature(template_code.download_data)
        params = list(sig.parameters.keys())
        print(f"   ✓ Funcția download_data acceptă parametrii: {params}")
        
    except Exception as e:
        print(f"   ✗ Eroare: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TOATE TESTELE AU TRECUT CU SUCCES! ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # Adaugă directorul curent în path pentru import
    sys.path.insert(0, '.')
    
    success = test_template()
    sys.exit(0 if success else 1)