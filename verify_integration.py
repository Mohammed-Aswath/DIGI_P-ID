"""
Quick Verification Script

This script helps verify that:
1. PaddleOCR microservice is running
2. Client can connect to microservice
3. Integration is working correctly

Run this BEFORE testing the full system.
"""

import requests
import sys

def check_microservice():
    """Check if PaddleOCR microservice is running"""
    print("=" * 60)
    print("Checking PaddleOCR Microservice...")
    print("=" * 60)
    
    try:
        response = requests.get("http://127.0.0.1:6000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Microservice is RUNNING")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Service: {data.get('service', 'unknown')}")
            return True
        else:
            print(f"❌ Microservice returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Microservice is NOT RUNNING")
        print("   Start it with: python paddle_ocr_service.py (in venv_paddle)")
        return False
    except Exception as e:
        print(f"❌ Error checking microservice: {e}")
        return False


def check_client_import():
    """Check if client can be imported"""
    print("\n" + "=" * 60)
    print("Checking PaddleOCR Client...")
    print("=" * 60)
    
    try:
        from paddle_ocr_client import call_paddle_ocr_service
        print("✅ Client module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Client import failed: {e}")
        print("   Make sure paddle_ocr_client.py is in the project directory")
        print("   Install requests: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_app_integration():
    """Check if app.py has been integrated"""
    print("\n" + "=" * 60)
    print("Checking app.py Integration...")
    print("=" * 60)
    
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for client import
        if 'from paddle_ocr_client import' in content:
            print("✅ Client import found in app.py")
            has_import = True
        else:
            print("❌ Client import NOT found in app.py")
            print("   Add: from paddle_ocr_client import call_paddle_ocr_service")
            has_import = False
        
        # Check for microservice usage
        if 'PADDLE_OCR_CLIENT_AVAILABLE' in content:
            print("✅ Client availability check found")
            has_check = True
        else:
            print("❌ Client availability check NOT found")
            has_check = False
        
        # Check for microservice call
        if 'call_paddle_ocr_service' in content:
            print("✅ Microservice call found in app.py")
            has_call = True
        else:
            print("❌ Microservice call NOT found in app.py")
            print("   Replace symbol-guided OCR section with microservice calls")
            has_call = False
        
        return has_import and has_check and has_call
        
    except FileNotFoundError:
        print("❌ app.py not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Error checking app.py: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("PADDLE OCR MICROSERVICE INTEGRATION VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Check 1: Microservice
    microservice_ok = check_microservice()
    results.append(("Microservice Running", microservice_ok))
    
    # Check 2: Client Import
    client_ok = check_client_import()
    results.append(("Client Import", client_ok))
    
    # Check 3: App Integration
    integration_ok = check_app_integration()
    results.append(("App Integration", integration_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for name, status in results:
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{status_str} - {name}")
        if not status:
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - System is ready!")
        print("   You can now start the main backend: python app.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("   Please fix the issues above before proceeding")
        print("\n   See INTEGRATION_STEPS.md for detailed instructions")
    print("=" * 60 + "\n")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
