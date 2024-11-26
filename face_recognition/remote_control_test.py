import requests
import time
from datetime import datetime

def test_hide_modules():
    """Test basic hide functionality for MagicMirror modules using Remote Control API"""
    BASE_URL = "http://localhost:8080"
    # List of modules from your config
    MODULES = ['weather', 'calendar', 'newsfeed', 'alert']
    
    def log_test(message):
        """Print timestamped test log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def test_module_hide(module_name):
        """Test hiding a specific module"""
        endpoint = f"{BASE_URL}/api/module/{module_name}/hide"
        log_test(f"Testing hide for module: {module_name}")
        log_test(f"Sending request to: {endpoint}")
        
        try:
            response = requests.get(endpoint)
            log_test(f"Status Code: {response.status_code}")
            try:
                result = response.json()
                log_test(f"Response: {result}")
            except:
                log_test(f"Raw Response: {response.text[:200]}")
            
            # Wait a bit to see the effect
            time.sleep(10)
            
            # Try to show the module again
            show_endpoint = f"{BASE_URL}/api/module/{module_name}/show"
            show_response = requests.get(show_endpoint)
            log_test(f"Showing module back: {show_response.status_code}")
            
            return response.status_code == 200
            
        except requests.exceptions.RequestException as e:
            log_test(f"Error testing module {module_name}: {str(e)}")
            return False
    
    # Start testing
    log_test("Starting MagicMirror Remote Control hide module test")
    log_test(f"Testing against URL: {BASE_URL}")
    print("-" * 50)
    
    results = {
        "success": 0,
        "failed": 0,
        "total": len(MODULES)
    }
    
    # Test each module
    for module in MODULES:
        if test_module_hide(module):
            results["success"] += 1
        else:
            results["failed"] += 1
        print("-" * 50)
    
    # Print summary
    print("\nTest Summary")
    print("=" * 50)
    print(f"Total Modules Tested: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['success']/results['total']*100):.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    test_hide_modules()
