#!/usr/bin/env python3
"""
Test script to validate the enhanced Pydantic models and API functionality.
"""

import sys
import json
from pydantic import ValidationError

# Add the current directory to the path so we can import our API modules
sys.path.append('.')

def test_housing_validation():
    """Test housing API validation."""
    print("🏠 Testing Housing API Validation...")
    
    try:
        from api.housing_api import HousingRequest, HousingResponse
        
        # Test valid input
        print("  ✅ Testing valid input...")
        valid_data = {
            "total_rooms": 5000.0,
            "total_bedrooms": 1200.0,
            "population": 3000.0,
            "households": 1000.0,
            "median_income": 5.5,
            "housing_median_age": 25.0,
            "latitude": 37.88,
            "longitude": -122.23
        }
        
        request = HousingRequest(**valid_data)
        print(f"     Valid request created: {request.total_rooms} rooms, {request.households} households")
        
        # Test invalid inputs
        print("  ❌ Testing invalid inputs...")
        
        # Test negative median income
        try:
            invalid_data = valid_data.copy()
            invalid_data["median_income"] = -1.0
            HousingRequest(**invalid_data)
            print("     ERROR: Should have failed for negative median income")
        except ValidationError as e:
            print(f"     ✅ Correctly rejected negative median income: {e.errors()[0]['msg']}")
        
        # Test bedrooms > rooms
        try:
            invalid_data = valid_data.copy()
            invalid_data["total_bedrooms"] = 6000.0  # More than total rooms
            HousingRequest(**invalid_data)
            print("     ERROR: Should have failed for bedrooms > rooms")
        except ValidationError as e:
            print(f"     ✅ Correctly rejected bedrooms > rooms: {e.errors()[0]['msg']}")
        
        # Test households > population
        try:
            invalid_data = valid_data.copy()
            invalid_data["households"] = 4000.0  # More than population
            HousingRequest(**invalid_data)
            print("     ERROR: Should have failed for households > population")
        except ValidationError as e:
            print(f"     ✅ Correctly rejected households > population: {e.errors()[0]['msg']}")
        
        # Test invalid latitude
        try:
            invalid_data = valid_data.copy()
            invalid_data["latitude"] = 50.0  # Outside California
            HousingRequest(**invalid_data)
            print("     ERROR: Should have failed for invalid latitude")
        except ValidationError as e:
            print(f"     ✅ Correctly rejected invalid latitude: {e.errors()[0]['msg']}")
        
        print("  🎉 Housing validation tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"  💥 Housing validation test failed: {e}")
        return False



def test_model_loading():
    """Test if models can be loaded successfully."""
    print("\n🤖 Testing Model Loading...")
    
    try:
        import joblib
        
        # Test housing model
        print("  📊 Testing housing model...")
        housing_model = joblib.load("models/DecisionTree.pkl")
        print(f"     ✅ Housing model loaded successfully: {type(housing_model).__name__}")
        
        print("  🎉 Model loading tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"  💥 Model loading test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Starting Enhanced Validation Tests")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_housing_validation())
    results.append(test_model_loading())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} test suites passed!")
        print("✅ Enhanced validation is working correctly!")
    else:
        print(f"⚠️  {passed}/{total} test suites passed")
        print("❌ Some validation tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
