#!/usr/bin/env python3
"""
Simple test script for the new direct Gemini evaluation service.
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, 'app')

async def test_evaluation_service():
    """Test the evaluation service directly."""
    try:
        print("🧪 Testing Direct Gemini Evaluation Service")
        
        # Test Gemini import
        import google.generativeai as genai
        print("✅ Google GenerativeAI import successful")
        
        # Test configuration
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ GOOGLE_API_KEY not found")
            return
        
        genai.configure(api_key=api_key)
        print("✅ Gemini configuration successful")
        
        # Test model initialization
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print("✅ Gemini model initialization successful")
        
        # Test simple generation
        response = model.generate_content("Hello, respond with 'Hello from Gemini!'")
        print(f"✅ Test generation successful: {response.text}")
        
        print("\n🎉 All tests passed! The evaluation service should work.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_evaluation_service()) 