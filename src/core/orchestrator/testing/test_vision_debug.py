#!/usr/bin/env python3
"""Debug vision API issues"""

import os
import base64
from pathlib import Path

# Try to load from .env, but don't fail if dotenv not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    exit(1)

# Check keys
openai_key = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY set: {bool(openai_key)}")
if openai_key:
    print(f"  Key starts with: {openai_key[:10]}...")

# Read and encode image
# Find test_image.png in the orchestrator directory
orchestrator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(orchestrator_dir, "test_image.png")
print(f"\nImage path: {image_path}")
print(f"Image exists: {Path(image_path).exists()}")
if Path(image_path).exists():
    print(f"Image size: {Path(image_path).stat().st_size} bytes")
else:
    print("ERROR: test_image.png not found in orchestrator directory")
    exit(1)

with open(image_path, 'rb') as f:
    image_data = f.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    print(f"Base64 encoded: {len(base64_image)} chars")

# Test vision API
print("\n" + "="*60)
print("Testing Vision API Call")
print("="*60)

try:
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Describe all text and diagrams."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    print("✓ API call succeeded!")
    print(f"Response: {response.choices[0].message.content[:200]}...")

except Exception as e:
    print(f"✗ API call failed: {e}")
    print(f"Error type: {type(e).__name__}")
