#!/usr/bin/env python3
"""Quick MongoDB Connection Test"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

load_dotenv()

uri = os.getenv('MONGODB_URI')

print("=" * 60)
print("MONGODB CONNECTION TEST - PYTHON 3.11")
print("=" * 60)

if not uri:
    print("❌ MONGODB_URI not found in .env file")
    exit(1)

print(f"\n🐍 Python Version: 3.11")
print(f"📦 Testing connection...")

try:
    # Simple connection with certifi
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000,
        tlsCAFile=certifi.where()
    )
    
    # Force connection
    client.admin.command('ping')
    
    print("✅ CONNECTION SUCCESSFUL!")
    
    # Test database access
    db = client['neuroaid_db']
    users = db['users']
    
    count = users.count_documents({})
    print(f"✅ Database: neuroaid_db")
    print(f"✅ Users collection: {count} users")
    
    print("\n" + "=" * 60)
    print("🎉 MONGODB IS WORKING!")
    print("=" * 60)
    print("\nYou can now run: python app.py")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nIf still failing, the issue is network/firewall related.")