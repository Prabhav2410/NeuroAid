#!/usr/bin/env python3
"""
MongoDB Atlas Connection Diagnostic Tool
Helps identify the exact issue with your connection
"""

import os
import sys
import ssl
import certifi
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import urlparse, parse_qs

print("=" * 70)
print("MONGODB ATLAS DIAGNOSTIC TOOL")
print("=" * 70)

# Load environment
load_dotenv()
uri = os.getenv('MONGODB_URI')

# Check 1: Environment Variable
print("\n[1/7] Checking .env file...")
if not uri:
    print("   ❌ MONGODB_URI not found in .env file")
    print("   → Create .env file with MONGODB_URI=your_connection_string")
    sys.exit(1)
else:
    print("   ✅ MONGODB_URI found")
    # Mask password
    if '@' in uri:
        parts = uri.split('@')
        masked = f"{parts[0].split(':')[0]}:****@{parts[1]}"
        print(f"   URI: {masked[:60]}...")

# Check 2: URI Format
print("\n[2/7] Validating URI format...")
if not uri.startswith('mongodb+srv://') and not uri.startswith('mongodb://'):
    print("   ❌ Invalid URI format")
    print("   → Must start with mongodb:// or mongodb+srv://")
    sys.exit(1)
else:
    print("   ✅ URI format valid")

# Check 3: Parse URI Components
print("\n[3/7] Parsing URI components...")
try:
    if uri.startswith('mongodb+srv://'):
        # Extract components
        after_scheme = uri.replace('mongodb+srv://', '')
        if '@' in after_scheme:
            creds, rest = after_scheme.split('@', 1)
            if ':' in creds:
                username, password = creds.split(':', 1)
                cluster = rest.split('/')[0]
                
                print(f"   Username: {username}")
                print(f"   Password: {'*' * len(password)}")
                print(f"   Cluster: {cluster}")
                
                # Check for special characters in password
                special_chars = '@#$%^&*()+=[]{}|\\:;"\'<>,.?/~`'
                unencoded = [c for c in special_chars if c in password]
                if unencoded:
                    print(f"   ⚠️  Password has unencoded special characters: {unencoded}")
                    print(f"   → Use fix_mongodb_uri.py to encode password")
                else:
                    print("   ✅ Password appears properly encoded")
        print("   ✅ URI components valid")
except Exception as e:
    print(f"   ⚠️  Could not parse URI: {e}")

# Check 4: Internet Connection
print("\n[4/7] Checking internet connection...")
try:
    response = requests.get('https://www.google.com', timeout=5)
    print("   ✅ Internet connection active")
except:
    print("   ❌ No internet connection")
    print("   → Check your network connection")
    sys.exit(1)

# Check 5: MongoDB Atlas Reachability
print("\n[5/7] Checking MongoDB Atlas reachability...")
try:
    cluster = uri.split('@')[1].split('/')[0]
    response = requests.get(f'https://{cluster}', timeout=10)
    print("   ✅ MongoDB Atlas cluster reachable")
except Exception as e:
    print(f"   ⚠️  Could not reach cluster: {e}")

# Check 6: SSL/TLS Configuration
print("\n[6/7] Checking SSL/TLS setup...")
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    print(f"   Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"   SSL version: {ssl.OPENSSL_VERSION}")
    print(f"   Certifi CA bundle: {certifi.where()}")
    print("   ✅ SSL/TLS configuration OK")
except Exception as e:
    print(f"   ❌ SSL/TLS error: {e}")

# Check 7: Actual MongoDB Connection
print("\n[7/7] Testing MongoDB connection...")
print("   Attempting connection (timeout: 10s)...")

try:
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000,
        tlsCAFile=certifi.where()
    )
    
    # Force connection attempt
    info = client.server_info()
    
    print("   ✅ CONNECTION SUCCESSFUL!")
    print(f"   MongoDB version: {info.get('version', 'Unknown')}")
    
    # Test database
    db = client['vermaprabhav51_db']
    collections = db.list_collection_names()
    print(f"   Database: vermaprabhav51_db")
    print(f"   Collections: {len(collections)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL DIAGNOSTICS PASSED - MongoDB is working!")
    print("=" * 70)
    print("\nYour connection is fine. Try running: python app.py")
    
except Exception as e:
    error_str = str(e)
    print(f"   ❌ CONNECTION FAILED")
    print(f"   Error: {error_str[:150]}")
    
    print("\n" + "=" * 70)
    print("🔍 DIAGNOSIS")
    print("=" * 70)
    
    # Analyze the error
    if "SSL" in error_str or "TLS" in error_str:
        print("\n🔴 SSL/TLS Handshake Error")
        print("   Possible causes:")
        print("   1. Windows firewall blocking SSL connections")
        print("   2. Antivirus interfering with connections")
        print("   3. Corporate proxy/VPN blocking MongoDB ports")
        print("\n   Solutions:")
        print("   → Temporarily disable antivirus and retry")
        print("   → Check Windows Firewall settings")
        print("   → Try different network (mobile hotspot)")
        print("   → Run: pip install --upgrade pymongo certifi")
        
    elif "authentication" in error_str.lower() or "auth" in error_str.lower():
        print("\n🔴 Authentication Error")
        print("   Your username or password is incorrect")
        print("\n   Solutions:")
        print("   1. Go to MongoDB Atlas → Database Access")
        print("   2. Verify username: vermaprabhav51_db_user")
        print("   3. Reset password if needed")
        print("   4. Update .env file with new password")
        print("   5. Run: python fix_mongodb_uri.py")
        
    elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
        print("\n🔴 Network Timeout Error")
        print("   Cannot reach MongoDB Atlas servers")
        print("\n   Solutions:")
        print("   1. Go to MongoDB Atlas → Network Access")
        print("   2. Click 'Add IP Address'")
        print("   3. Select 'Add Current IP Address'")
        print("   4. OR use 0.0.0.0/0 to allow all IPs (testing only)")
        
    elif "IP" in error_str or "whitelist" in error_str.lower():
        print("\n🔴 IP Whitelist Error")
        print("   Your IP address is not whitelisted")
        print("\n   Solutions:")
        print("   1. Go to MongoDB Atlas → Network Access")
        print("   2. Click 'Add IP Address'")
        print("   3. Select 'Add Current IP Address'")
        print("   4. Wait 1-2 minutes for changes to propagate")
        
    else:
        print("\n🔴 Unknown Error")
        print("\n   Try these steps:")
        print("   1. Restart your computer")
        print("   2. Run: pip install --upgrade pymongo certifi")
        print("   3. Check MongoDB Atlas status: https://status.mongodb.com")
        print("   4. Try mobile hotspot to rule out network issues")
    
    print("\n" + "=" * 70)
    sys.exit(1)