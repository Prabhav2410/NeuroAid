#!/usr/bin/env python3
"""
Fix MongoDB Connection String - Encode Special Characters
Run this to automatically fix your MONGODB_URI in .env file
"""

import re
from urllib.parse import quote_plus

def fix_mongodb_uri():
    """Fix MongoDB URI by encoding password"""
    
    print("=" * 60)
    print("MONGODB URI FIXER")
    print("=" * 60)
    print()
    
    # Read current .env file
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
    except FileNotFoundError:
        print("❌ .env file not found!")
        print("Please create .env file first.")
        return False
    
    # Find MONGODB_URI line
    uri_match = re.search(r'MONGODB_URI=(.+?)(?:\n|$)', env_content)
    
    if not uri_match:
        print("❌ MONGODB_URI not found in .env file")
        return False
    
    current_uri = uri_match.group(1).strip()
    print(f"Current URI: {current_uri[:50]}...")
    print()
    
    # Parse the connection string
    # Format: mongodb+srv://username:password@cluster.mongodb.net/...
    pattern = r'mongodb\+srv://([^:]+):([^@]+)@(.+)'
    match = re.match(pattern, current_uri)
    
    if not match:
        print("❌ Could not parse MongoDB URI")
        print("Expected format: mongodb+srv://username:password@cluster.mongodb.net/...")
        return False
    
    username = match.group(1)
    password = match.group(2)
    rest = match.group(3)
    
    print(f"Username: {username}")
    print(f"Password: {password}")
    print()
    
    # Check if password needs encoding
    needs_encoding = any(char in password for char in '@#$%^&*()+=[]{}|\\:;"\'<>,.?/~`')
    
    if not needs_encoding:
        print("✅ Password doesn't have special characters - no encoding needed!")
        print("Your connection string is already correct.")
        return True
    
    print("⚠️  Password contains special characters that need encoding:")
    for char in '@#$%^&*()+=[]{}|\\:;"\'<>,.?/~`':
        if char in password:
            print(f"   Found: {char}")
    
    print()
    print("🔧 Encoding password...")
    
    # Encode password
    encoded_password = quote_plus(password)
    
    # Build new URI
    new_uri = f"mongodb+srv://{username}:{encoded_password}@{rest}"
    
    print()
    print("Old URI:")
    print(f"  {current_uri}")
    print()
    print("New URI (encoded):")
    print(f"  {new_uri}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to update .env file? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Cancelled. No changes made.")
        return False
    
    # Update .env file
    new_env_content = env_content.replace(
        f"MONGODB_URI={current_uri}",
        f"MONGODB_URI={new_uri}"
    )
    
    # Backup old .env
    with open('.env.backup', 'w') as f:
        f.write(env_content)
    print("✅ Backup saved to .env.backup")
    
    # Write new .env
    with open('.env', 'w') as f:
        f.write(new_env_content)
    print("✅ .env file updated!")
    
    print()
    print("=" * 60)
    print("✅ MONGODB URI FIXED!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run: python test_mongodb.py")
    print("  2. Should see: ✅ ALL TESTS PASSED")
    print()
    
    return True


def manual_fix():
    """Guide user through manual fix"""
    
    print()
    print("=" * 60)
    print("MANUAL FIX GUIDE")
    print("=" * 60)
    print()
    
    print("If automatic fix didn't work, follow these steps:")
    print()
    
    print("1. Get your MongoDB credentials:")
    username = input("   Enter MongoDB username: ").strip()
    password = input("   Enter MongoDB password: ").strip()
    cluster = input("   Enter cluster URL (e.g., cluster0.xxxxx.mongodb.net): ").strip()
    
    print()
    print("2. Encoding password...")
    
    encoded_password = quote_plus(password)
    
    print()
    print("3. Your encoded connection string:")
    print()
    print(f"mongodb+srv://{username}:{encoded_password}@{cluster}/?retryWrites=true&w=majority")
    print()
    
    print("4. Copy the above connection string and paste it into your .env file:")
    print()
    print("   MONGODB_URI=mongodb+srv://...")
    print()
    print("5. Save the file and run: python test_mongodb.py")
    print()


if __name__ == "__main__":
    import sys
    
    try:
        success = fix_mongodb_uri()
        
        if not success:
            print()
            response = input("Would you like to try manual fix? (y/n): ")
            if response.lower() == 'y':
                manual_fix()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)