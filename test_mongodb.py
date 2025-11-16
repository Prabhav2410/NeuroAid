import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('MONGODB_URI')

print("=" * 60)
print("MONGODB CONNECTION - RELAXED SSL MODE")
print("=" * 60)

try:
    print("\n🔌 Connecting with relaxed SSL...")
    
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=15000,
        tls=True,
        tlsAllowInvalidCertificates=True,  # Bypass SSL verification
        tlsInsecure=True
    )
    
    client.server_info()
    print("   ✅ CONNECTION SUCCESSFUL!")
    
    db = client['neuroaid_db']
    users = db['users']
    user_count = users.count_documents({})
    
    print(f"\n✅ Database: neuroaid_db")
    print(f"✅ Users: {user_count}")
    print("\n" + "=" * 60)
    print("SUCCESS! MongoDB is working!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Still failed: {e}")