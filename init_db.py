#!/usr/bin/env python3
"""
Database Initialization Script
Run this once to set up MongoDB collections and indexes
"""

from database import get_database
from datetime import datetime

def initialize_database():
    """Initialize MongoDB database with proper indexes and collections"""
    
    print("=" * 60)
    print("NEUROAID - DATABASE INITIALIZATION")
    print("=" * 60)
    print()
    
    # Connect to database
    print("📦 Connecting to MongoDB...")
    db = get_database()
    
    if not db.client:
        print("❌ Failed to connect to MongoDB")
        return False
    
    print("✅ Connected successfully")
    print()
    
    # Create users collection indexes
    print("🔧 Creating indexes for users collection...")
    try:
        # Username index (unique)
        db.users.create_index('username', unique=True)
        print("   ✅ Username index created")
        
        # Email index (unique)
        db.users.create_index('email', unique=True)
        print("   ✅ Email index created")
        
        # Created date index (for sorting)
        db.users.create_index('created_at')
        print("   ✅ Created date index created")
        
        # Active users index (for filtering)
        db.users.create_index('is_active')
        print("   ✅ Active status index created")
        
    except Exception as e:
        print(f"   ⚠️  Index creation warning: {e}")
    
    print()
    
    # Create predictions collection (for storing prediction history)
    print("🔧 Creating predictions collection...")
    try:
        predictions = db.db['predictions']
        
        # User ID index
        predictions.create_index('user_id')
        print("   ✅ User ID index created")
        
        # Prediction type index
        predictions.create_index('prediction_type')
        print("   ✅ Prediction type index created")
        
        # Created date index
        predictions.create_index('created_at')
        print("   ✅ Date index created")
        
    except Exception as e:
        print(f"   ⚠️  Predictions collection warning: {e}")
    
    print()
    
    # Display database statistics
    print("📊 Database Statistics:")
    try:
        user_count = db.users.count_documents({})
        print(f"   Users: {user_count}")
        
        active_users = db.users.count_documents({'is_active': True})
        print(f"   Active users: {active_users}")
        
        # List all collections
        collections = db.db.list_collection_names()
        print(f"   Collections: {', '.join(collections)}")
        
    except Exception as e:
        print(f"   ⚠️  Could not get statistics: {e}")
    
    print()
    print("=" * 60)
    print("✅ DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)
    
    return True


def create_admin_user():
    """Create default admin user"""
    
    print("\n" + "=" * 60)
    print("CREATE ADMIN USER (OPTIONAL)")
    print("=" * 60)
    
    response = input("\nDo you want to create an admin user? (y/n): ")
    
    if response.lower() != 'y':
        print("Skipped admin user creation")
        return
    
    print()
    username = input("Enter admin username (default: admin): ").strip() or "admin"
    email = input("Enter admin email (default: admin@neuroaid.com): ").strip() or "admin@neuroaid.com"
    password = input("Enter admin password (default: admin123): ").strip() or "admin123"
    fullname = input("Enter admin full name (default: Admin User): ").strip() or "Admin User"
    
    print("\nCreating admin user...")
    
    db = get_database()
    success, message, user_id = db.create_user(
        username=username,
        email=email,
        password=password,
        fullname=fullname
    )
    
    if success:
        print(f"✅ Admin user created successfully!")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   User ID: {user_id}")
    else:
        print(f"❌ Failed to create admin user: {message}")


def show_connection_info():
    """Display MongoDB connection information"""
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("\n" + "=" * 60)
    print("CONNECTION INFORMATION")
    print("=" * 60)
    
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    
    # Hide password in URI
    if '@' in mongo_uri and ':' in mongo_uri:
        parts = mongo_uri.split('@')
        credentials = parts[0].split(':')
        if len(credentials) > 1:
            display_uri = f"{credentials[0]}:****@{parts[1]}"
        else:
            display_uri = mongo_uri
    else:
        display_uri = mongo_uri
    
    print(f"\nMongoDB URI: {display_uri}")
    print(f"Database: neuroaid_db")
    print()


def main():
    """Main function"""
    
    try:
        # Show connection info
        show_connection_info()
        
        # Initialize database
        success = initialize_database()
        
        if not success:
            print("\n❌ Database initialization failed")
            return 1
        
        # Optional: Create admin user
        create_admin_user()
        
        print("\n🎉 Setup complete! You can now run your application:")
        print("   python app.py")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())