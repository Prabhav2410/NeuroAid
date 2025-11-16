# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import joblib
import os
import json
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import MongoDB database
try:
    from database import get_database
    db = get_database()
    DATABASE_AVAILABLE = True
    print("✅ MongoDB database connected")
except ImportError:
    print("⚠️ Warning: database.py not found. Using fallback authentication.")
    DATABASE_AVAILABLE = False
    db = None
except Exception as e:
    print(f"⚠️ Warning: MongoDB connection error - {e}")
    DATABASE_AVAILABLE = False
    db = None

# Import your prediction functions
try:
    from disease_prediction import predict_disease, validate_symptoms
except ImportError:
    print("⚠️ Warning: disease_prediction.py not found")
    predict_disease = None
    validate_symptoms = None

# Import skin disease prediction
try:
    from skin_disease_prediction import predict_with_details as skin_predict_with_details
    from skin_disease_prediction import validate_image as skin_validate_image
    SKIN_MODEL_LOADED = True
    print("✅ Skin disease model loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: skin_disease_prediction.py not found - {e}")
    SKIN_MODEL_LOADED = False
except Exception as e:
    print(f"⚠️ Warning: Error loading skin disease model - {e}")
    SKIN_MODEL_LOADED = False

# Import X-ray disease prediction
try:
    from xray_disease_prediction import predict_with_details as xray_predict_with_details
    from xray_disease_prediction import validate_image as xray_validate_image
    XRAY_MODEL_LOADED = True
    print("✅ X-ray disease model loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: xray_disease_prediction.py not found - {e}")
    XRAY_MODEL_LOADED = False
except Exception as e:
    print(f"⚠️ Warning: Error loading X-ray disease model - {e}")
    XRAY_MODEL_LOADED = False

# Import chatbot
try:
    from chatbot_gemini import get_chatbot
    CHATBOT_AVAILABLE = True
    print("✅ Gemini chatbot loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: chatbot_gemini.py not found - {e}")
    print("   Install: pip install google-generativeai")
    CHATBOT_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Warning: Error loading chatbot - {e}")
    CHATBOT_AVAILABLE = False

# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# FILE UPLOAD CONFIGURATION
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model artifacts
try:
    symptoms_list = joblib.load("symptom_list.pkl")
    print(f"✅ Loaded {len(symptoms_list)} symptoms")
except FileNotFoundError:
    print("⚠️ Warning: symptom_list.pkl not found. Run training first.")
    symptoms_list = []

try:
    model_metadata = joblib.load("model_metadata.pkl")
    print(f"✅ Model metadata loaded: {model_metadata['test_accuracy']:.2%} accuracy")
except FileNotFoundError:
    print("⚠️ Warning: model_metadata.pkl not found")
    model_metadata = None


# ============================================================
# AUTHENTICATION DECORATOR
# ============================================================

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# ROUTES - HOME & STATIC PAGES
# ============================================================

@app.route('/')
def home():
    """Homepage with hero section and features"""
    return render_template('home.html', title='NeuroAid - AI Healthcare Assistant')


@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html', title='About NeuroAid')


@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html', title='Contact Us')


# ============================================================
# ROUTES - AUTHENTICATION WITH MONGODB
# ============================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page with MongoDB authentication"""
    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username_or_email or not password:
            return render_template('login.html', 
                                 error='Please enter both username/email and password',
                                 title='Login')
        
        # Use MongoDB authentication if available
        if DATABASE_AVAILABLE and db:
            # Verify user with MongoDB
            success, message, user_data = db.verify_user(username_or_email, password)
            
            if success:
                # Store user data in session
                session['user'] = user_data['username']
                session['user_id'] = user_data['user_id']
                session['fullname'] = user_data['fullname']
                session['email'] = user_data['email']
                
                print(f"✅ User logged in: {user_data['username']}")
                
                # Redirect to home page
                return redirect(url_for('home'))
            else:
                return render_template('login.html', 
                                     error=message,
                                     title='Login')
        else:
            # Fallback to hardcoded authentication if MongoDB not available
            if username_or_email == 'admin' and password == 'admin':
                session['user'] = 'admin'
                session['user_id'] = '1'
                session['fullname'] = 'Admin User'
                session['email'] = 'admin@neuroaid.com'
                return redirect(url_for('home'))
            else:
                return render_template('login.html', 
                                     error='Invalid credentials. MongoDB not connected - use admin/admin',
                                     title='Login')
    
    # If already logged in, redirect to home
    if 'user' in session:
        return redirect(url_for('home'))
    
    return render_template('login.html', title='Login')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page with MongoDB integration"""
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validation
        if not all([fullname, email, username, password, confirm_password]):
            return render_template('signup.html', 
                                 error='All fields are required',
                                 title='Sign Up')
        
        if password != confirm_password:
            return render_template('signup.html', 
                                 error='Passwords do not match',
                                 title='Sign Up')
        
        if len(password) < 6:
            return render_template('signup.html', 
                                 error='Password must be at least 6 characters',
                                 title='Sign Up')
        
        if len(username) < 3:
            return render_template('signup.html', 
                                 error='Username must be at least 3 characters',
                                 title='Sign Up')
        
        # Use MongoDB if available
        if DATABASE_AVAILABLE and db:
            # Create user in MongoDB
            success, message, user_id = db.create_user(username, email, password, fullname)
            
            if success:
                # Auto-login after successful signup
                session['user'] = username.lower()
                session['user_id'] = user_id
                session['fullname'] = fullname
                session['email'] = email
                
                print(f"✅ New user registered: {username}")
                
                return redirect(url_for('home'))
            else:
                return render_template('signup.html', 
                                     error=message,
                                     title='Sign Up')
        else:
            # Fallback if MongoDB not available
            return render_template('signup.html', 
                                 error='Database not available. Please contact administrator.',
                                 title='Sign Up')
    
    # If already logged in, redirect to home
    if 'user' in session:
        return redirect(url_for('home'))
    
    return render_template('signup.html', title='Sign Up')


@app.route('/logout')
def logout():
    """Logout and clear session"""
    username = session.get('user', 'Unknown')
    session.clear()
    print(f"✅ User logged out: {username}")
    return redirect(url_for('home'))


@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    if DATABASE_AVAILABLE and db:
        user_id = session.get('user_id')
        user_data = db.get_user_by_id(user_id)
        
        if not user_data:
            return redirect(url_for('logout'))
        
        return render_template('profile.html', 
                             user=user_data,
                             title='My Profile')
    else:
        # Fallback profile data from session
        user_data = {
            'username': session.get('user'),
            'email': session.get('email'),
            'fullname': session.get('fullname')
        }
        return render_template('profile.html', 
                             user=user_data,
                             title='My Profile')


@app.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile"""
    if not DATABASE_AVAILABLE or not db:
        return jsonify({
            'success': False,
            'error': 'Database not available'
        }), 503
    
    user_id = session.get('user_id')
    fullname = request.form.get('fullname', '').strip()
    email = request.form.get('email', '').strip()
    
    if not fullname or not email:
        return jsonify({'success': False, 'error': 'All fields required'}), 400
    
    update_data = {
        'fullname': fullname,
        'email': email.lower()
    }
    
    success = db.update_user(user_id, update_data)
    
    if success:
        session['fullname'] = fullname
        session['email'] = email
        return jsonify({'success': True, 'message': 'Profile updated'})
    else:
        return jsonify({'success': False, 'error': 'Update failed'}), 500


# ============================================================
# ROUTES - DISEASE PREDICTION (FIXED!)
# ============================================================

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms_page():
    """Main symptom selection and prediction page"""
    
    if not symptoms_list:
        return render_template('error.html', 
                             error='Model not loaded. Please train the model first.',
                             title='Error')
    
    result = None
    selected_symptoms = []
    
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        
        print("=" * 60)
        print("🔍 DEBUG: SYMPTOM PREDICTION")
        print("=" * 60)
        print(f"Selected symptoms: {selected_symptoms}")
        
        if selected_symptoms:
            if validate_symptoms:
                valid_symptoms, invalid_symptoms = validate_symptoms(selected_symptoms)
                
                print(f"Valid symptoms: {valid_symptoms}")
                print(f"Invalid symptoms: {invalid_symptoms}")
                
                if invalid_symptoms:
                    print(f"⚠️ Invalid symptoms: {invalid_symptoms}")
                
                if valid_symptoms and predict_disease:
                    try:
                        result = predict_disease(valid_symptoms)
                        
                        print(f"Raw result type: {type(result)}")
                        print(f"Raw result: {result}")
                        
                        # Make sure result is a list of tuples
                        if result and isinstance(result, list):
                            print(f"First result item: {result[0]}")
                            print(f"First result type: {type(result[0])}")
                        
                        session['last_prediction'] = {
                            'symptoms': valid_symptoms,
                            'results': result
                        }
                        
                        print("✅ Prediction successful!")
                        print("=" * 60)
                        
                    except Exception as e:
                        print(f"❌ Prediction error: {e}")
                        import traceback
                        traceback.print_exc()
                        result = [("Error in prediction", 0.0, "N/A")]
                else:
                    result = [("No valid symptoms selected", 0.0, "N/A")]
            else:
                if predict_disease:
                    result = predict_disease(selected_symptoms)
        else:
            result = [("Please select at least one symptom", 0.0, "N/A")]
    
    formatted_symptoms = [
        {
            'value': sym,
            'label': sym.replace('_', ' ').title()
        }
        for sym in symptoms_list
    ]
    
    return render_template('symptoms.html', 
                         symptoms=symptoms_list,
                         formatted_symptoms=formatted_symptoms,
                         selected=selected_symptoms,
                         result=result,
                         title='Symptom Checker')


# ============================================================
# SKIN DISEASE ROUTES
# ============================================================

@app.route('/skin-checker')
def skin_checker():
    """Skin disease image checker page"""
    if not SKIN_MODEL_LOADED:
        return render_template('error.html',
                             error='Skin disease model not loaded. Please ensure skin_disease_model.h5 exists.',
                             title='Error')
    
    return render_template('skin_checker.html', title='Skin Disease Checker')


@app.route('/upload-skin-image', methods=['POST'])
def upload_skin_image():
    """Handle skin disease image upload and prediction"""
    try:
        if not SKIN_MODEL_LOADED:
            return jsonify({
                'error': 'Skin disease model not loaded',
                'success': False
            }), 503
        
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"skin_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate image
        is_valid, message = skin_validate_image(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({
                'error': message,
                'success': False
            }), 400
        
        # Make prediction
        results = skin_predict_with_details(filepath)
        
        # Store in session
        session['last_skin_prediction'] = {
            'image': filename,
            'results': results
        }
        
        return jsonify({
            'success': True,
            'top_prediction': results['top_prediction'],
            'top_confidence': results['top_confidence'],
            'is_confident': results['is_confident'],
            'all_predictions': results['all_predictions'][:5],
            'image_url': url_for('uploaded_file', filename=filename)
        })
        
    except Exception as e:
        print(f"❌ ERROR in upload_skin_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


# ============================================================
# X-RAY DISEASE ROUTES
# ============================================================

@app.route('/xray-checker')
def xray_checker():
    """X-ray disease checker page"""
    if not XRAY_MODEL_LOADED:
        return render_template('error.html',
                             error='X-ray disease model not loaded. Please ensure xray_disease_model.h5 exists.',
                             title='Error')
    
    return render_template('xray_checker.html', title='X-Ray Disease Checker')


@app.route('/upload-xray-image', methods=['POST'])
def upload_xray_image():
    """Handle X-ray image upload and prediction"""
    try:
        if not XRAY_MODEL_LOADED:
            return jsonify({
                'error': 'X-ray disease model not loaded',
                'success': False
            }), 503
        
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"xray_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"DEBUG: X-ray file saved to {filepath}")
        
        # Validate image
        is_valid, message = xray_validate_image(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({
                'error': message,
                'success': False
            }), 400
        
        print("DEBUG: X-ray image validated, making prediction...")
        
        # Make prediction
        results = xray_predict_with_details(filepath)
        
        print(f"DEBUG: X-ray prediction complete - {results['top_prediction']}")
        
        # Store in session
        session['last_xray_prediction'] = {
            'image': filename,
            'results': results
        }
        
        return jsonify({
            'success': True,
            'top_prediction': results['top_prediction'],
            'top_confidence': results['top_confidence'],
            'is_confident': results['is_confident'],
            'severity': results['severity'],
            'is_normal': results['is_normal'],
            'all_predictions': results['all_predictions'][:4],
            'image_url': url_for('uploaded_file', filename=filename)
        })
        
    except Exception as e:
        print(f"❌ ERROR in upload_xray_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/xray-results')
def xray_results():
    """Display last X-ray prediction results"""
    prediction_data = session.get('last_xray_prediction')
    
    if not prediction_data:
        return redirect(url_for('xray_checker'))
    
    return render_template('xray_results.html',
                         image=prediction_data['image'],
                         results=prediction_data['results'],
                         title='X-Ray Analysis Results')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================
# CHATBOT ROUTES
# ============================================================

@app.route('/chatbot')
def chatbot():
    """Chatbot page"""
    if not CHATBOT_AVAILABLE:
        return render_template('error.html',
                             error='Chatbot is not available. Please install google-generativeai and configure GEMINI_API_KEY.',
                             title='Error')
    return render_template('chatbot.html', title='AI Health Assistant')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chatbot messages"""
    try:
        if not CHATBOT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Chatbot service is not available. Please install google-generativeai.'
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        print(f"💬 Chatbot request: {message[:50]}...")
        
        chatbot = get_chatbot()
        
        if not chatbot.is_available():
            return jsonify({
                'success': False,
                'error': 'Chatbot is not properly configured. Please set GEMINI_API_KEY environment variable.'
            }), 503
        
        success, response = chatbot.get_response(message)
        
        if success:
            print(f"✅ Chatbot response: {response[:50]}...")
            return jsonify({
                'success': True,
                'response': response
            })
        else:
            print(f"❌ Chatbot error: {response}")
            return jsonify({
                'success': False,
                'error': response
            }), 500
        
    except Exception as e:
        print(f"❌ Chat API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/chat/reset', methods=['POST'])
def reset_chat():
    """Reset the chat conversation"""
    try:
        if not CHATBOT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Chatbot not available'
            }), 503
        
        chatbot = get_chatbot()
        chatbot.reset_conversation()
        
        return jsonify({
            'success': True,
            'message': 'Conversation reset successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chatbot-status')
def chatbot_status():
    """Check if chatbot is available"""
    if CHATBOT_AVAILABLE:
        try:
            chatbot = get_chatbot()
            is_ready = chatbot.is_available()
            return jsonify({
                'available': True,
                'ready': is_ready,
                'message': 'Chatbot is ready' if is_ready else 'Chatbot needs configuration'
            })
        except Exception as e:
            return jsonify({
                'available': True,
                'ready': False,
                'message': f'Error: {str(e)}'
            })
    else:
        return jsonify({
            'available': False,
            'ready': False,
            'message': 'Chatbot module not loaded'
        })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error='Page not found',
                         title='404 Error'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error='Internal server error',
                         title='500 Error'), 500


# ============================================================
# CONTEXT PROCESSORS
# ============================================================

@app.context_processor
def inject_globals():
    """Inject global variables into all templates"""
    return {
        'app_name': 'NeuroAid',
        'version': '1.1.0',
        'model_accuracy': f"{model_metadata['test_accuracy']:.1%}" if model_metadata else 'N/A',
        'chatbot_available': CHATBOT_AVAILABLE,
        'xray_available': XRAY_MODEL_LOADED,
        'skin_available': SKIN_MODEL_LOADED,
        'database_available': DATABASE_AVAILABLE
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("NEUROAID - STARTING APPLICATION")
    print("=" * 60)
    
    # Check MongoDB connection
    if DATABASE_AVAILABLE and db:
        print("✅ MongoDB database connected and ready")
    else:
        print("⚠️  MongoDB not connected - using fallback authentication")
        print("   Login with: admin/admin")
    
    # Check symptom model files
    required_files = ['rf_disease_model.pkl', 'symptom_list.pkl', 'class_names.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️  WARNING: Missing symptom model files:")
        print(f"   {', '.join(missing_files)}")
    else:
        print("✅ Symptom model files loaded")
    
    # Check skin model files
    skin_files = ['skin_disease_model.h5', 'skin_disease_classes.json']
    missing_skin = [f for f in skin_files if not os.path.exists(f)]
    
    if missing_skin:
        print("⚠️  WARNING: Missing skin disease model files:")
        print(f"   {', '.join(missing_skin)}")
        print("   Run: python skin_disease_trainer.py")
    else:
        print("✅ Skin disease model files found")
    
    # Check X-ray model files
    xray_files = ['xray_disease_model.h5', 'xray_disease_classes.json']
    missing_xray = [f for f in xray_files if not os.path.exists(f)]
    
    if missing_xray:
        print("⚠️  WARNING: Missing X-ray disease model files:")
        print(f"   {', '.join(missing_xray)}")
        print("   Run: python xray_disease_trainer.py")
    else:
        print("✅ X-ray disease model files found")
    
    # Check chatbot status
    if CHATBOT_AVAILABLE:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print(f"✅ Chatbot ready (API Key: {api_key[:10]}...)")
        else:
            print("⚠️  Chatbot loaded but missing API key")
            print("   Set: GEMINI_API_KEY environment variable")
    else:
        print("⚠️  Chatbot not available")
        print("   Install: pip install google-generativeai")
    
    print(f"\nDATABASE_AVAILABLE: {DATABASE_AVAILABLE}")
    print(f"SKIN_MODEL_LOADED: {SKIN_MODEL_LOADED}")
    print(f"XRAY_MODEL_LOADED: {XRAY_MODEL_LOADED}")
    print(f"CHATBOT_AVAILABLE: {CHATBOT_AVAILABLE}")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )