from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import secrets
import requests
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import cv2
import time
from dotenv import load_dotenv
import re
import traceback
import sys


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# API Keys
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '2f021261bac8c9f5f35de84b6486589e')

# Model Configuration
SKIN_CANCER_MODEL_PATH = os.getenv('SKIN_CANCER_MODEL_PATH', 'skin_cancer_model.h5')
SKIN_CANCER_IMG_SIZE = (224, 224)
GOOGLE_DRIVE_FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID', '1Mt2Xvx--d04qxP-rrrfZcjAsj8RN_IPN')

# Memory optimization: Lazy import TensorFlow only when needed
tensorflow_available = False
skin_cancer_model = None
ML_ENABLED = False
ML_ERROR_MESSAGE = None

# Check memory constraints
MEMORY_LIMIT_MB = int(os.environ.get('MEMORY_LIMIT_MB', '512'))
ENABLE_ML = os.environ.get('ENABLE_ML', 'false').lower() == 'true'

print(f"🔧 Configuration:")
print(f"   Memory limit: {MEMORY_LIMIT_MB}MB")
print(f"   ML explicitly enabled: {ENABLE_ML}")
print(f"   Model path: {SKIN_CANCER_MODEL_PATH}")
print(f"   Google Drive ID: {GOOGLE_DRIVE_FILE_ID}")

# Auto-download model on startup if not present
if ENABLE_ML and not os.path.exists(SKIN_CANCER_MODEL_PATH):
    print("🚀 Model not found, attempting download...")
    try:
        from download_model import download_model
        download_model()
    except Exception as e:
        print(f"⚠️ Could not auto-download model: {e}")

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disable Ollama for production deployment
OLLAMA_ENABLED = False

def check_memory_usage():
    """Check current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except ImportError:
        print("⚠️  psutil not available - install with: pip install psutil")
        return 0

def diagnose_ml_setup():
    """Comprehensive ML setup diagnosis"""
    print("\n🩺 DIAGNOSING ML SETUP...")
    print("-" * 50)
    
    issues = []
    
    # Check 1: Environment Variables
    print("1. 🔍 Environment Variables:")
    if not ENABLE_ML:
        issues.append("ENABLE_ML is set to False")
        print(f"   ❌ ENABLE_ML: {os.environ.get('ENABLE_ML', 'not set')} (should be 'true')")
    else:
        print(f"   ✅ ENABLE_ML: {ENABLE_ML}")
    
    # Check 2: Model File
    print("2. 📁 Model File:")
    if os.path.exists(SKIN_CANCER_MODEL_PATH):
        file_size = os.path.getsize(SKIN_CANCER_MODEL_PATH) / (1024*1024)
        print(f"   ✅ Model file exists: {SKIN_CANCER_MODEL_PATH} ({file_size:.1f} MB)")
    else:
        issues.append(f"Model file not found at {SKIN_CANCER_MODEL_PATH}")
        print(f"   ❌ Model file not found: {SKIN_CANCER_MODEL_PATH}")
        print(f"   💡 Download from: https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view")
    
    # Check 3: TensorFlow
    print("3. 🤖 TensorFlow:")
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow available: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   🎮 GPU available: {len(gpus)} device(s)")
        else:
            print("   💻 Running on CPU")
            
    except ImportError as e:
        issues.append("TensorFlow not installed")
        print(f"   ❌ TensorFlow not available: {e}")
        print(f"   💡 Install with: pip install tensorflow")
    
    # Check 4: Keras
    print("4. 🧠 Keras:")
    try:
        from keras.models import load_model
        print("   ✅ Keras available")
    except ImportError:
        try:
            from tensorflow.keras.models import load_model
            print("   ✅ Keras available via TensorFlow")
        except ImportError as e:
            issues.append("Keras not available")
            print(f"   ❌ Keras not available: {e}")
    
    # Check 5: Memory
    print("5. 💾 Memory:")
    current_memory = check_memory_usage()
    if current_memory > 0:
        memory_percent = (current_memory / MEMORY_LIMIT_MB) * 100
        print(f"   📊 Current usage: {current_memory:.1f}MB ({memory_percent:.1f}% of {MEMORY_LIMIT_MB}MB limit)")
        if current_memory > MEMORY_LIMIT_MB * 0.8:
            issues.append("High memory usage")
            print("   ⚠️  High memory usage - may prevent model loading")
        else:
            print("   ✅ Memory usage acceptable")
    else:
        print("   ⚠️  Cannot check memory (psutil not available)")
    
    # Summary
    if issues:
        print(f"\n❌ ISSUES FOUND ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print(f"\n💡 QUICK FIXES:")
        print(f"   export ENABLE_ML=true")
        print(f"   pip install tensorflow keras pillow opencv-python psutil")
        print(f"   # Download model from Google Drive to {SKIN_CANCER_MODEL_PATH}")
    else:
        print(f"\n✅ All checks passed! ML should be working.")
    
    return len(issues) == 0

def lazy_import_tensorflow():
    """Lazy import TensorFlow to save memory"""
    global tensorflow_available, ML_ERROR_MESSAGE
    
    if not tensorflow_available:
        try:
            print("📦 Loading TensorFlow...")
            import tensorflow as tf
            from keras.models import load_model
            
            # Optimize TensorFlow for memory
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Limit GPU memory growth if GPU is available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🎮 Configured {len(gpus)} GPU(s) for memory growth")
            
            tensorflow_available = True
            print("✅ TensorFlow loaded successfully")
            return True
            
        except ImportError as e:
            ML_ERROR_MESSAGE = f"TensorFlow not available: {e}"
            print(f"❌ {ML_ERROR_MESSAGE}")
            print("💡 Install with: pip install tensorflow")
            return False
        except Exception as e:
            ML_ERROR_MESSAGE = f"TensorFlow error: {e}"
            print(f"❌ {ML_ERROR_MESSAGE}")
            return False
    
    return True

def load_model_on_demand():
    """Load model only when needed to save memory"""
    global skin_cancer_model, ML_ENABLED, ML_ERROR_MESSAGE
    
    print(f"🔄 Attempting to load ML model...")
    
    # Check if ML is enabled
    if not ENABLE_ML:
        ML_ERROR_MESSAGE = "ML disabled by environment variable ENABLE_ML=false"
        print(f"❌ {ML_ERROR_MESSAGE}")
        print("💡 Fix: Set environment variable ENABLE_ML=true")
        return None
    
    # Check memory constraints
    current_memory = check_memory_usage()
    memory_threshold = MEMORY_LIMIT_MB * 0.8  # 80% of limit
    
    if current_memory > memory_threshold:
        ML_ERROR_MESSAGE = f"Memory usage too high ({current_memory:.1f}MB > {memory_threshold:.1f}MB threshold)"
        print(f"❌ {ML_ERROR_MESSAGE}")
        return None
    
    # Check if model file exists
    if not os.path.exists(SKIN_CANCER_MODEL_PATH):
        ML_ERROR_MESSAGE = f"Model file not found: {SKIN_CANCER_MODEL_PATH}"
        print(f"❌ {ML_ERROR_MESSAGE}")
        print(f"💡 Download from: https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view")
        return None
    
    # Import TensorFlow
    if not lazy_import_tensorflow():
        return None
    
    # Load model if not already loaded
    if skin_cancer_model is None:
        try:
            from keras.models import load_model
            
            print(f"📂 Loading model from: {SKIN_CANCER_MODEL_PATH}")
            file_size = os.path.getsize(SKIN_CANCER_MODEL_PATH) / (1024*1024)
            print(f"📏 Model file size: {file_size:.1f} MB")
            
            # Load model
            skin_cancer_model = load_model(SKIN_CANCER_MODEL_PATH, compile=False)
            
            # Compile model
            skin_cancer_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            ML_ENABLED = True
            new_memory = check_memory_usage()
            memory_increase = new_memory - current_memory
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Memory usage: {current_memory:.1f}MB → {new_memory:.1f}MB (+{memory_increase:.1f}MB)")
            print(f"🔍 Model input shape: {skin_cancer_model.input_shape}")
            print(f"🎯 Model output shape: {skin_cancer_model.output_shape}")
            
        except Exception as e:
            ML_ERROR_MESSAGE = f"Failed to load model: {str(e)}"
            print(f"❌ {ML_ERROR_MESSAGE}")
            print("📋 Full error traceback:")
            traceback.print_exc()
            return None
    
    return skin_cancer_model

# Template filter for datetime formatting
@app.template_filter('datetime')
def datetime_filter(dt):
    """Format datetime for display"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime('%b %d, %H:%M')

# Context processor to make datetime available in templates
@app.context_processor
def inject_datetime():
    """Make datetime available in all templates"""
    return {
        'now': datetime.now(),
        'datetime': datetime
    }

# Skin types and their characteristics
SKIN_TYPES = {
    'dry': {
        'characteristics': ['Feels tight after washing', 'Flaky or rough texture', 'Prone to fine lines and wrinkles', 'Rarely gets breakouts'],
        'ingredients': ['Hyaluronic Acid', 'Glycerin', 'Ceramides', 'Shea Butter', 'Squalane'],
        'avoid': ['Alcohol', 'Fragrances', 'Sulfates']
    },
    'oily': {
        'characteristics': ['Shiny appearance', 'Enlarged pores', 'Prone to blackheads and pimples', 'Makeup slides off easily'],
        'ingredients': ['Niacinamide', 'Salicylic Acid', 'Tea Tree Oil', 'Clay', 'Zinc'],
        'avoid': ['Heavy oils', 'Petroleum', 'Coconut oil']
    },
    'combination': {
        'characteristics': ['Oily T-zone (forehead, nose, chin)', 'Dry cheeks', 'Enlarged pores in T-zone', 'Occasional breakouts'],
        'ingredients': ['Hyaluronic Acid', 'Niacinamide', 'AHAs', 'Light moisturizers', 'Green Tea'],
        'avoid': ['Heavy creams', 'Harsh exfoliants']
    },
    'sensitive': {
        'characteristics': ['Reacts easily to products', 'Prone to redness', 'May sting or burn', 'Easily irritated'],
        'ingredients': ['Aloe Vera', 'Oatmeal', 'Chamomile', 'Centella Asiatica', 'Calendula'],
        'avoid': ['Fragrances', 'Essential oils', 'Alcohol', 'Sulfates']
    },
    'normal': {
        'characteristics': ['Few imperfections', 'Not sensitive', 'Small pores', 'Balanced hydration'],
        'ingredients': ['Peptides', 'Antioxidants', 'Vitamin C', 'Retinol', 'Ceramides'],
        'avoid': ['Nothing specific to avoid, but moderation is key']
    }
}

# Skin conditions and home remedies
SKIN_CONDITIONS = {
    'acne': {
        'description': 'Inflammatory condition of the skin characterized by pimples, blackheads, and cysts.',
        'ingredients': ['Salicylic Acid', 'Benzoyl Peroxide', 'Tea Tree Oil', 'Niacinamide', 'Zinc'],
        'home_remedies': [
            {'name': 'Tea Tree Oil', 'usage': 'Dilute with carrier oil and apply to affected areas'},
            {'name': 'Honey Mask', 'usage': 'Apply raw honey to face for 15-20 minutes'},
            {'name': 'Green Tea', 'usage': 'Apply cooled green tea as a face wash or toner'}
        ],
        'image': 'static/images/conditions/acne.jpg'
    },
    'dryness': {
        'description': 'Lack of moisture in the skin leading to flakiness, tightness, and discomfort.',
        'ingredients': ['Hyaluronic Acid', 'Glycerin', 'Ceramides', 'Shea Butter', 'Squalane'],
        'home_remedies': [
            {'name': 'Oatmeal Bath', 'usage': 'Add colloidal oatmeal to bathwater and soak'},
            {'name': 'Coconut Oil', 'usage': 'Apply a thin layer as a natural moisturizer'},
            {'name': 'Avocado Mask', 'usage': 'Mash ripe avocado and apply for 15-20 minutes'}
        ],
        'image': 'static/images/conditions/dryness.jpg'
    },
    'hyperpigmentation': {
        'description': 'Darkening of areas of skin due to excess melanin production.',
        'ingredients': ['Vitamin C', 'Alpha Arbutin', 'Kojic Acid', 'Licorice Extract', 'Niacinamide'],
        'home_remedies': [
            {'name': 'Lemon Juice', 'usage': 'Dilute and apply to dark spots (avoid sun exposure)'},
            {'name': 'Aloe Vera', 'usage': 'Apply fresh gel to affected areas'},
            {'name': 'Turmeric Mask', 'usage': 'Mix with yogurt and honey, apply for 15 minutes'}
        ],
        'image': 'static/images/conditions/hyperpigmentation.jpg'
    },
    'rosacea': {
        'description': 'Chronic inflammatory skin condition causing redness and visible blood vessels.',
        'ingredients': ['Azelaic Acid', 'Centella Asiatica', 'Green Tea', 'Aloe Vera', 'Niacinamide'],
        'home_remedies': [
            {'name': 'Green Tea Compress', 'usage': 'Apply cooled green tea bags to affected areas'},
            {'name': 'Oatmeal Mask', 'usage': 'Mix with water and apply to face for 15 minutes'},
            {'name': 'Chamomile Tea', 'usage': 'Use as a facial rinse or toner'}
        ],
        'image': 'static/images/conditions/rosacea.jpg'
    },
    'eczema': {
        'description': 'Inflammatory skin condition causing dry, itchy, and red skin.',
        'ingredients': ['Colloidal Oatmeal', 'Ceramides', 'Shea Butter', 'Aloe Vera', 'Squalane'],
        'home_remedies': [
            {'name': 'Coconut Oil', 'usage': 'Apply to affected areas as needed'},
            {'name': 'Colloidal Oatmeal Bath', 'usage': 'Soak in lukewarm water with colloidal oatmeal'},
            {'name': 'Honey', 'usage': 'Apply raw honey to affected areas for 15-20 minutes'}
        ],
        'image': 'static/images/conditions/eczema.jpg'
    }
}

def get_intelligent_fallback_recommendations(weather_data, skin_type=None, skin_concerns=None):
    """
    Intelligent fallback with sophisticated weather-based logic
    """
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    weather_condition = weather_data['weather'][0]['main'].lower()
    uv_index = weather_data.get('uvi', 5)
    wind_speed = weather_data.get('wind', {}).get('speed', 0)
    
    morning_recs = []
    evening_recs = []
    
    # Morning routine logic
    if skin_type == 'oily':
        morning_recs.append('Salicylic acid gel cleanser for oil control')
    elif skin_type == 'dry':
        morning_recs.append('Creamy hydrating cleanser with ceramides')
    elif skin_type == 'sensitive':
        morning_recs.append('Gentle, fragrance-free cream cleanser')
    else:
        morning_recs.append('Gentle foaming cleanser for daily use')
    
    if temp > 25 and humidity > 70:
        morning_recs.append('Niacinamide serum to control oil and minimize pores in humid weather')
    elif temp < 15 or humidity < 40:
        morning_recs.append('Hyaluronic acid serum for intense hydration in dry conditions')
    elif uv_index > 7:
        morning_recs.append('Vitamin C serum for enhanced antioxidant protection against high UV')
    else:
        morning_recs.append('Balanced antioxidant serum with vitamin E for daily protection')
    
    if humidity > 80:
        morning_recs.append('Oil-free gel moisturizer perfect for high humidity conditions')
    elif humidity < 30:
        morning_recs.append('Rich cream moisturizer with hyaluronic acid for dry air')
    elif temp > 30:
        morning_recs.append('Lightweight, cooling gel moisturizer for hot weather comfort')
    else:
        morning_recs.append('Balanced daily moisturizer with light SPF protection')
    
    if uv_index > 8:
        morning_recs.append(f'SPF 50+ broad spectrum sunscreen (UV index: {uv_index} - very high), reapply every 2 hours')
    elif uv_index > 5:
        morning_recs.append(f'SPF 30+ mineral sunscreen with zinc oxide (UV index: {uv_index} - moderate to high)')
    else:
        morning_recs.append('SPF 30 daily sunscreen with moisturizing benefits')
    
    # Evening routine logic
    if humidity > 70 or 'rain' in weather_condition:
        evening_recs.append('Oil cleanser then foam cleanser (double cleanse) to remove humidity buildup')
    else:
        evening_recs.append('Micellar water followed by gentle cleanser for thorough cleansing')
    
    if temp > 25 and skin_type == 'oily':
        evening_recs.append('BHA toner (2-3 times per week) for deep pore cleansing in warm weather')
    elif temp < 15 or humidity < 40:
        evening_recs.append('Hydrating toner with hyaluronic acid for cold, dry conditions')
    else:
        evening_recs.append('Gentle pH-balancing toner to prep skin for treatments')
    
    if skin_concerns and 'hyperpigmentation' in skin_concerns:
        evening_recs.append('Vitamin C or alpha arbutin serum for dark spot correction')
    elif skin_type == 'dry' or humidity < 30:
        evening_recs.append('Nourishing serum with peptides and ceramides for dry conditions')
    else:
        evening_recs.append('Retinol serum (start 1x/week, build up gradually) for skin renewal')
    
    if temp < 10:
        evening_recs.append(f'Rich night cream with shea butter for cold weather ({temp}°C) protection')
    elif humidity > 75:
        evening_recs.append(f'Lightweight night moisturizer with niacinamide for high humidity ({humidity}%)')
    else:
        evening_recs.append('Restorative night cream with peptides for overnight repair')
    
    return {
        'morning': morning_recs[:4],
        'evening': evening_recs[:4],
        'ai_generated': False,
        'source': 'Intelligent Weather Algorithm',
        'weather_adapted': True
    }

def get_ai_weather_recommendations(weather_data, skin_type=None, skin_concerns=None):
    """Generate personalized skincare recommendations"""
    try:
        return get_intelligent_fallback_recommendations(weather_data, skin_type, skin_concerns)
    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        return get_intelligent_fallback_recommendations(weather_data, skin_type, skin_concerns)

def get_enhanced_weather_data(city):
    """Get comprehensive weather data including UV index"""
    if not OPENWEATHER_API_KEY:
        return None
        
    city = city.strip().title()
    city_mappings = {
        'New Delhi': 'Delhi',
        'Busan': 'Busan,KR',
        'Seoul': 'Seoul,KR',
        'Tokyo': 'Tokyo,JP',
        'New York': 'New York,US',
        'London': 'London,GB'
    }
    
    api_city = city_mappings.get(city, city)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={api_city}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            weather_data = response.json()
            
            # Try to get UV index data
            try:
                lat = weather_data['coord']['lat']
                lon = weather_data['coord']['lon']
                uv_url = f"https://api.openweathermap.org/data/2.5/uvi?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
                uv_response = requests.get(uv_url, timeout=5)
                if uv_response.status_code == 200:
                    uv_data = uv_response.json()
                    weather_data['uvi'] = uv_data.get('value', 5)
                else:
                    weather_data['uvi'] = 5
            except:
                weather_data['uvi'] = 5
                
            return weather_data
        elif response.status_code == 404 and ',' in api_city:
            simple_city = api_city.split(',')[0]
            simple_url = f"https://api.openweathermap.org/data/2.5/weather?q={simple_city}&appid={OPENWEATHER_API_KEY}&units=metric"
            retry_response = requests.get(simple_url, timeout=10)
            if retry_response.status_code == 200:
                weather_data = retry_response.json()
                weather_data['uvi'] = 5
                return weather_data
        return None
            
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_cancer_detection(img):
    img = cv2.resize(img, SKIN_CANCER_IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Run diagnostics on startup
print("\n" + "="*60)
print("🚀 SKINSENSE APPLICATION STARTUP")
print("="*60)

# Run ML diagnostics
ml_status = diagnose_ml_setup()
print(f"\n📊 ML Status: {'✅ Ready' if ml_status else '❌ Issues Found'}")

current_memory = check_memory_usage()
print(f"💾 Current memory usage: {current_memory:.1f}MB")

# Routes
@app.route('/')
def index():
    return render_template('index.html', ml_enabled=ENABLE_ML, ollama_enabled=OLLAMA_ENABLED)

@app.route('/ml-debug')
def ml_debug():
    """ML debugging endpoint"""
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'ENABLE_ML': ENABLE_ML,
            'MEMORY_LIMIT_MB': MEMORY_LIMIT_MB,
            'SKIN_CANCER_MODEL_PATH': SKIN_CANCER_MODEL_PATH,
            'GOOGLE_DRIVE_FILE_ID': GOOGLE_DRIVE_FILE_ID
        },
        'environment': {
            'model_file_exists': os.path.exists(SKIN_CANCER_MODEL_PATH),
            'model_file_size_mb': os.path.getsize(SKIN_CANCER_MODEL_PATH) / (1024*1024) if os.path.exists(SKIN_CANCER_MODEL_PATH) else 0,
            'tensorflow_available': tensorflow_available,
            'ml_enabled': ML_ENABLED,
            'ml_error': ML_ERROR_MESSAGE,
            'current_memory_mb': check_memory_usage()
        },
        'system': {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'environment_variables': {
                'ENABLE_ML': os.environ.get('ENABLE_ML', 'not set'),
                'MEMORY_LIMIT_MB': os.environ.get('MEMORY_LIMIT_MB', 'not set'),
                'SKIN_CANCER_MODEL_PATH': os.environ.get('SKIN_CANCER_MODEL_PATH', 'not set')
            }
        },
        'quick_fixes': [
            "Set environment variable: ENABLE_ML=true",
            "Install dependencies: pip install tensorflow keras pillow opencv-python psutil",
            f"Download model to: {SKIN_CANCER_MODEL_PATH}",
            f"Download URL: https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view"
        ]
    }
    
    # Try to get TensorFlow info
    try:
        import tensorflow as tf
        debug_info['tensorflow'] = {
            'version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'gpu_count': len(tf.config.list_physical_devices('GPU'))
        }
    except ImportError:
        debug_info['tensorflow'] = {'error': 'TensorFlow not installed'}
    
    return jsonify(debug_info)

@app.route('/weather', methods=['GET', 'POST'])
def weather_recommendations():
    recommendation = None
    error = None
    
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            weather_data = get_enhanced_weather_data(city)
            if weather_data:
                skin_type = session.get('skin_type')
                skin_concerns = session.get('skin_concerns', [])
                
                recommendation = get_ai_weather_recommendations(
                    weather_data, 
                    skin_type=skin_type,
                    skin_concerns=skin_concerns
                )
                
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                weather_condition = weather_data['weather'][0]['description']
                uv_index = weather_data.get('uvi', 'N/A')
                
                return render_template('weather.html', 
                                    recommendation=recommendation,
                                    city=city,
                                    temperature=temperature,
                                    humidity=humidity,
                                    weather_condition=weather_condition,
                                    uv_index=uv_index,
                                    ai_powered=recommendation.get('ai_generated', False),
                                    source=recommendation.get('source', 'Unknown'))
            else:
                error = "City not found. Please check the spelling and try again."
    
    return render_template('weather.html', error=error)

@app.route('/quiz')
def skin_quiz():
    return render_template('quiz.html')

@app.route('/quiz/result', methods=['POST'])
def quiz_result():
    answers = request.form.to_dict()
    
    scores = {
        'dry': 0, 'oily': 0, 'combination': 0, 'sensitive': 0, 'normal': 0
    }
    
    # Process quiz answers (simplified for memory optimization)
    if 'q1' in answers:
        if answers['q1'] == 'tight':
            scores['dry'] += 2
        elif answers['q1'] == 'oily':
            scores['oily'] += 2
        elif answers['q1'] == 'fine':
            scores['normal'] += 2
            
    if 'q2' in answers:
        if answers['q2'] == 'large':
            scores['oily'] += 2
        elif answers['q2'] == 'small':
            scores['normal'] += 1
            scores['dry'] += 1
        elif answers['q2'] == 'mixed':
            scores['combination'] += 2
    
    skin_concerns = []
    if 'concerns' in answers:
        concerns_input = answers['concerns']
        if isinstance(concerns_input, list):
            skin_concerns = concerns_input
        else:
            skin_concerns = [concerns_input]
            
    skin_type = max(scores, key=scores.get)
    
    session['skin_type'] = skin_type
    session['skin_concerns'] = skin_concerns
    
    return render_template('quiz_result.html', 
                          skin_type=skin_type,
                          info=SKIN_TYPES[skin_type],
                          concerns=skin_concerns)

@app.route('/remedies')
def remedies():
    return render_template('remedies.html', conditions=SKIN_CONDITIONS)

@app.route('/remedies/<condition>')
def condition_detail(condition):
    if condition in SKIN_CONDITIONS:
        return render_template('condition_detail.html', 
                              condition=condition,
                              info=SKIN_CONDITIONS[condition])
    return redirect(url_for('remedies'))

@app.route('/cancer-predict', methods=['GET', 'POST'])
def cancer_predict():
    if request.method == 'GET':
        # Show the upload form with current ML status
        return render_template('predict.html', 
                             ml_enabled=ENABLE_ML, 
                             model_loaded=ML_ENABLED,
                             error=ML_ERROR_MESSAGE,
                             model_path=SKIN_CANCER_MODEL_PATH,
                             google_drive_id=GOOGLE_DRIVE_FILE_ID)
    
    # POST request - handle file upload and prediction
    if not ENABLE_ML:
        flash('Skin cancer detection is disabled. Set ENABLE_ML=true to enable.')
        return render_template('predict.html', 
                             ml_enabled=False, 
                             model_loaded=False,
                             error="ML features disabled by environment variable",
                             model_path=SKIN_CANCER_MODEL_PATH,
                             google_drive_id=GOOGLE_DRIVE_FILE_ID)
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"🖼️  Image saved to: {filepath}")
            
            # Load and validate image
            img_array = cv2.imread(filepath)
            if img_array is None:
                flash('Invalid image file. Please upload a valid PNG, JPG, or JPEG image.')
                return redirect(request.url)
            
            # Try to load model
            model = load_model_on_demand()
            
            if model is not None:
                try:
                    print(f"🔬 Processing image for prediction...")
                    
                    # Preprocess image
                    processed_img = preprocess_image_for_cancer_detection(img_array)
                    print(f"📐 Image preprocessed to shape: {processed_img.shape}")
                    
                    # Make prediction
                    print(f"🤖 Running model prediction...")
                    pred = model.predict(processed_img, verbose=0)
                    
                    # Process prediction results
                    raw_probability = float(pred[0][0])
                    threshold = 0.5
                    label = 'Cancer' if raw_probability > threshold else 'Not Cancer'
                    confidence = raw_probability if label == 'Cancer' else (1 - raw_probability)
                    
                    # Create detailed prediction info
                    prediction_info = {
                        'label': label,
                        'raw_probability': raw_probability,
                        'confidence': confidence,
                        'threshold': threshold,
                        'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'model_info': {
                            'input_shape': str(model.input_shape),
                            'output_shape': str(model.output_shape)
                        }
                    }
                    
                    print(f"✅ Prediction complete:")
                    print(f"   Label: {label}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Raw probability: {raw_probability:.3f}")
                    
                    return render_template('predict.html', 
                                           image_path=filepath.replace('static/', ''), 
                                           prediction=prediction_info,
                                           ml_enabled=True,
                                           model_loaded=True,
                                           success=True)
                                           
                except Exception as e:
                    error_msg = f"Error during prediction: {str(e)}"
                    print(f"❌ {error_msg}")
                    print("📋 Full error traceback:")
                    traceback.print_exc()
                    
                    flash(f'Prediction failed: {error_msg}')
                    return render_template('predict.html', 
                                           ml_enabled=ENABLE_ML,
                                           model_loaded=False,
                                           error=error_msg,
                                           model_path=SKIN_CANCER_MODEL_PATH,
                                           google_drive_id=GOOGLE_DRIVE_FILE_ID)
            else:
                flash(f'Model unavailable: {ML_ERROR_MESSAGE}')
                return render_template('predict.html', 
                                       ml_enabled=ENABLE_ML,
                                       model_loaded=False,
                                       error=ML_ERROR_MESSAGE,
                                       model_path=SKIN_CANCER_MODEL_PATH,
                                       google_drive_id=GOOGLE_DRIVE_FILE_ID)
                                       
        except Exception as e:
            error_msg = f"Error processing upload: {str(e)}"
            print(f"❌ {error_msg}")
            flash(error_msg)
            return redirect(request.url)
    else:
        flash('Invalid file type. Please upload PNG, JPG, or JPEG images only.')
        return redirect(request.url)

@app.route('/test-ml')
def test_ml():
    """Test ML functionality endpoint"""
    if not ENABLE_ML:
        return jsonify({
            'status': 'disabled',
            'message': 'ML disabled by environment variable',
            'fix': 'Set ENABLE_ML=true'
        })
    
    try:
        model = load_model_on_demand()
        if model is not None:
            # Create a dummy test image
            import numpy as np
            test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            # Test prediction
            pred = model.predict(test_img, verbose=0)
            
            return jsonify({
                'status': 'success',
                'message': 'ML model is working correctly',
                'test_prediction': float(pred[0][0]),
                'model_loaded': True,
                'tensorflow_available': tensorflow_available
            })
        else:
            return jsonify({
                'status': 'error',
                'message': ML_ERROR_MESSAGE or 'Model failed to load',
                'model_loaded': False
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'ML test failed: {str(e)}',
            'traceback': traceback.format_exc()
        })

@app.route('/download-model')
def download_model():
    """Provide model download instructions"""
    instructions = {
        'model_path': SKIN_CANCER_MODEL_PATH,
        'google_drive_id': GOOGLE_DRIVE_FILE_ID,
        'download_methods': {
            'manual': f'https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view',
            'wget': f'wget --no-check-certificate "https://docs.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}" -O {SKIN_CANCER_MODEL_PATH}',
            'curl': f'curl -L "https://docs.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}" -o {SKIN_CANCER_MODEL_PATH}'
        },
        'verification': {
            'check_file': f'ls -la {SKIN_CANCER_MODEL_PATH}',
            'file_should_exist': os.path.exists(SKIN_CANCER_MODEL_PATH),
            'current_file_size': os.path.getsize(SKIN_CANCER_MODEL_PATH) / (1024*1024) if os.path.exists(SKIN_CANCER_MODEL_PATH) else 0
        }
    }
    
    return jsonify(instructions)

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """API endpoint for getting skincare recommendations"""
    try:
        data = request.get_json()
        city = data.get('city')
        skin_type = data.get('skin_type')
        skin_concerns = data.get('skin_concerns', [])
        
        if not city:
            return jsonify({'error': 'City is required'}), 400
            
        weather_data = get_enhanced_weather_data(city)
        if not weather_data:
            return jsonify({'error': 'Weather data not available'}), 404
            
        recommendations = get_ai_weather_recommendations(
            weather_data, 
            skin_type=skin_type, 
            skin_concerns=skin_concerns
        )
        
        response_data = {
            'city': city,
            'weather': {
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'condition': weather_data['weather'][0]['description'],
                'uv_index': weather_data.get('uvi', 'N/A')
            },
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    current_memory = check_memory_usage()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_mb': current_memory,
        'memory_limit_mb': MEMORY_LIMIT_MB,
        'memory_usage_percent': (current_memory / MEMORY_LIMIT_MB) * 100 if MEMORY_LIMIT_MB > 0 else 0,
        'ml_configuration': {
            'enabled': ENABLE_ML,
            'model_loaded': ML_ENABLED,
            'model_path': SKIN_CANCER_MODEL_PATH,
            'model_exists': os.path.exists(SKIN_CANCER_MODEL_PATH),
            'tensorflow_available': tensorflow_available,
            'error': ML_ERROR_MESSAGE
        },
        'services': {
            'web_app': {'healthy': True},
            'weather_api': {'healthy': bool(OPENWEATHER_API_KEY)},
            'ml_model': {
                'enabled': ENABLE_ML,
                'loaded': ML_ENABLED,
                'memory_optimized': True,
                'status': 'ready' if ML_ENABLED else 'not_loaded',
                'error': ML_ERROR_MESSAGE
            }
        },
        'quick_fixes': [
            f"Set ENABLE_ML=true (currently: {os.environ.get('ENABLE_ML', 'not set')})",
            f"Download model to: {SKIN_CANCER_MODEL_PATH}",
            "Install: pip install tensorflow keras pillow opencv-python psutil",
            f"Model download: https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view"
        ] if not ML_ENABLED else []
    })

@app.route('/clear-session')
def clear_session():
    session.clear()
    flash('Session cleared')
    return redirect(url_for('index'))

@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = secrets.token_hex(8)

@app.context_processor
def inject_user_info():
    return {
        'user_skin_type': session.get('skin_type'),
        'user_concerns': session.get('skin_concerns', []),
        'ml_enabled': ENABLE_ML,
        'ml_loaded': ML_ENABLED,
        'ml_error': ML_ERROR_MESSAGE,
        'ollama_enabled': OLLAMA_ENABLED,
        'now': datetime.now(),
        'datetime': datetime
    }

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🚀 SkinSense Application Starting...")
    print(f"📍 Port: {port}")
    print(f"💾 Memory usage: {check_memory_usage():.1f}MB / {MEMORY_LIMIT_MB}MB")
    print(f"🤖 ML enabled: {ENABLE_ML}")
    print(f"🔬 ML loaded: {ML_ENABLED}")
    if ML_ERROR_MESSAGE:
        print(f"⚠️  ML Error: {ML_ERROR_MESSAGE}")
    
    print(f"\n🔗 Available endpoints:")
    print(f"   http://localhost:{port}/ - Main application")
    print(f"   http://localhost:{port}/health - Health check")
    print(f"   http://localhost:{port}/ml-debug - ML diagnostics")
    print(f"   http://localhost:{port}/test-ml - Test ML functionality")
    print(f"   http://localhost:{port}/download-model - Model download info")
    
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

