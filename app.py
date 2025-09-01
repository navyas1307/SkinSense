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

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# API Keys
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '2f021261bac8c9f5f35de84b6486589e')

# Model Configuration - DISABLE FOR MEMORY OPTIMIZATION
SKIN_CANCER_MODEL_PATH = 'skin_cancer_model.h5'
SKIN_CANCER_IMG_SIZE = (224, 224)
GOOGLE_DRIVE_FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID', '1Mt2Xvx--d04qxP-rrrfZcjAsj8RN_IPN')

# Memory optimization: Lazy import TensorFlow only when needed
tensorflow_available = False
skin_cancer_model = None
ML_ENABLED = False

# Check memory constraints - disable ML on low memory environments
MEMORY_LIMIT_MB = int(os.environ.get('MEMORY_LIMIT_MB', '512'))
ENABLE_ML = os.environ.get('ENABLE_ML', 'false').lower() == 'true'

print(f"Memory limit: {MEMORY_LIMIT_MB}MB")
print(f"ML explicitly enabled: {ENABLE_ML}")

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
        return 0

def lazy_import_tensorflow():
    """Lazy import TensorFlow to save memory"""
    global tensorflow_available
    if not tensorflow_available:
        try:
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
            
            tensorflow_available = True
            return True
        except ImportError as e:
            print(f"TensorFlow not available: {e}")
            return False
    return True

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

def load_model_on_demand():
    """Load model only when needed to save memory"""
    global skin_cancer_model, ML_ENABLED
    
    if not ENABLE_ML:
        print("ML disabled by environment variable")
        return None
        
    current_memory = check_memory_usage()
    if current_memory > MEMORY_LIMIT_MB * 0.8:  # 80% of limit
        print(f"Memory usage too high ({current_memory:.1f}MB), skipping ML model")
        return None
    
    if not lazy_import_tensorflow():
        return None
        
    if skin_cancer_model is None:
        try:
            import tensorflow as tf
            from keras.models import load_model
            
            if os.path.exists(SKIN_CANCER_MODEL_PATH):
                print("Loading model on demand...")
                skin_cancer_model = load_model(SKIN_CANCER_MODEL_PATH, compile=False)
                skin_cancer_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                ML_ENABLED = True
                print(f"Model loaded successfully, memory usage: {check_memory_usage():.1f}MB")
            else:
                print("Model file not found")
                return None
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    return skin_cancer_model

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

# Initialize with memory optimization
print("Initializing SkinSense Application...")
print(f"Memory optimization enabled - ML features: {'enabled' if ENABLE_ML else 'disabled'}")
print(f"Current memory usage: {check_memory_usage():.1f}MB")

# Routes
@app.route('/')
def index():
    return render_template('index.html', ml_enabled=ENABLE_ML, ollama_enabled=OLLAMA_ENABLED)

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
    if not ENABLE_ML:
        flash('Skin cancer detection is currently disabled to optimize memory usage.')
        return render_template('predict.html', ml_enabled=False, model_loaded=False,
                             error="ML features disabled for memory optimization")
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img_array = cv2.imread(filepath)
            if img_array is None:
                flash('Invalid image file')
                return redirect(request.url)
            
            model = load_model_on_demand()
            
            if model is not None:
                try:
                    processed_img = preprocess_image_for_cancer_detection(img_array)
                    pred = model.predict(processed_img, verbose=0)
                    
                    raw_probability = float(pred[0][0])
                    threshold = 0.5
                    label = 'Cancer' if raw_probability > threshold else 'Not Cancer'
                    confidence = raw_probability if label == 'Cancer' else (1 - raw_probability)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    return render_template('predict.html', 
                                           image_path=filepath, 
                                           label=label, 
                                           probability=raw_probability,
                                           confidence=confidence,
                                           threshold=threshold,
                                           ml_enabled=True,
                                           model_loaded=True,
                                           timestamp=timestamp)
                                           
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    flash('Error processing image. Please try again.')
                    return redirect(request.url)
            else:
                flash('Model temporarily unavailable due to memory constraints.')
                return render_template('predict.html', 
                                       ml_enabled=ENABLE_ML,
                                       model_loaded=False,
                                       error="Model unavailable - memory optimization active")
    
    return render_template('predict.html', ml_enabled=ENABLE_ML, model_loaded=False)

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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_mb': check_memory_usage(),
        'memory_limit_mb': MEMORY_LIMIT_MB,
        'ml_enabled': ENABLE_ML,
        'services': {
            'web_app': {'healthy': True},
            'weather_api': {'healthy': bool(OPENWEATHER_API_KEY)},
            'ml_model': {
                'enabled': ENABLE_ML,
                'loaded': skin_cancer_model is not None,
                'memory_optimized': True
            }
        }
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
        'ollama_enabled': OLLAMA_ENABLED,
        'now': datetime.now(),
        'datetime': datetime
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\nSkinSense Application Starting...")
    print(f"Memory usage: {check_memory_usage():.1f}MB")
    print(f"ML enabled: {ENABLE_ML}")
    print(f"Port: {port}")
    
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

