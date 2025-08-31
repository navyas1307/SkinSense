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
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
import time
from dotenv import load_dotenv
import gdown
import re

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
SKIN_CANCER_MODEL_PATH = 'skin_cancer_model.h5'
SKIN_CANCER_IMG_SIZE = (224, 224)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disable Ollama for production deployment
OLLAMA_ENABLED = False
print("Ollama disabled for production deployment")

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
    
    print(f"Using intelligent recommendations for {temp}°C, {humidity}% humidity, {weather_condition}")
    
    # Morning routine logic
    # Step 1: Cleanser
    if skin_type == 'oily':
        morning_recs.append('Salicylic acid gel cleanser for oil control')
    elif skin_type == 'dry':
        morning_recs.append('Creamy hydrating cleanser with ceramides')
    elif skin_type == 'sensitive':
        morning_recs.append('Gentle, fragrance-free cream cleanser')
    else:
        morning_recs.append('Gentle foaming cleanser for daily use')
    
    # Step 2: Treatment serum
    if temp > 25 and humidity > 70:
        morning_recs.append('Niacinamide serum to control oil and minimize pores in humid weather')
    elif temp < 15 or humidity < 40:
        morning_recs.append('Hyaluronic acid serum for intense hydration in dry conditions')
    elif uv_index > 7:
        morning_recs.append('Vitamin C serum for enhanced antioxidant protection against high UV')
    else:
        morning_recs.append('Balanced antioxidant serum with vitamin E for daily protection')
    
    # Step 3: Moisturizer
    if humidity > 80:
        morning_recs.append('Oil-free gel moisturizer perfect for high humidity conditions')
    elif humidity < 30:
        morning_recs.append('Rich cream moisturizer with hyaluronic acid for dry air')
    elif temp > 30:
        morning_recs.append('Lightweight, cooling gel moisturizer for hot weather comfort')
    else:
        morning_recs.append('Balanced daily moisturizer with light SPF protection')
    
    # Step 4: Sun protection
    if uv_index > 8:
        morning_recs.append(f'SPF 50+ broad spectrum sunscreen (UV index: {uv_index} - very high), reapply every 2 hours')
    elif uv_index > 5:
        morning_recs.append(f'SPF 30+ mineral sunscreen with zinc oxide (UV index: {uv_index} - moderate to high)')
    else:
        morning_recs.append('SPF 30 daily sunscreen with moisturizing benefits')
    
    # Step 5: Weather-specific protection
    if wind_speed > 5:
        morning_recs.append(f'Barrier cream or balm for wind protection (wind speed: {wind_speed} m/s)')
    elif 'rain' in weather_condition:
        morning_recs.append('Water-resistant sunscreen and setting spray for rainy conditions')
    elif temp < 5:
        morning_recs.append('Rich facial oil for extreme cold protection and barrier repair')
    else:
        morning_recs.append('Light facial mist for hydration throughout the day')
    
    # Step 6: Special care
    if skin_concerns and 'acne' in skin_concerns:
        morning_recs.append('Spot treatment with tea tree oil for targeted blemish care')
    elif skin_concerns and 'aging' in skin_concerns:
        morning_recs.append('Peptide serum for anti-aging and skin firming benefits')
    else:
        morning_recs.append('Eye cream with caffeine to reduce puffiness and brighten under-eyes')
    
    # Evening routine logic
    # Step 1: Double cleanse
    if humidity > 70 or 'rain' in weather_condition:
        evening_recs.append('Oil cleanser then foam cleanser (double cleanse) to remove humidity buildup')
    else:
        evening_recs.append('Micellar water followed by gentle cleanser for thorough cleansing')
    
    # Step 2: Toner/Treatment
    if temp > 25 and skin_type == 'oily':
        evening_recs.append('BHA toner (2-3 times per week) for deep pore cleansing in warm weather')
    elif temp < 15 or humidity < 40:
        evening_recs.append('Hydrating toner with hyaluronic acid for cold, dry conditions')
    else:
        evening_recs.append('Gentle pH-balancing toner to prep skin for treatments')
    
    # Step 3: Treatment serum
    if skin_concerns and 'hyperpigmentation' in skin_concerns:
        evening_recs.append('Vitamin C or alpha arbutin serum for dark spot correction')
    elif skin_type == 'dry' or humidity < 30:
        evening_recs.append('Nourishing serum with peptides and ceramides for dry conditions')
    else:
        evening_recs.append('Retinol serum (start 1x/week, build up gradually) for skin renewal')
    
    # Step 4: Moisturizer
    if temp < 10:
        evening_recs.append(f'Rich night cream with shea butter for cold weather ({temp}°C) protection')
    elif humidity > 75:
        evening_recs.append(f'Lightweight night moisturizer with niacinamide for high humidity ({humidity}%)')
    else:
        evening_recs.append('Restorative night cream with peptides for overnight repair')
    
    # Step 5: Special treatment
    if humidity < 30 or temp < 5:
        evening_recs.append('Nourishing facial oil (argan or rosehip) for extra protection against harsh conditions')
    elif skin_type == 'oily' and humidity > 80:
        evening_recs.append('Clay mask once a week to control excess oil in humid conditions')
    else:
        evening_recs.append('Weekly gentle exfoliation with AHA/BHA for cell turnover')
    
    # Step 6: Eye and lip care
    evening_recs.append('Hydrating eye cream and nourishing lip balm for overnight repair')
    
    return {
        'morning': morning_recs[:6],
        'evening': evening_recs[:6],
        'ai_generated': False,
        'source': 'Intelligent Weather Algorithm',
        'weather_adapted': True
    }

def get_ai_weather_recommendations(weather_data, skin_type=None, skin_concerns=None):
    """
    Generate personalized skincare recommendations
    """
    try:
        # Extract weather data
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        weather_condition = weather_data['weather'][0]['description']
        uv_index = weather_data.get('uvi', 5)
        
        print(f"Weather: {temp}°C, {humidity}% humidity, {weather_condition}, UV: {uv_index}")
        print(f"Skin type: {skin_type}, Concerns: {skin_concerns}")
        
        # Use intelligent fallback for production
        print("Using intelligent algorithm for recommendations")
        return get_intelligent_fallback_recommendations(weather_data, skin_type, skin_concerns)
        
    except Exception as e:
        print(f"Error in get_ai_weather_recommendations: {str(e)}")
        return get_intelligent_fallback_recommendations(weather_data, skin_type, skin_concerns)

def get_enhanced_weather_data(city):
    """
    Get comprehensive weather data including UV index
    """
    if not OPENWEATHER_API_KEY:
        print("OpenWeather API key not configured")
        return None
        
    # Clean city name and handle common variations
    city = city.strip().title()
    
    # Handle common city name variations
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
        print(f"Making weather API request for: {api_city}")
        response = requests.get(url, timeout=10)
        
        print(f"API Response status: {response.status_code}")
        
        if response.status_code == 200:
            weather_data = response.json()
            print(f"Weather data retrieved for {weather_data.get('name', city)}")
            
            # Try to get UV index data
            try:
                lat = weather_data['coord']['lat']
                lon = weather_data['coord']['lon']
                uv_url = f"https://api.openweathermap.org/data/2.5/uvi?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
                uv_response = requests.get(uv_url, timeout=5)
                if uv_response.status_code == 200:
                    uv_data = uv_response.json()
                    weather_data['uvi'] = uv_data.get('value', 5)
                    print(f"UV index retrieved: {weather_data['uvi']}")
                else:
                    weather_data['uvi'] = 5  # Default moderate UV index
                    print("UV index not available, using default")
            except Exception as e:
                weather_data['uvi'] = 5  # Default moderate UV index
                print(f"Could not fetch UV index: {str(e)}")
                
            return weather_data
        elif response.status_code == 404:
            print(f"City not found: {api_city}")
            # Try without country code if it was added
            if ',' in api_city:
                simple_city = api_city.split(',')[0]
                print(f"Retrying with simple city name: {simple_city}")
                simple_url = f"https://api.openweathermap.org/data/2.5/weather?q={simple_city}&appid={OPENWEATHER_API_KEY}&units=metric"
                retry_response = requests.get(simple_url, timeout=10)
                if retry_response.status_code == 200:
                    weather_data = retry_response.json()
                    weather_data['uvi'] = 5  # Default UV index
                    print(f"Weather data retrieved for {weather_data.get('name', simple_city)} on retry")
                    return weather_data
            return None
        elif response.status_code == 401:
            print("Invalid API key for OpenWeather")
            return None
        else:
            print(f"Weather API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Weather API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Weather API request failed: {str(e)}")
        return None

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_cancer_detection(img):
    img = cv2.resize(img, SKIN_CANCER_IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

try:
    # Load skin cancer detection model
    skin_cancer_model = load_model(SKIN_CANCER_MODEL_PATH)
    ML_ENABLED = True
    print("Skin cancer model loaded successfully")
except:
    ML_ENABLED = False
    print("Skin cancer model not found, ML features disabled")

# Routes
@app.route('/')
def index():
    return render_template('index.html', ml_enabled=ML_ENABLED, ollama_enabled=OLLAMA_ENABLED)

@app.route('/weather', methods=['GET', 'POST'])
def weather_recommendations():
    recommendation = None
    error = None
    
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            print(f"Getting weather data for: {city}")
            weather_data = get_enhanced_weather_data(city)
            if weather_data:
                # Get user's skin type and concerns from session
                skin_type = session.get('skin_type')
                skin_concerns = session.get('skin_concerns', [])
                
                print(f"Weather data retrieved: {weather_data['main']['temp']}°C, {weather_data['main']['humidity']}%")
                
                # Get AI-powered recommendations
                recommendation = get_ai_weather_recommendations(
                    weather_data, 
                    skin_type=skin_type,
                    skin_concerns=skin_concerns
                )
                
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                weather_condition = weather_data['weather'][0]['description']
                uv_index = weather_data.get('uvi', 'N/A')
                
                print(f"Recommendations generated successfully!")
                print(f"AI Generated: {recommendation.get('ai_generated', False)}")
                print(f"Source: {recommendation.get('source', 'Unknown')}")
                
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
                error = "City not found. Please check the spelling and try again. Try using just the city name (e.g., 'Delhi' instead of 'New Delhi')."
                print(f"Weather data not found for city: {city}")
    
    return render_template('weather.html', error=error)

@app.route('/quiz')
def skin_quiz():
    return render_template('quiz.html')

@app.route('/quiz/result', methods=['POST'])
def quiz_result():
    answers = request.form.to_dict()
    
    # Simple scoring system for skin type quiz
    scores = {
        'dry': 0,
        'oily': 0,
        'combination': 0,
        'sensitive': 0,
        'normal': 0
    }
    
    # Process quiz answers 
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
            
    if 'q3' in answers:
        if answers['q3'] == 'often':
            scores['sensitive'] += 2
        elif answers['q3'] == 'sometimes':
            scores['sensitive'] += 1
            
    if 'q4' in answers:
        if answers['q4'] == 'shiny':
            scores['oily'] += 2
        elif answers['q4'] == 'flaky':
            scores['dry'] += 2
        elif answers['q4'] == 'both':
            scores['combination'] += 2
            
    if 'q5' in answers:
        if answers['q5'] == 'tzone':
            scores['combination'] += 2
        elif answers['q5'] == 'all':
            scores['oily'] += 2
        elif answers['q5'] == 'none':
            scores['normal'] += 2
            scores['dry'] += 1
    
    # Additional questions for skin concerns
    skin_concerns = []
    if 'concerns' in answers:
        concerns_input = answers['concerns']
        if isinstance(concerns_input, list):
            skin_concerns = concerns_input
        else:
            skin_concerns = [concerns_input]
            
    # Find the skin type with the highest score
    skin_type = max(scores, key=scores.get)
    
    # Store in session for future use
    session['skin_type'] = skin_type
    session['skin_concerns'] = skin_concerns
    
    print(f"User skin type determined: {skin_type}")
    print(f"User skin concerns: {skin_concerns}")
    
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
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process the image
            img_array = cv2.imread(filepath)
            if img_array is None:
                flash('Invalid image file')
                return redirect(request.url)
            
            # Predict using skin cancer model
            if ML_ENABLED:
                try:
                    processed_img = preprocess_image_for_cancer_detection(img_array)
                    pred = skin_cancer_model.predict(processed_img)
                    
                    label = 'Cancer' if pred[0][0] > 0.7452 else 'Not Cancer'
                    probability = float(pred[0][0])
                    
                    print(f"Skin cancer prediction: {label} (probability: {probability:.4f})")
                    
                    return render_template('predict.html', 
                                           image_path=filepath, 
                                           label=label, 
                                           probability=probability,
                                           ml_enabled=ML_ENABLED)
                except Exception as e:
                    print(f"Error in skin cancer prediction: {str(e)}")
                    flash('Error processing image for prediction')
                    return redirect(request.url)
            else:
                flash('Skin cancer detection model is not available')
                return redirect(request.url)
    
    return render_template('predict.html', ml_enabled=ML_ENABLED)

# API endpoint for getting personalized recommendations
@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """
    API endpoint for getting AI-powered skincare recommendations
    """
    try:
        data = request.get_json()
        city = data.get('city')
        skin_type = data.get('skin_type')
        skin_concerns = data.get('skin_concerns', [])
        
        if not city:
            return jsonify({'error': 'City is required'}), 400
            
        print(f"API request for city: {city}, skin_type: {skin_type}")
            
        weather_data = get_enhanced_weather_data(city)
        if not weather_data:
            return jsonify({'error': 'Weather data not available for this city'}), 404
            
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
        
        print(f"API response generated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_enabled': ML_ENABLED,
        'ai_services': {
            'ollama': {
                'enabled': OLLAMA_ENABLED,
                'healthy': False
            }
        }
    })

# Clear session route
@app.route('/clear-session')
def clear_session():
    """
    Clear user session data
    """
    session.clear()
    flash('Session data cleared successfully')
    return redirect(url_for('index'))

# Before request handler to create session ID for new users
@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = secrets.token_hex(8)
        print(f"New user session created: {session['user_id']}")

# Enhanced context processor to make certain variables available in all templates
@app.context_processor
def inject_user_info():
    return {
        'user_skin_type': session.get('skin_type'),
        'user_concerns': session.get('skin_concerns', []),
        'ml_enabled': ML_ENABLED,
        'ollama_enabled': OLLAMA_ENABLED,
        'now': datetime.now(),
        'datetime': datetime
    }

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting SkinSense Application...")
    print(f"OpenWeather API: {'Configured' if OPENWEATHER_API_KEY else 'Missing'}")
    print(f"Ollama: {'Enabled' if OLLAMA_ENABLED else 'Disabled'}")
    print(f"ML Model: {'Loaded' if ML_ENABLED else 'Not available'}")
    print(f"Port: {port}")
    print("=" * 50)
    
    # For production (like Render), don't use debug mode
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

