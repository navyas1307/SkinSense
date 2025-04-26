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
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
OPENWEATHER_API_KEY = '2f021261bac8c9f5f35de84b6486589e' 
SKIN_CANCER_MODEL_PATH = 'skin_cancer_model.h5'
SKIN_CANCER_IMG_SIZE = (224, 224)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    },
    'psoriasis': {
        'description': 'Chronic autoimmune condition causing rapid skin cell growth, resulting in thick, scaly patches of skin.',
        'ingredients': ['Salicylic Acid', 'Coal Tar', 'Aloe Vera', 'Vitamin D', 'Zinc'],
        'home_remedies': [
            {'name': 'Aloe Vera Gel', 'usage': 'Apply pure aloe vera gel to affected areas'},
            {'name': 'Epsom Salt Bath', 'usage': 'Soak in warm bath with Epsom salts for 15 minutes'},
            {'name': 'Coconut Oil', 'usage': 'Gently massage coconut oil to reduce scaling and itching'}
        ],
        'image': 'static/images/conditions/psoriasis.jpg'
    },
    'melasma': {
        'description': 'Hyperpigmentation condition causing brown to gray-brown patches, commonly triggered by hormonal changes.',
        'ingredients': ['Vitamin C', 'Kojic Acid', 'Tranexamic Acid', 'Niacinamide', 'Alpha Arbutin'],
        'home_remedies': [
            {'name': 'Turmeric Mask', 'usage': 'Mix turmeric with yogurt and apply for 15-20 minutes'},
            {'name': 'Apple Cider Vinegar', 'usage': 'Dilute and use as a spot treatment'},
            {'name': 'Aloe Vera', 'usage': 'Apply fresh gel to affected areas daily'}
        ],
        'image': 'static/images/conditions/melasma.jpeg'
    },
    'fungal acne': {
        'description': 'Caused by an overgrowth of yeast, resembling acne but requiring different treatment approach.',
        'ingredients': ['Tea Tree Oil', 'Zinc Pyrithione', 'Ketoconazole', 'Sulfur', 'Salicylic Acid'],
        'home_remedies': [
            {'name': 'Apple Cider Vinegar Toner', 'usage': 'Dilute and apply as a daily toner'},
            {'name': 'Garlic Paste', 'usage': 'Apply diluted garlic paste for its antifungal properties'},
            {'name': 'Honey Mask', 'usage': 'Apply raw honey with a pinch of turmeric for 15 minutes'}
        ],
        'image': 'static/images/conditions/fungal_acne.png'
    },
    'Contact Dermatitis': {
        'description': 'Inflammatory reaction resulting in redness, itching, and sometimes blisters.',
        'ingredients': ['Colloidal Oatmeal', 'Calendula', 'Chamomile', 'Panthenol', 'Ceramides'],
        'home_remedies': [
            {'name': 'Cool Compress', 'usage': 'Apply cold chamomile tea compress to affected areas'},
            {'name': 'Oatmeal Bath', 'usage': 'Soak in lukewarm bath with colloidal oatmeal'},
            {'name': 'Aloe Vera Gel', 'usage': 'Apply pure aloe vera to soothe inflammation'}
        ],
        'image': 'static/images/conditions/Contact_Dermatitis.jpg'
    },
    'Seborrheic Dermatitis': {
        'description': 'Scaly patches, red skin, and stubborn dandruff, primarily affecting oily areas of the body.',
        'ingredients': ['Zinc Pyrithione', 'Salicylic Acid', 'Tea Tree Oil', 'Green Tea', 'Niacinamide'],
        'home_remedies': [
            {'name': 'Tea Tree Oil Treatment', 'usage': 'Dilute and apply to affected areas'},
            {'name': 'Apple Cider Vinegar Rinse', 'usage': 'Use as a diluted scalp treatment'},
            {'name': 'Coconut Oil', 'usage': 'Gentle massage to reduce inflammation and scaling'}
        ],
        'image': 'static/images/conditions/Seborrheic_Dermatitis.jpeg'
    },
    'Perioral Dermatitis': {
        'description': 'Characterized by small red bumps and potential burning sensation.',
        'ingredients': ['Azelaic Acid', 'Sulfur', 'Zinc', 'Green Tea Extract', 'Centella Asiatica'],
        'home_remedies': [
            {'name': 'Green Tea Compress', 'usage': 'Apply cooled green tea bags to affected areas'},
            {'name': 'Honey Mask', 'usage': 'Apply raw honey for 15-20 minutes'},
            {'name': 'Aloe Vera Gel', 'usage': 'Apply pure aloe vera to reduce inflammation'}
        ],
        'image': 'static/images/conditions/Perioral_Dermatitis.jpg'
    },
    'Keratosis Pilaris': {
        'description': 'Small, rough bumps typically on arms, thighs, and cheeks due to keratin buildup.',
        'ingredients': ['Lactic Acid', 'Glycolic Acid', 'Urea', 'Vitamin A', 'Ceramides'],
        'home_remedies': [
            {'name': 'Sugar Scrub', 'usage': 'Gentle exfoliation with sugar and olive oil'},
            {'name': 'Coconut Oil', 'usage': 'Daily moisturizing to soften skin texture'},
            {'name': 'Baking Soda Scrub', 'usage': 'Mild exfoliation mixed with gentle moisturizer'}
        ],
        'image': 'static/images/conditions/keratosis_pilaris.jpg'
    }
}

for condition, info in SKIN_CONDITIONS.items():
    if 'category' not in info:
        # Assign default categories based on condition type
        if condition in ['acne', 'fungal acne']:
            info['category'] = 'Acne'
        elif condition in ['eczema', 'Contact Dermatitis', 'Seborrheic Dermatitis', 'Perioral Dermatitis']:
            info['category'] = 'Dermatitis'
        elif condition in ['psoriasis', 'Keratosis Pilaris']:
            info['category'] = 'Skin Disorders'
        elif condition in ['hyperpigmentation', 'melasma']:
            info['category'] = 'Pigmentation'
        elif condition in ['dryness', 'rosacea', 'sunburn', 'aging']:
            info['category'] = 'Skin Care'
        else:
            info['category'] = 'Skin Care'
# Update your SKIN_CONDITIONS dictionary
for condition, info in SKIN_CONDITIONS.items():
    # Add image path if it doesn't exist
    if 'image' not in info:
        # Create standardized image path
        image_filename = condition.lower().replace(' ', '_') + '.jpg'
        info['image'] = f'static/images/conditions/{image_filename}'
            
        
            


# Weather-based skincare recommendations
def get_weather_recommendations(weather_data):
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    weather_condition = weather_data['weather'][0]['main'].lower()
    uv_index = weather_data.get('uvi', 5)  # Default to moderate if not available
    
    recommendations = {
        'morning': [],
        'evening': []
    }
    
    # Base recommendations for all weather
    recommendations['morning'].append('Cleanser')
    recommendations['evening'].append('Cleanser')
    recommendations['evening'].append('Moisturizer')
    
    # Temperature based recommendations
    if temp > 30:  # Hot weather
        recommendations['morning'].append('Light moisturizer')
        recommendations['morning'].append('Sunscreen SPF 50+')
        recommendations['morning'].append('Antioxidant serum')
        recommendations['evening'].append('Cooling gel')
    elif temp < 10:  # Cold weather
        recommendations['morning'].append('Rich moisturizer')
        recommendations['morning'].append('Sunscreen SPF 30+')
        recommendations['evening'].append('Hydrating serum')
        recommendations['evening'].append('Rich night cream')
    else:  # Moderate temperature
        recommendations['morning'].append('Moisturizer')
        recommendations['morning'].append('Sunscreen SPF 30+')
        recommendations['evening'].append('Night cream')
    
    # Humidity based recommendations
    if humidity > 70:  # High humidity
        recommendations['morning'].append('Oil-control primer')
        recommendations['evening'].append('Clay mask (2-3 times a week)')
    elif humidity < 30:  # Low humidity
        recommendations['morning'].append('Hydrating serum')
        recommendations['evening'].append('Overnight hydrating mask (1-2 times a week)')
    
    # Weather condition based recommendations
    if 'rain' in weather_condition or 'drizzle' in weather_condition:
        recommendations['morning'].append('Waterproof sunscreen')
    elif 'snow' in weather_condition:
        recommendations['morning'].append('Barrier cream')
        recommendations['evening'].append('Extra moisturizer')
    elif 'clear' in weather_condition:
        recommendations['morning'].append('Antioxidant serum')
    
    # UV index based recommendations
    if uv_index >= 8:  # High UV
        recommendations['morning'].append('Sunscreen SPF 50+')
        recommendations['morning'].append('UV protective clothing')
        recommendations['evening'].append('Aloe vera gel')
    elif uv_index >= 3:  # Moderate UV
        recommendations['morning'].append('Sunscreen SPF 30+')
    
    return recommendations

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def preprocess_image_for_cancer_detection(img):
    img = cv2.resize(img, SKIN_CANCER_IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

try:
    # Load skin cancer detection model
    skin_cancer_model = load_model(SKIN_CANCER_MODEL_PATH)
    ML_ENABLED = True
except:
    ML_ENABLED = False

# Routes
@app.route('/')
def index():
    return render_template('index.html', ml_enabled=ML_ENABLED)

@app.route('/weather', methods=['GET', 'POST'])
def weather_recommendations():
    recommendation = None
    error = None
    
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            weather_data = get_weather_data(city)
            if weather_data:
                recommendation = get_weather_recommendations(weather_data)
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                weather_condition = weather_data['weather'][0]['description']
                
                return render_template('weather.html', 
                                    recommendation=recommendation,
                                    city=city,
                                    temperature=temperature,
                                    humidity=humidity,
                                    weather_condition=weather_condition)
            else:
                error = "City not found. Please try again."
    
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
            
    # Find the skin type with the highest score
    skin_type = max(scores, key=scores.get)
    
    # Store in session for future use
    session['skin_type'] = skin_type
    
    return render_template('quiz_result.html', 
                          skin_type=skin_type,
                          info=SKIN_TYPES[skin_type])

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
                processed_img = preprocess_image_for_cancer_detection(img_array)
                pred = skin_cancer_model.predict(processed_img)
                
                label = 'Cancer' if pred[0][0] > 0.7452 else 'Not Cancer'
                probability = float(pred[0][0])
                
                return render_template('predict.html', 
                                       image_path=filepath, 
                                       label=label, 
                                       probability=probability,
                                       ml_enabled=ML_ENABLED)
            else:
                flash('Skin cancer detection model is not available')
                return redirect(request.url)
    
    return render_template('predict.html', ml_enabled=ML_ENABLED)

# Error handling routes
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Before request handler to create session ID for new users
@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = secrets.token_hex(8)

if __name__ == '__main__':
    app.run(debug=True)
