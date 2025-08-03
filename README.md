# 🌿 SkinSense

**AI-Powered Skincare Platform**

SkinSense is a full-stack web application that provides smart, personalized skincare analysis and product recommendations using AI and Generative AI. Users can upload images for skin condition detection, receive real-time weather-based skincare tips, and interact with an AI skincare assistant that generates genuine, non-hardcoded insights.

---

## Features

- **AI-Powered Skin Cancer Detection**
  - Detects potential skin issues using a fine-tuned MobileNetV2 deep learning model (5-fold cross-validation, early stopping; 84% precision, 86% recall).
  - Real-time inference powered by Flask API.

- **Dynamic Weather-Based Skincare Suggestions (GenAI)**
  - Fetches real-time weather data via the OpenWeatherMap API.
  - Provides **context-aware** skincare tips based on temperature, humidity, and UV index using **Generative AI prompts**.

- **AI Skincare Assistant (GenAI)**
  - Generates personalized skincare advice using Generative AI (ChatGPT API or local Ollama models).
  - Avoids hardcoded logic; outputs change dynamically based on user input and weather conditions.

- **Personalized Product Recommendations**
  - Recommends skincare products tailored to the user’s skin type and concerns.

- **Skin Type Analysis**
  - Guides users in identifying their skin type through an interactive questionnaire.

---

## Tech Stack

**Frontend:**
- HTML, CSS
- JavaScript (Vanilla JS / React optional)
- Responsive UI (mobile-friendly)

**Backend:**
- Python (Flask)
- TensorFlow / Keras (MobileNetV2 Model)
- OpenWeatherMap API
- Generative AI API (OpenAI ChatGPT / Ollama)

---

## Screenshots 
- Weather Recommendations(Using Gen-AI)
- <img width="500" height="500" alt="Screenshot (167)" src="https://github.com/user-attachments/assets/0b5b5948-f723-4d5a-b743-0e58875f5ed4" />

-<img width="500" height="500" alt="Screenshot (168)" src="https://github.com/user-attachments/assets/0489a1b1-11b4-4941-af1d-4b77226bde19" />

- Skin-Cancer Prediction
-<img width="500" height="500" alt="Screenshot (169)" src="https://github.com/user-attachments/assets/578e8672-1ee6-44eb-b60b-d4036cbe6153" />

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/skinsense.git
cd skinsense

# Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the Flask app
python app.py
