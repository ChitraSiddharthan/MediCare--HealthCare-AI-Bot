
import os
import gradio as gr
import google.generativeai as genai
from datetime import datetime
import pandas as pd # Keep pandas import although not directly used in the final version, might be useful for future file processing
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting in backend environments
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import json
import time
import re
import random
import logging

# --- Configuration & Setup ---

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API using environment variable
try:
    # Best practice: Load API key from environment variable
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logging.warning("GOOGLE_API_KEY environment variable not set. Using placeholder.")
        # Use a placeholder or raise an error depending on desired behavior
        # For demonstration, we'll allow it to proceed but log a warning.
        # raise ValueError("GOOGLE_API_KEY environment variable not set.")
        GOOGLE_API_KEY = "YOUR_API_KEY_HERE" # Replace with your key ONLY for local testing if necessary

    # Check if the key is still the placeholder
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or GOOGLE_API_KEY == "your_gemini_api_key_here":
         logging.warning("Using a placeholder API key. Please set the GOOGLE_API_KEY environment variable.")
         # Optionally, disable API calls if using placeholder
         # model = None

    genai.configure(api_key=GOOGLE_API_KEY)
    # Set up Gemini Flash model
    model = genai.GenerativeModel('gemini-flash')
    logging.info("Gemini API configured successfully.")

except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}")
    model = None # Ensure model is None if configuration fails


# --- Custom CSS ---
# (CSS remains the same as provided, it's extensive and generally well-structured)
custom_css = """
:root {
    --primary-color: #0069b3;
    --secondary-color: #6ac6ff;
    --accent-color: #ff6a6a;
    --text-color: #333333;
    --light-bg: #f3f8ff;
    --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    --success-color: #28a745;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

body {
    font-family: var(--font-main);
    color: var(--text-color);
    background-color: var(--light-bg);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

.health-header {
    background-color: var(--primary-color);
    color: white;
    padding: 1rem;
    border-radius: 10px 10px 0 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.health-header h1 {
    margin: 0;
    font-size: 1.8rem;
}

.health-logo {
    width: 40px;
    height: 40px;
    margin-right: 1rem;
}

.chat-container {
    border-radius: 10px;
    background-color: white;
    box-shadow: var(--card-shadow);
    overflow: hidden;
    transition: all 0.3s ease;
}

.chat-container:hover {
    box-shadow: 0 10px 15px rgba(0, 0, 0, 0.15);
}

/* Add styles for Gradio's default chatbot messages */
.gradio-chatbot .message.user {
    background-color: #e1f5fe;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    float: right; /* Ensure user messages align right */
    clear: both; /* Prevent overlap */
}

.gradio-chatbot .message.bot {
    background-color: white;
    border: 1px solid #e0e0e0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-start;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    float: left; /* Ensure bot messages align left */
    clear: both; /* Prevent overlap */
}

/* Override default message text alignment if needed */
.gradio-chatbot .message.user p { text-align: right; }
.gradio-chatbot .message.bot p { text-align: left; }


.message-emergency {
    border-left: 4px solid var(--danger-color);
    background-color: #fff8f8;
    padding: 16px;
    margin: 16px 0;
    border-radius: 4px;
}

.vital-card {
    background-color: white;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: var(--card-shadow);
    transition: transform 0.2s ease;
}

.vital-card:hover {
    transform: translateY(-5px);
}

.vital-title {
    color: var(--primary-color);
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 8px;
    margin-bottom: 12px;
    font-weight: 600;
}

.vital-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-color);
}

.vital-timestamp {
    font-size: 0.8rem;
    color: #777;
}

.resource-card {
    background-color: white;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 3px solid var(--secondary-color);
}

.resource-category {
    font-weight: 600;
    color: var(--primary-color);
}

.tab-content {
    padding: 16px;
    background-color: white;
    border-radius: 0 0 10px 10px;
}

/* Target Gradio buttons specifically if needed */
.gradio-button.health-submit-btn {
    background-color: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
}

.gradio-button.health-submit-btn:hover {
    background-color: #004d80 !important;
}

.gradio-button.health-clear-btn {
    background-color: #f0f0f0 !important;
    color: #555 !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
}

.gradio-button.health-clear-btn:hover {
    background-color: #e0e0e0 !important;
}

.message-input textarea { /* Target the textarea within the input component */
    border: 1px solid #ddd;
    border-radius: 20px;
    padding: 12px 20px;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: none !important; /* Override Gradio default shadows if needed */
}

.message-input textarea:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(0, 105, 179, 0.2) !important;
}

.emergency-banner {
    background-color: var(--danger-color);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
}

.emergency-icon {
    margin-right: 12px;
    font-size: 1.5rem;
}

.health-footer {
    text-align: center;
    font-size: 0.9rem;
    color: #777;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #eee;
}

.health-chart {
    width: 100%;
    /* height: 300px; */ /* Let image dictate height */
    margin: 1rem 0;
    box-shadow: var(--card-shadow);
    border-radius: 8px;
    overflow: hidden;
}

.health-chart img { /* Style the image inside the chart div */
    display: block; /* Remove extra space below image */
    width: 100%;
    height: auto; /* Maintain aspect ratio */
    border-radius: 8px;
}


.symptom-badge {
    display: inline-block;
    background-color: var(--secondary-color);
    color: white;
    border-radius: 16px;
    padding: 4px 10px;
    font-size: 0.8rem;
    margin-right: 6px;
    margin-bottom: 6px;
}

.medication-reminder {
    background-color: #fff8e1;
    border-left: 3px solid var(--warning-color);
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 4px;
}

.medication-name {
    font-weight: 600;
    color: var(--text-color);
}

.medication-schedule {
    font-size: 0.9rem;
    color: #777;
}

.progress-bar {
    height: 10px; /* Slightly thicker */
    background-color: #eee;
    border-radius: 5px; /* Match height */
    overflow: hidden;
    margin: 8px 0;
}

.progress-bar-inner {
    height: 100%;
    background-color: var(--primary-color);
    border-radius: 5px; /* Match height */
    transition: width 0.5s ease-in-out; /* Animate width changes */
}

.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}

.tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

.tooltip-text {
    visibility: hidden;
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    text-align: center;
    border-radius: 6px;
    padding: 8px 12px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    width: 200px;
    font-size: 0.8rem;
}

.health-report {
    background-color: white;
    border-radius: 8px;
    padding: 24px;
    margin-top: 16px;
    box-shadow: var(--card-shadow);
}

.report-header {
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 12px;
    margin-bottom: 16px;
}
.report-header h2 { margin-top: 0; }

.report-section {
    margin-bottom: 20px;
}

.report-section-title {
    color: var(--primary-color);
    margin-bottom: 12px;
    font-size: 1.2rem;
}

.health-score {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    text-align: center;
    margin: 1rem 0;
}

.health-score-label {
    font-size: 0.9rem;
    color: #777;
    text-align: center;
    margin-bottom: 0.5rem; /* Added margin */
}

.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    width: 16px; /* Smaller spinner */
    height: 16px;
    animation: spin 1s linear infinite;
    display: inline-block;
    margin-right: 8px;
    vertical-align: middle;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.pdf-export-btn {
    background-color: #e53935;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 8px 16px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex; /* Use inline-flex for button content alignment */
    align-items: center;
    justify-content: center;
    transition: background-color 0.3s ease; /* Added transition */
}

.pdf-export-btn:hover {
    background-color: #c62828;
}

.pdf-icon {
    margin-right: 8px;
}

.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background-color: white;
    border-left: 4px solid var(--primary-color);
    padding: 12px 16px;
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 1000;
    transform: translateX(120%);
    transition: transform 0.3s ease;
    max-width: 300px;
}

.notification.show {
    transform: translateX(0);
}

.notification-title {
    font-weight: 600;
    margin-bottom: 4px;
}

.notification-body {
    font-size: 0.9rem;
    color: #555;
}

.health-tips {
    background-color: #e8f5e9;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    border-left: 3px solid #4caf50;
}

.tip-title {
    font-weight: 600;
    color: #2e7d32;
    margin-bottom: 8px;
}

/* Style for example buttons */
.example-btn {
    background: none;
    border: 1px solid #ddd;
    border-radius: 15px;
    padding: 5px 10px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #555;
    margin: 2px; /* Add small margin */
}
.example-btn:hover {
    background-color: #f0f0f0;
    border-color: #ccc;
}
"""

# --- Healthcare Bot Configuration ---

SYSTEM_PROMPT = """
You are MediGuide, an advanced healthcare assistant chatbot powered by Gemini. Your role is to provide helpful,
accurate health information while being mindful of the following guidelines:

1.  Provide general health information, wellness tips, and guidance on common medical conditions.
2.  **Crucially, always state clearly that you are an AI assistant and cannot provide medical advice or diagnosis.**
3.  **Consistently advise users to consult qualified healthcare professionals for diagnosis, treatment decisions, and serious medical concerns.** Emphasize this for any symptom analysis or condition discussion.
4.  Maintain user privacy. Do not ask for personally identifiable information beyond what's necessary for context (like age or general conditions if offered by the user). Avoid storing sensitive data long-term.
5.  Provide information based on generally accepted medical knowledge. Avoid definitive statements about specific user conditions.
6.  Do NOT make medical diagnoses or prescribe specific treatments or medications. You can discuss general types of treatments or medications for conditions but not recommend them for the user specifically.
7.  Be empathetic and supportive, but maintain professional boundaries.
8.  Suggest reliable health resources (like CDC, WHO, Mayo Clinic, MedlinePlus) for further information, but clarify these are for informational purposes only.
9.  Keep responses clear, concise, and easy to understand. Avoid jargon where possible.
10. For complex topics, break down information simply.
11. If relevant and provided by the user, use contextual information like age or mentioned conditions to tailor general information, but reiterate the need for professional consultation.
12. **If a user describes potentially life-threatening symptoms (e.g., severe chest pain, difficulty breathing, sudden weakness, suicidal thoughts), immediately and strongly advise them to seek emergency medical help (e.g., call 911 or local emergency number).**

Your primary goal is to be a helpful informational guide, always prioritizing user safety and directing them towards professional medical care when appropriate.
"""

# --- User Session Management ---

class UserSession:
    """Manages data for a single user session."""
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.conversation_history = []
        self.user_profile = {
            "name": "",
            "age": None,
            "gender": "",
            "height_cm": None, # Store consistently in cm
            "weight_kg": None, # Store consistently in kg
            "allergies": [],
            "chronic_conditions": [],
            "current_medications": [], # Simplified, detailed reminders separate
            "last_checkup": "",
            "risk_factors": [],
            "wellness_goals": []
        }
        self.previous_recommendations = []
        self.vital_signs = {} # e.g., {"blood_pressure": [{"value": "120/80", "unit": "mmHg", "timestamp": ...}]}
        self.medication_reminders = [] # More structured reminders
        self.symptom_log = [] # e.g., [{"symptom": "headache", "severity": "mild", ...}]
        self.wellness_activities = []
        self.health_analytics = {
            "interaction_count": 0,
            "topics_discussed": set(), # Use a set for unique topics
            "last_health_score": None,
            "health_score_history": [], # Track score changes
            "wellness_trend": "stable" # Could be calculated later
        }
        self.notification_preferences = { # Placeholder for settings
            "medication_reminders": True,
            "checkup_reminders": True,
            "health_tips": True,
            "data_summaries": True
        }
        logging.info(f"UserSession created for user: {self.user_id}")

    def add_message(self, role, message, timestamp=None):
        """Adds a message to the conversation history."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({"role": role, "message": message, "timestamp": timestamp})
        if role == "user":
            self.health_analytics["interaction_count"] += 1

    def update_profile(self, key, value):
        """Updates a specific field in the user profile."""
        if key in self.user_profile:
            # Basic validation/conversion
            if key == "age" and value is not None:
                try:
                    self.user_profile[key] = int(value)
                except (ValueError, TypeError):
                    logging.warning(f"Invalid age value: {value}. Not updated.")
            elif key == "height_cm" and value is not None:
                 try:
                    self.user_profile[key] = float(value)
                 except (ValueError, TypeError):
                    logging.warning(f"Invalid height value: {value}. Not updated.")
            elif key == "weight_kg" and value is not None:
                 try:
                    self.user_profile[key] = float(value)
                 except (ValueError, TypeError):
                    logging.warning(f"Invalid weight value: {value}. Not updated.")
            elif isinstance(self.user_profile[key], list) and isinstance(value, list):
                 # Add unique items to list fields like allergies/conditions
                 current_list = self.user_profile[key]
                 new_items = [item for item in value if item not in current_list]
                 self.user_profile[key].extend(new_items)
            else:
                self.user_profile[key] = value
            logging.info(f"Profile updated for user {self.user_id}: {key} = {self.user_profile[key]}")
        else:
             logging.warning(f"Attempted to update non-existent profile key: {key}")

    def add_recommendation(self, recommendation, category="general"):
        """Adds a health recommendation provided by the bot."""
        self.previous_recommendations.append({
            "recommendation": recommendation,
            "category": category,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "implemented": False # Placeholder for future tracking
        })

    def add_vital_sign(self, vital_type, value, unit):
        """Adds a vital sign measurement."""
        if not vital_type or value is None:
            logging.warning("Attempted to add vital sign with missing type or value.")
            return
        if vital_type not in self.vital_signs:
            self.vital_signs[vital_type] = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.vital_signs[vital_type].append({
            "value": value, # Keep original value string if complex (like BP)
            "unit": unit,
            "timestamp": timestamp
        })
        logging.info(f"Vital sign added for user {self.user_id}: {vital_type}={value} {unit}")

    def add_medication_reminder(self, medication, dosage, schedule, duration=None, notes=None):
        """Adds a medication reminder."""
        # Avoid duplicates
        if any(m['medication'].lower() == medication.lower() for m in self.medication_reminders):
            logging.info(f"Medication reminder for {medication} already exists.")
            return

        self.medication_reminders.append({
            "medication": medication,
            "dosage": dosage,
            "schedule": schedule,
            "duration": duration,
            "notes": notes,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "adhered_doses": 0, # Placeholder
            "missed_doses": 0   # Placeholder
        })
        logging.info(f"Medication reminder added for user {self.user_id}: {medication}")
        # Also add to simple profile list if not already there
        if medication not in self.user_profile["current_medications"]:
             self.user_profile["current_medications"].append(medication)


    def log_symptom(self, symptom, severity="moderate", related_factors=None):
        """Logs a symptom reported by the user."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.symptom_log.append({
            "symptom": symptom,
            "severity": severity,
            "related_factors": related_factors,
            "timestamp": timestamp
        })
        logging.info(f"Symptom logged for user {self.user_id}: {symptom} ({severity})")

    def add_wellness_activity(self, activity_type, duration=None, notes=None):
        """Adds a wellness activity reported by the user."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.wellness_activities.append({
            "activity_type": activity_type,
            "duration": duration,
            "notes": notes,
            "timestamp": timestamp
        })
        logging.info(f"Wellness activity added for user {self.user_id}: {activity_type}")

    def calculate_bmi(self):
        """Calculates BMI if height and weight are available."""
        if self.user_profile["height_cm"] and self.user_profile["weight_kg"]:
            try:
                height_m = float(self.user_profile["height_cm"]) / 100
                weight_kg = float(self.user_profile["weight_kg"])
                if height_m > 0 and weight_kg > 0:
                    bmi = weight_kg / (height_m * height_m)
                    return round(bmi, 1)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logging.error(f"Error calculating BMI for user {self.user_id}: {e}")
        return None

    def get_bmi_category(self):
        """Returns the BMI category based on calculated BMI."""
        bmi = self.calculate_bmi()
        if bmi is None:
            return "N/A"
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Healthy Weight"
        elif 25 <= bmi < 30:
            return "Overweight"
        else:
            return "Obesity"

    def calculate_health_score(self):
        """
        Calculates a simple health score based on available data.
        NOTE: This is a highly simplified heuristic and NOT a clinical assessment.
        """
        score = 70  # Base score
        logging.debug(f"Calculating health score for user {self.user_id}. Base score: {score}")

        # Adjust based on BMI
        bmi = self.calculate_bmi()
        if bmi is not None:
            if 18.5 <= bmi < 25: score += 5
            elif 25 <= bmi < 30: score += 0
            else: score -= 5
            logging.debug(f"Score after BMI ({bmi}): {score}")

        # Adjust based on recent vital signs (simplified check on last reading)
        for vital, readings in self.vital_signs.items():
            if readings:
                latest_reading = readings[-1]
                in_range = is_vital_in_normal_range(vital, latest_reading['value'])
                if in_range is True: score += 3
                elif in_range is False: score -= 3
                logging.debug(f"Score after vital {vital} ({latest_reading['value']}, range={in_range}): {score}")

        # Adjust based on recent wellness activities (last 7 days)
        recent_activities = [a for a in self.wellness_activities
                             if (datetime.now() - datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")).days <= 7]
        if len(recent_activities) >= 3: score += 5
        elif len(recent_activities) > 0: score += 2
        logging.debug(f"Score after wellness activities ({len(recent_activities)} recent): {score}")


        # Adjust based on recent symptoms (last 7 days)
        recent_symptoms = [s for s in self.symptom_log
                           if (datetime.now() - datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S")).days <= 7]
        if len(recent_symptoms) >= 3: score -= 5
        elif len(recent_symptoms) > 0: score -= 2
        # Penalize more for severe symptoms
        severe_symptoms = [s for s in recent_symptoms if s.get("severity") == "severe"]
        score -= len(severe_symptoms) * 2
        logging.debug(f"Score after symptoms ({len(recent_symptoms)} recent, {len(severe_symptoms)} severe): {score}")


        # Adjust based on chronic conditions
        score -= len(self.user_profile["chronic_conditions"]) * 2
        logging.debug(f"Score after chronic conditions ({len(self.user_profile['chronic_conditions'])}): {score}")


        # Ensure score is within bounds [0, 100]
        score = max(0, min(100, int(round(score)))) # Round to integer

        # Update history and last score
        if self.health_analytics["last_health_score"] is not None:
             self.health_analytics["health_score_history"].append({
                 "score": self.health_analytics["last_health_score"],
                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Approx time of previous score
             })
        self.health_analytics["last_health_score"] = score
        logging.info(f"Calculated health score for user {self.user_id}: {score}")

        return score

# Initialize user sessions dictionary
user_sessions = {}

# --- Health Data & Resources ---
# (Keeping these inline for simplicity, but could be loaded from JSON/CSV)

RELIABLE_HEALTH_RESOURCES = {
    # ... (Resource data remains the same) ...
    "General Health": [
        {"name": "CDC", "url": "https://www.cdc.gov/", "description": "Centers for Disease Control and Prevention - Official health information from the US government"},
        {"name": "WHO", "url": "https://www.who.int/", "description": "World Health Organization - Global health guidance and information"},
        {"name": "Mayo Clinic", "url": "https://www.mayoclinic.org/", "description": "Comprehensive medical information on diseases and conditions"},
        {"name": "MedlinePlus", "url": "https://medlineplus.gov/", "description": "Health information from the US National Library of Medicine"}
    ],
    "Mental Health": [
        {"name": "NIMH", "url": "https://www.nimh.nih.gov/", "description": "National Institute of Mental Health - Information on mental disorders"},
        {"name": "MentalHealth.gov", "url": "https://www.mentalhealth.gov/", "description": "US government information on mental health"},
        {"name": "NAMI", "url": "https://www.nami.org/", "description": "National Alliance on Mental Illness - Resources and support"}
    ],
    "Nutrition": [
        {"name": "Nutrition.gov", "url": "https://www.nutrition.gov/", "description": "US government information on nutrition and healthy eating"},
        {"name": "EatRight.org", "url": "https://www.eatright.org/", "description": "Academy of Nutrition and Dietetics information"},
        {"name": "MyPlate", "url": "https://www.myplate.gov/", "description": "USDA food guidance system"}
    ],
    "Fitness": [
        {"name": "Health.gov", "url": "https://health.gov/moveyourway", "description": "US Department of Health physical activity guidelines"},
        {"name": "CDC Physical Activity", "url": "https://www.cdc.gov/physicalactivity/", "description": "Information on exercise and physical activity benefits"}
    ],
    "Medications": [
        {"name": "MedlinePlus Drugs", "url": "https://medlineplus.gov/druginformation.html", "description": "Drug information from the US National Library of Medicine"},
        {"name": "FDA Drug Information", "url": "https://www.fda.gov/drugs", "description": "US Food and Drug Administration drug information"}
    ],
    "Emergency": [
        {"name": "Emergency Services", "url": "#", "description": "Call 911 (US) or local emergency services immediately"}
    ]
}

COMMON_SYMPTOMS = {
    # ... (Symptom data remains the same) ...
    "Common Cold": {
        "symptoms": ["runny nose", "sneezing", "congestion", "sore throat", "cough", "mild fever"],
        "duration": "7-10 days",
        "contagiousness": "high",
        "severity": "mild",
        "self_care": ["rest", "fluids", "over-the-counter cold medicine", "humidifier"],
        "when_to_see_doctor": ["fever over 101.3°F (38.5°C)", "symptoms lasting more than 10 days", "severe symptoms", "shortness of breath"]
    },
    "Influenza (Flu)": {
        "symptoms": ["fever", "chills", "body aches", "fatigue", "headache", "cough", "sore throat"],
        "duration": "1-2 weeks",
        "contagiousness": "high",
        "severity": "moderate",
        "self_care": ["rest", "fluids", "over-the-counter pain relievers", "prescription antivirals (if taken early)"],
        "when_to_see_doctor": ["difficulty breathing", "chest pain", "severe weakness", "worsening of chronic conditions"]
    },
    # Add more conditions as needed...
    "COVID-19": {
        "symptoms": ["fever", "cough", "shortness of breath", "fatigue", "body aches", "loss of taste", "loss of smell", "sore throat", "congestion", "runny nose", "nausea", "diarrhea"],
        "duration": "1-3 weeks for mild cases, longer for severe",
        "contagiousness": "high",
        "severity": "mild to severe",
        "self_care": ["rest", "fluids", "over-the-counter pain/fever reducers", "isolation"],
        "when_to_see_doctor": ["difficulty breathing", "persistent chest pain or pressure", "new confusion", "inability to wake or stay awake", "bluish lips or face"]
    },
     "Migraine": {
        "symptoms": ["severe headache", "throbbing pain", "pulsating pain", "sensitivity to light", "sensitivity to sound", "nausea", "vomiting", "aura"],
        "duration": "4-72 hours",
        "contagiousness": "none",
        "severity": "moderate to severe",
        "self_care": ["rest in dark quiet room", "over-the-counter pain relievers (e.g., ibuprofen, naproxen)", "prescription migraine medication", "cold compress", "hydration"],
        "when_to_see_doctor": ["first severe headache", "headache with fever, stiff neck, confusion, seizure, double vision, weakness, numbness", "headache after head injury", "chronic headache pattern changes"]
    },
}

WELLNESS_TIPS = {
    # ... (Wellness tips data remains the same) ...
    "Nutrition": [
        {"tip": "Aim for 5 servings of fruits and vegetables daily", "benefit": "Provides essential vitamins, minerals, and fiber"},
        {"tip": "Stay hydrated with at least 8 glasses of water daily", "benefit": "Supports digestion, circulation, and temperature regulation"},
        {"tip": "Limit processed foods and added sugars", "benefit": "Reduces risk of chronic diseases like diabetes and heart disease"},
        {"tip": "Include lean proteins in your meals", "benefit": "Supports muscle maintenance and provides sustained energy"},
        {"tip": "Incorporate healthy fats like avocados, nuts, and olive oil", "benefit": "Supports brain health and reduces inflammation"}
    ],
    "Physical Activity": [
        {"tip": "Aim for 150 minutes of moderate exercise weekly", "benefit": "Improves cardiovascular health and mood"},
        {"tip": "Include strength training 2-3 times per week", "benefit": "Builds muscle, improves metabolism, and supports bone health"},
        {"tip": "Take short walking breaks throughout the day", "benefit": "Reduces prolonged sitting which is linked to health risks"},
        {"tip": "Find activities you enjoy to make exercise sustainable", "benefit": "Increases likelihood of maintaining regular physical activity"},
        {"tip": "Start with small, achievable fitness goals", "benefit": "Builds confidence and prevents injury from overexertion"}
    ],
    # Add more categories/tips as needed...
    "Mental Wellbeing": [
        {"tip": "Practice mindfulness or meditation for 10 minutes daily", "benefit": "Reduces stress and improves focus"},
        {"tip": "Maintain social connections with friends and family", "benefit": "Boosts mood and provides emotional support"},
        {"tip": "Establish a consistent sleep schedule", "benefit": "Improves cognitive function and emotional regulation"},
        {"tip": "Take time for activities you enjoy", "benefit": "Reduces stress and prevents burnout"},
        {"tip": "Consider keeping a gratitude journal", "benefit": "Shifts focus to positive aspects of life, improving overall outlook"}
    ],
}

# --- Helper Functions ---

def get_current_timestamp():
    """Returns the current timestamp in a standard format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_health_chart(data_type, data_values, dates, chart_type="line", unit=""):
    """Generates a base64 encoded PNG chart string using Matplotlib."""
    if not data_values or not dates or len(data_values) != len(dates):
        logging.warning(f"Insufficient or mismatched data for plotting {data_type}.")
        return None

    # Ensure data_values are numeric where possible
    numeric_values = []
    valid_dates = []
    for i, val in enumerate(data_values):
        try:
             # Handle simple numeric strings and numbers
            if isinstance(val, (int, float)):
                 numeric_values.append(val)
                 valid_dates.append(dates[i])
            elif isinstance(val, str) and val.replace('.', '', 1).replace('-', '', 1).isdigit():
                 numeric_values.append(float(val))
                 valid_dates.append(dates[i])
            else:
                 logging.debug(f"Skipping non-numeric value '{val}' for plotting {data_type}")
        except Exception as e:
            logging.warning(f"Error converting value '{val}' to numeric for plotting {data_type}: {e}")

    if len(numeric_values) < 2: # Need at least 2 points to plot a line/bar
        logging.warning(f"Not enough valid numeric data points to plot {data_type}.")
        return None

    # Use dates directly if they are datetime objects, otherwise try parsing
    try:
        plot_dates = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in valid_dates]
        # Sort data by date for line plots
        if chart_type == "line":
            sorted_pairs = sorted(zip(plot_dates, numeric_values))
            plot_dates, numeric_values = zip(*sorted_pairs)
    except (ValueError, TypeError) as e:
        logging.error(f"Error parsing dates for plotting {data_type}: {e}. Using indices.")
        plot_dates = range(len(numeric_values)) # Fallback to indices


    plt.style.use('seaborn-v0_8-fivethirtyeight') # Use a nice style
    fig, ax = plt.subplots(figsize=(8, 4)) # Smaller figure size for dashboard

    try:
        if chart_type == "line":
            ax.plot(plot_dates, numeric_values, marker='o', linestyle='-', color='var(--primary-color)', linewidth=2)
            ax.fill_between(plot_dates, numeric_values, alpha=0.1, color='var(--secondary-color)')
        elif chart_type == "bar":
            ax.bar(plot_dates, numeric_values, color='var(--primary-color)')
        elif chart_type == "scatter":
            ax.scatter(plot_dates, numeric_values, color='var(--primary-color)', s=60, alpha=0.7)

        ax.set_title(f'{data_type} Trend', fontsize=14, fontweight='bold', color='var(--text-color)')
        ax.set_ylabel(f"{data_type}{(' (' + unit + ')') if unit else ''}", fontsize=10, color='#555555')
        ax.grid(True, linestyle='--', alpha=0.6, axis='y') # Grid on y-axis only
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # Add reference ranges if applicable (simplified)
        range_info = get_normal_range(data_type)
        if range_info:
             min_val, max_val = range_info
             ax.axhspan(min_val, max_val, alpha=0.15, color='var(--success-color)', label=f'Normal ({min_val}-{max_val})')
             ax.legend(fontsize=8)

        # Improve layout
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=90) # Lower DPI for web
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # Close the figure to free memory
        logging.debug(f"Generated chart for {data_type}")
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        logging.error(f"Error generating plot for {data_type}: {e}")
        plt.close(fig) # Ensure figure is closed even on error
        return None

def get_normal_range(vital_type):
    """Returns typical normal range (min, max) for common vital signs."""
    vital_type = vital_type.lower()
    if "blood pressure" in vital_type and "systolic" in vital_type: return (90, 120)
    if "blood pressure" in vital_type and "diastolic" in vital_type: return (60, 80)
    if "heart rate" in vital_type: return (60, 100)
    if "blood sugar" in vital_type or "glucose" in vital_type: return (70, 100) # Fasting
    if "temperature" in vital_type: return (97.0, 99.0) # Fahrenheit
    if "oxygen saturation" in vital_type or "spo2" in vital_type: return (95, 100)
    return None

def is_vital_in_normal_range(vital_type, value):
    """Checks if a vital sign value is within a typical normal range."""
    vital_type = vital_type.lower()
    range_info = get_normal_range(vital_type)
    if range_info is None:
        return None # Cannot determine range

    min_val, max_val = range_info

    try:
        if "blood pressure" in vital_type:
            # Expects value like "120/80"
            parts = str(value).split('/')
            if len(parts) == 2:
                systolic = int(parts[0].strip())
                diastolic = int(parts[1].strip())
                # Check both components against *their* specific ranges
                sys_range = get_normal_range("systolic blood pressure")
                dia_range = get_normal_range("diastolic blood pressure")
                if sys_range and dia_range:
                    return sys_range[0] <= systolic <= sys_range[1] and dia_range[0] <= diastolic <= dia_range[1]
            return False # Invalid format
        else:
            # Handle numeric types
            numeric_value = float(value)
            # Simple temperature scale check (heuristic)
            if "temperature" in vital_type:
                 if numeric_value > 50: # Assume Fahrenheit
                     pass # Use default F range
                 else: # Assume Celsius
                     min_val, max_val = (36.1, 37.2) # Celsius range

            return min_val <= numeric_value <= max_val
    except (ValueError, TypeError, IndexError):
        logging.warning(f"Could not parse value '{value}' for vital type '{vital_type}' range check.")
        return None # Cannot parse value

def analyze_health_trends(session: UserSession):
    """Analyzes trends in vital signs, symptoms, and activities."""
    if not session: return None
    trends = {}
    now = datetime.now()

    # Analyze vital signs trends (simple slope calculation if > 2 points)
    for vital_type, measurements in session.vital_signs.items():
        if len(measurements) < 2: continue

        numeric_values = []
        timestamps = []
        is_bp = "blood_pressure" in vital_type.lower()

        for m in measurements:
            try:
                timestamp = datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M:%S")
                if is_bp:
                    parts = str(m["value"]).split('/')
                    if len(parts) == 2:
                        numeric_values.append(int(parts[0].strip())) # Use Systolic for trend indication
                        timestamps.append(timestamp)
                else:
                    numeric_values.append(float(m["value"]))
                    timestamps.append(timestamp)
            except (ValueError, TypeError, IndexError):
                continue # Skip invalid entries for trend analysis

        if len(numeric_values) >= 2:
            # Simple trend: Compare last value to average of previous ones
            latest_val = numeric_values[-1]
            avg_previous = np.mean(numeric_values[:-1]) if len(numeric_values) > 1 else latest_val

            trend_direction = "stable"
            if latest_val > avg_previous + (0.05 * abs(avg_previous)): # > 5% increase
                trend_direction = "increasing"
            elif latest_val < avg_previous - (0.05 * abs(avg_previous)): # > 5% decrease
                trend_direction = "decreasing"

            # Determine if trend is improving/declining based on vital type
            is_improving = None
            if trend_direction != "stable":
                # Generally lower is better for BP, HR (within limits), sometimes Temp, Sugar
                # Generally higher is better for SpO2
                lower_is_better = any(vt in vital_type.lower() for vt in ["pressure", "heart", "sugar", "glucose", "temperature"])
                higher_is_better = any(vt in vital_type.lower() for vt in ["oxygen", "spo2"])

                if lower_is_better:
                    is_improving = trend_direction == "decreasing"
                elif higher_is_better:
                    is_improving = trend_direction == "increasing"

            trends[vital_type] = {
                "direction": trend_direction,
                "improving": is_improving, # True, False, or None
                "latest_value": measurements[-1]['value'],
                "is_in_range": is_vital_in_normal_range(vital_type, measurements[-1]['value'])
            }

    # Analyze symptom trends (frequency in last 14 days)
    recent_symptoms = [s for s in session.symptom_log
                       if (now - datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S")).days <= 14]
    if recent_symptoms:
        symptom_frequency = {}
        for entry in recent_symptoms:
            symptom = entry["symptom"]
            symptom_frequency[symptom] = symptom_frequency.get(symptom, 0) + 1
        trends["symptoms"] = {
            "most_frequent": sorted(symptom_frequency.items(), key=lambda item: item[1], reverse=True)[:3],
            "total_reported": len(recent_symptoms),
            "unique_count": len(symptom_frequency)
        }

    # Analyze wellness activity trends (frequency in last 14 days)
    recent_activities = [a for a in session.wellness_activities
                         if (now - datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")).days <= 14]
    if recent_activities:
        activity_types = {}
        for activity in recent_activities:
            activity_type = activity["activity_type"]
            activity_types[activity_type] = activity_types.get(activity_type, 0) + 1
        trends["wellness"] = {
            "most_frequent": sorted(activity_types.items(), key=lambda item: item[1], reverse=True)[:3],
            "total_activities": len(recent_activities),
            "unique_types": len(activity_types)
        }

    # Health score trend
    current_score = session.calculate_health_score() # Ensure latest score is calculated
    score_history = session.health_analytics["health_score_history"]
    if score_history:
        previous_score_entry = score_history[-1]
        previous_score = previous_score_entry["score"]
        score_change = current_score - previous_score
        score_direction = "improving" if score_change > 0 else "declining" if score_change < 0 else "stable"
        trends["health_score"] = {
            "current": current_score,
            "previous": previous_score,
            "change": score_change,
            "direction": score_direction
        }
    elif session.health_analytics["last_health_score"] is not None:
         trends["health_score"] = {
             "current": current_score,
             "direction": "stable" # Initial score
         }


    return trends

def preprocess_health_query(message):
    """Detects potential emergencies and extracts basic health data using regex."""
    message_lower = message.lower()
    is_emergency = False
    extracted_data = {}

    # Emergency Keywords (Weighted) - More aggressive check
    emergency_keywords = {
        "chest pain": 10, "severe pain": 8, "cannot breathe": 10, "can't breathe": 10,
        "difficulty breathing": 9, "shortness of breath": 8, "stroke symptoms": 10,
        "sudden weakness": 9, "sudden numbness": 9, "facial droop": 10, "slurred speech": 9,
        "severe bleeding": 9, "uncontrolled bleeding": 10, "loss of consciousness": 10,
        "unconscious": 10, "unresponsive": 10, "seizure": 9, "collapse": 8,
        "head injury": 8, "major trauma": 8, "overdose": 9, "poisoning": 8,
        "suicidal": 10, "want to die": 10, "kill myself": 10,
        "allergic reaction severe": 8, "anaphylaxis": 10,
        "emergency": 5 # Lower weight for just the word
    }
    emergency_score = 0
    for keyword, score in emergency_keywords.items():
        if keyword in message_lower:
            logging.warning(f"Emergency keyword detected: '{keyword}'")
            emergency_score += score

    if emergency_score >= 9: # Adjusted threshold
        is_emergency = True
        logging.warning(f"Potential emergency detected! Score: {emergency_score}")


    # Regex for Data Extraction (using re.IGNORECASE)
    # Blood Pressure: "bp 120/80", "blood pressure is 120 / 75"
    bp_match = re.search(r'(?:blood pressure|bp)\s*(?:is|was|:|)\s*(\d{2,3})\s*/\s*(\d{2,3})', message_lower, re.IGNORECASE)
    if bp_match:
        extracted_data["blood_pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"

    # Temperature: "temp 98.6 F", "temperature 37 C"
    temp_match = re.search(r'(?:temperature|temp)\s*(?:is|was|:|)\s*(\d{2,3}(?:\.\d)?)\s*(?:°|degrees)?\s*([CF])?', message_lower, re.IGNORECASE)
    if temp_match:
        value = float(temp_match.group(1))
        unit = temp_match.group(2).upper() if temp_match.group(2) else None
        # Basic unit inference if not provided
        if not unit:
             unit = "°F" if value > 50 else "°C"
        elif unit == 'F': unit = "°F"
        elif unit == 'C': unit = "°C"
        extracted_data["temperature"] = {"value": value, "unit": unit}

    # Heart Rate: "hr 70 bpm", "pulse is 65"
    hr_match = re.search(r'(?:heart rate|hr|pulse)\s*(?:is|was|:|)\s*(\d{2,3})\s*(bpm)?', message_lower, re.IGNORECASE)
    if hr_match:
        extracted_data["heart_rate"] = {"value": int(hr_match.group(1)), "unit": "bpm"}

    # Blood Sugar: "sugar 100 mg/dl", "glucose 5.5 mmol/l"
    sugar_match = re.search(r'(?:blood sugar|glucose|sugar level)\s*(?:is|was|:|)\s*(\d{1,3}(?:\.\d)?)\s*(mg/dl|mmol/l)?', message_lower, re.IGNORECASE)
    if sugar_match:
        unit = sugar_match.group(2) if sugar_match.group(2) else "mg/dL" # Default unit
        extracted_data["blood_sugar"] = {"value": float(sugar_match.group(1)), "unit": unit}

    # SpO2: "spo2 98%", "oxygen saturation 97"
    spo2_match = re.search(r'(?:spo2|oxygen saturation|o2 sat)\s*(?:is|was|:|)\s*(\d{2,3})\s*(%)?', message_lower, re.IGNORECASE)
    if spo2_match:
         extracted_data["oxygen_saturation"] = {"value": int(spo2_match.group(1)), "unit": "%"}

    # Weight: "weight 70 kg", "weigh 150 lbs"
    weight_match = re.search(r'(?:weight|weighs|weigh)\s*(?:is|was|:|)\s*(\d{2,4}(?:\.\d)?)\s*(kg|kilos|kilograms|lb|lbs|pounds)', message_lower, re.IGNORECASE)
    if weight_match:
        value = float(weight_match.group(1))
        unit_str = weight_match.group(2).lower()
        unit = "kg"
        if "lb" in unit_str or "pound" in unit_str:
            value = round(value * 0.453592, 1) # Convert lbs to kg
        extracted_data["weight_kg"] = {"value": value, "unit": "kg"}

    # Height: "height 175 cm", "tall 5 ft 10 in", "height 6'1\""
    # Simpler cm/m pattern first
    height_cm_match = re.search(r'(?:height|tall)\s*(?:is|was|:|)\s*(\d{2,3}(?:\.\d)?)\s*(cm|centimeters)', message_lower, re.IGNORECASE)
    height_m_match = re.search(r'(?:height|tall)\s*(?:is|was|:|)\s*(\d(?:\.\d{1,2})?)\s*(m|meters)', message_lower, re.IGNORECASE)
    # Feet/inches patterns (more complex)
    height_ft_in_match = re.search(r'(?:height|tall)\s*(?:is|was|:|)\s*(\d+)\s*(?:ft|feet|\')(?:\s*(\d{1,2})\s*(?:in|inches|"))?', message_lower, re.IGNORECASE)

    if height_cm_match:
        extracted_data["height_cm"] = {"value": float(height_cm_match.group(1)), "unit": "cm"}
    elif height_m_match:
        value_m = float(height_m_match.group(1))
        extracted_data["height_cm"] = {"value": round(value_m * 100, 1), "unit": "cm"}
    elif height_ft_in_match:
        feet = int(height_ft_in_match.group(1))
        inches = int(height_ft_in_match.group(2)) if height_ft_in_match.group(2) else 0
        total_inches = (feet * 12) + inches
        cm_value = round(total_inches * 2.54, 1)
        extracted_data["height_cm"] = {"value": cm_value, "unit": "cm"}


    # Age: "age 30", "I am 25 years old"
    age_match = re.search(r'\b(age|aged)\s*(\d{1,3})\b|\bI(?: am|\'m)\s*(\d{1,3})\s*(?:years? old)?\b', message_lower, re.IGNORECASE)
    if age_match:
        # Group 2 is for "age XX", Group 3 is for "I am XX"
        age_val = age_match.group(2) or age_match.group(3)
        extracted_data["age"] = int(age_val)

    if extracted_data:
        logging.info(f"Pre-processed extracted data: {extracted_data}")

    # Return original message, emergency flag, and extracted data
    return message, is_emergency, extracted_data


def identify_potential_conditions(message, symptom_db=COMMON_SYMPTOMS):
    """Identifies potential conditions based on keywords matching symptoms."""
    message_lower = message.lower()
    potential_conditions = []
    reported_symptoms = set() # Use a set to avoid duplicates initially

    # 1. Find all mentioned symptoms from our known list
    all_known_symptoms = set()
    for condition_data in symptom_db.values():
        all_known_symptoms.update(condition_data.get("symptoms", []))

    for known_symptom in all_known_symptoms:
        # Use word boundaries for more specific matching
        if re.search(r'\b' + re.escape(known_symptom) + r'\b', message_lower, re.IGNORECASE):
            reported_symptoms.add(known_symptom)

    if not reported_symptoms:
        return [], [] # No relevant symptoms found

    # 2. Match reported symptoms to conditions
    for condition, condition_data in symptom_db.items():
        condition_symptoms = set(condition_data.get("symptoms", []))
        matched_symptoms = list(reported_symptoms.intersection(condition_symptoms))

        if matched_symptoms:
            match_percentage = (len(matched_symptoms) / len(condition_symptoms)) * 100 if condition_symptoms else 0
            specificity_score = calculate_symptom_specificity(matched_symptoms, symptom_db)

            potential_conditions.append({
                "condition": condition,
                "matched_symptoms": matched_symptoms,
                "match_percentage": round(match_percentage, 1),
                "specificity_score": round(specificity_score, 3),
                "severity_info": condition_data.get("severity", "N/A"), # Use a different key than symptom severity
                "when_to_see_doctor": condition_data.get("when_to_see_doctor", []),
                "self_care": condition_data.get("self_care", [])
            })

    # Sort by match percentage primarily, then specificity
    potential_conditions.sort(key=lambda x: (x["match_percentage"], x["specificity_score"]), reverse=True)

    logging.info(f"Identified reported symptoms: {list(reported_symptoms)}")
    logging.info(f"Potential conditions identified: {[pc['condition'] for pc in potential_conditions[:3]]}") # Log top 3

    return potential_conditions, list(reported_symptoms)


def calculate_symptom_specificity(symptoms, condition_db):
    """Calculates average specificity score for a list of symptoms."""
    if not symptoms: return 0
    total_specificity = 0
    num_symptoms = len(symptoms)

    for symptom in symptoms:
        symptom_lower = symptom.lower()
        # Count how many conditions list this symptom (case-insensitive)
        count = sum(1 for data in condition_db.values()
                   if symptom_lower in [s.lower() for s in data.get("symptoms", [])])
        # Specificity is inverse frequency (add 1 to avoid division by zero and give non-listed symptoms zero score)
        total_specificity += 1.0 / (count + 1) if count > 0 else 0

    return total_specificity / num_symptoms if num_symptoms > 0 else 0


def suggest_health_resources(message, reported_symptoms=None, resource_db=RELIABLE_HEALTH_RESOURCES):
    """Suggests relevant health resource categories based on message content."""
    message_lower = message.lower()
    suggested_categories = set() # Use set to avoid duplicates

    # Always suggest General Health
    suggested_categories.add("General Health")

    # Topic keywords
    topic_map = {
        "Mental Health": ["stress", "anxiety", "depress", "mood", "mental", "feeling down", "coping"],
        "Nutrition": ["diet", "food", "eating", "nutrition", "meal", "weight", "calorie", "recipes"],
        "Fitness": ["exercise", "workout", "activity", "fitness", "gym", "run", "walk", "strength"],
        "Medications": ["medication", "drug", "prescription", "pill", "medicine", "dose", "pharmacy"],
        "Emergency": ["emergency", "urgent", "severe", "critical", "911"] # Add emergency link if keywords present
    }

    for category, keywords in topic_map.items():
        if any(keyword in message_lower for keyword in keywords):
            suggested_categories.add(category)

    # Suggest based on symptoms -> broader categories (can be refined)
    if reported_symptoms:
        symptom_to_category = {
            "respiratory": ["cough", "shortness of breath", "congestion", "runny nose", "sore throat", "sneezing", "flu", "cold", "covid"],
            "digestive": ["nausea", "vomiting", "diarrhea", "stomach", "abdominal pain", "constipation", "heartburn", "acid reflux"],
            "neurological": ["headache", "migraine", "dizzy", "lightheaded", "confusion", "memory"],
            "skin": ["rash", "itch", "hives", "eczema", "dermatitis"],
            # Add more mappings
        }
        for cat, symps in symptom_to_category.items():
             if any(symptom in symps for symptom in reported_symptoms):
                 # Map symptom category to a resource category if logical (e.g., resp -> General, neuro -> General/Mental)
                 if cat == "respiratory" or cat == "digestive" or cat == "skin":
                      suggested_categories.add("General Health") # Covered by Mayo, CDC etc.
                 elif cat == "neurological":
                      suggested_categories.add("General Health")
                      suggested_categories.add("Mental Health") # Headaches can be stress related

    # Structure the output
    output_resources = []
    for category in suggested_categories:
        if category in resource_db:
             output_resources.append((category, resource_db[category]))

    logging.info(f"Suggested resource categories: {suggested_categories}")
    return output_resources


def extract_health_data(session: UserSession, message, pre_extracted_data=None):
    """Extracts health data from message using regex and updates session."""
    if not session: return False
    data_updated = False
    message_lower = message.lower()

    # 1. Use pre-extracted data first (from preprocess_health_query)
    if pre_extracted_data:
        for key, data in pre_extracted_data.items():
            if isinstance(data, dict) and 'value' in data and 'unit' in data:
                # Handle vitals with units
                session.add_vital_sign(key, data['value'], data['unit'])
                data_updated = True
            elif key in ["age", "height_cm", "weight_kg"]:
                # Handle profile data (which might be a dict or direct value)
                value_to_update = data['value'] if isinstance(data, dict) else data
                profile_key = key # Profile keys match extracted keys
                session.update_profile(profile_key, value_to_update)
                data_updated = True
            elif key == "blood_pressure": # BP is special case (string value)
                 session.add_vital_sign(key, data, "mmHg")
                 data_updated = True

    # 2. Regex for data not typically caught in preprocess (more conversational)
    # Allergies: "allergic to penicillin", "have an allergy to nuts"
    allergy_match = re.search(r'(?:allergic to|allergy to|allergies include)\s+([a-z\s,]+?)(?:\.|\s+and\s+|\s*but\s+|\s*$)', message_lower, re.IGNORECASE)
    if allergy_match:
        allergens = [a.strip() for a in re.split(r',|\s+and\s+', allergy_match.group(1)) if a.strip()]
        if allergens:
            session.update_profile("allergies", allergens) # update_profile handles adding unique items
            data_updated = True
            logging.info(f"Extracted allergies: {allergens}")


    # Chronic Conditions: "diagnosed with diabetes", "I have asthma", "suffer from hypertension"
    condition_phrases = [
        r'(?:diagnosed with|have|has|suffers? from)\s+([a-z\s\-,\/]+?)(?:\.|\s+and\s+|\s*but\s+|\s+for\s+|\s*$)'
    ]
    # Avoid matching simple things like "I have a cold" as chronic
    common_acute = ["cold", "flu", "headache", "stomach ache", "fever", "cough"]
    for phrase in condition_phrases:
        for match in re.finditer(phrase, message_lower, re.IGNORECASE):
            conditions = [c.strip() for c in re.split(r',|\s+and\s+', match.group(1)) if c.strip()]
            valid_conditions = [c for c in conditions if c not in common_acute and len(c) > 2] # Basic filter
            if valid_conditions:
                session.update_profile("chronic_conditions", valid_conditions)
                data_updated = True
                logging.info(f"Extracted chronic conditions: {valid_conditions}")


    # Medications: "taking lisinopril 10mg", "on metformin", "prescribed atorvastatin"
    med_phrases = [
        # Pattern: taking [med name] [optional dosage] [optional schedule]
        r'(?:taking|take|on|prescribed|uses?)\s+([a-z\-\s]+?)\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|tablets?|pills?))?(?:\s+([a-z\s\d]+?(?:daily|weekly|nightly|morning|evening|times a day|hours?)))?'
        # Pattern: [med name] [dosage] [schedule]
        # r'([a-z\-\s]+?)\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|tablets?|pills?))\s+([a-z\s\d]+?(?:daily|weekly|nightly|morning|evening|times a day|hours?))' # More specific but less flexible
    ]
    # Avoid matching very common words as meds
    common_words_filter = ["it", "this", "that", "medication", "medicine", "pill", "tablet", "drug"]

    extracted_meds_this_message = set()
    for phrase in med_phrases:
         for match in re.finditer(phrase, message_lower, re.IGNORECASE):
             med_name = match.group(1).strip()
             # Filter out common words and very short strings
             if med_name and med_name not in common_words_filter and len(med_name) > 2:
                 # Check if already processed in this message to avoid duplicates from overlapping regex
                 if med_name in extracted_meds_this_message: continue

                 dosage = match.group(2).strip() if match.group(2) else "Unknown Dosage"
                 schedule = match.group(3).strip() if match.group(3) else "As Directed/Unknown Schedule"

                 session.add_medication_reminder(med_name, dosage, schedule) # Handles internal check for existing reminders
                 extracted_meds_this_message.add(med_name)
                 data_updated = True
                 logging.info(f"Extracted medication: {med_name}, Dosage: {dosage}, Schedule: {schedule}")


    # Symptoms (Extract severity if possible)
    potential_conditions, reported_symptoms = identify_potential_conditions(message)
    if reported_symptoms:
         severity_map = {
             "severe": ["severe", "intense", "unbearable", "worst", "bad", "terrible"],
             "mild": ["mild", "slight", "minor", "a little", "bit of a"],
             "moderate": ["moderate", "noticeable", "significant"] # Default if not specified
         }
         extracted_symptoms_this_message = set()
         for symptom in reported_symptoms:
             if symptom in extracted_symptoms_this_message: continue

             symptom_severity = "moderate" # Default
             # Search for severity keywords near the symptom word
             for sev_level, keywords in severity_map.items():
                 for keyword in keywords:
                     # Pattern: keyword symptom OR symptom keyword OR symptom is/was keyword
                     pattern = rf'\b({keyword})\s+({re.escape(symptom)})\b|\b({re.escape(symptom)})\s+({keyword})\b|\b({re.escape(symptom)})\s+(?:is|was)\s+({keyword})\b'
                     sev_match = re.search(pattern, message_lower, re.IGNORECASE)
                     if sev_match:
                         symptom_severity = sev_level
                         logging.debug(f"Found severity '{sev_level}' for symptom '{symptom}'")
                         break # Found severity for this symptom
                 if symptom_severity != "moderate": break # Stop searching severity levels

             # Log the symptom with extracted/default severity
             session.log_symptom(symptom, symptom_severity)
             extracted_symptoms_this_message.add(symptom)
             data_updated = True


    # Wellness Activities (Simplified)
    activity_keywords = {
        "exercise": ["exercise", "workout", "gym", "run", "ran", "walked", "swam", "cycled", "lifted weights", "yoga", "pilates"],
        "meditation": ["meditate", "meditation", "mindfulness"],
        "healthy eating": ["healthy meal", "ate well", "balanced diet", "vegetables", "fruits", "lean protein"],
        "sleep": ["slept well", "good sleep", "hours of sleep"], # Could extract hours later
        "social": ["saw friends", "family time", "social event"],
        "hobby": ["hobby", "leisure activity", "relaxed", "read book"]
    }
    for activity_type, keywords in activity_keywords.items():
        if any(re.search(r'\b' + keyword + r'\b', message_lower, re.IGNORECASE) for keyword in keywords):
             # Avoid logging duplicates rapidly
             recent_activity_timestamps = [
                 datetime.strptime(a['timestamp'], "%Y-%m-%d %H:%M:%S")
                 for a in session.wellness_activities
                 if a['activity_type'] == activity_type
             ]
             now = datetime.now()
             if not any((now - ts).total_seconds() < 3600 for ts in recent_activity_timestamps): # Don't log same activity type within an hour
                session.add_wellness_activity(activity_type)
                data_updated = True


    return data_updated

def format_health_response(session: UserSession, bot_response, user_message, is_emergency, health_data_extracted, detected_health_data):
    """Formats the bot response with contextual health info, warnings, and resources."""
    if not session:
        return bot_response # Should not happen if session is managed properly

    formatted_response_parts = []

    # 1. Emergency Banner (Highest Priority)
    if is_emergency:
        emergency_warning = (
            '<div class="emergency-banner">'
            '<span class="emergency-icon">⚠️</span>'
            '<strong>MEDICAL EMERGENCY POSSIBLE:</strong> Based on your message, '
            'this could be a serious medical situation. '
            '<strong>Please seek immediate medical attention. Call 911 (or your local emergency number) '
            'or go to the nearest emergency room.</strong> Do not rely on this chat for emergency help.'
            '</div>'
        )
        formatted_response_parts.append(emergency_warning)
        # Add the original bot response *after* the critical warning
        formatted_response_parts.append(f"<p><strong>AI Assistant:</strong> {bot_response}</p>")
        # Stop further formatting if it's a clear emergency to avoid distraction
        return "".join(formatted_response_parts)

    # 2. Standard Bot Response
    formatted_response_parts.append(f"<p>{bot_response}</p>") # Keep original bot response prominent


    # 3. Data Extraction Confirmation
    if health_data_extracted and detected_health_data:
        data_confirmation = '<div class="vital-card" style="background-color: #e8f5e9; border-left: 3px solid var(--success-color);"><div class="vital-title">Health Info Noted</div><ul style="list-style: none; padding-left: 0;">'
        for data_type, data in detected_health_data.items():
             value_display = ""
             unit_display = ""
             if isinstance(data, dict):
                 value_display = data.get('value', 'N/A')
                 unit_display = data.get('unit', '')
             else: # Handle direct values like BP string or age
                 value_display = data
                 if data_type == "blood_pressure": unit_display = "mmHg"
                 elif data_type == "age": unit_display = "years"

             formatted_type = data_type.replace('_', ' ').title()
             data_confirmation += f'<li style="margin-bottom: 5px;">✅ <strong>{formatted_type}:</strong> {value_display} {unit_display}</li>'
        data_confirmation += '</ul><div style="font-size: 0.8rem; color: #555; margin-top: 8px;">This information helps provide context. Remember to consult a healthcare professional for interpretation.</div></div>'
        # Insert confirmation near the beginning but after the main response
        formatted_response_parts.insert(1, data_confirmation)


    # 4. Symptom Analysis Section
    potential_conditions, reported_symptoms = identify_potential_conditions(user_message)
    if reported_symptoms:
        symptom_section = '<div class="vital-card"><div class="vital-title">Symptom Analysis (Informational Only)</div>'
        symptom_section += '<div><strong>Reported Symptoms:</strong> '
        symptom_section += ''.join(f'<span class="symptom-badge">{s}</span>' for s in reported_symptoms)
        symptom_section += '</div>'

        if potential_conditions:
            symptom_section += '<div style="margin-top: 15px;">'
            symptom_section += '<p><strong>Potential related conditions (based on keywords, not a diagnosis):</strong></p><ul>'
            # Show top 2-3 potential conditions
            for pc in potential_conditions[:min(len(potential_conditions), 3)]:
                symptom_section += f'<li><strong>{pc["condition"]}:</strong> Matches {len(pc["matched_symptoms"])} symptoms ({pc["match_percentage"]}% symptom overlap).</li>'
            symptom_section += '</ul></div>'

            # Add self-care/when to see doctor if a strong match exists (optional)
            strong_match = next((pc for pc in potential_conditions if pc["match_percentage"] > 50 and len(pc["matched_symptoms"]) >= 2), None)
            if strong_match:
                 if strong_match.get("self_care"):
                     symptom_section += '<p><strong>General self-care ideas sometimes associated:</strong> ' + ', '.join(strong_match["self_care"][:3]) + '...</p>'
                 if strong_match.get("when_to_see_doctor"):
                     symptom_section += '<p><strong>Consider professional evaluation if experiencing:</strong> ' + ', '.join(strong_match["when_to_see_doctor"][:2]) + '...</p>'

        # CRITICAL DISCLAIMER for symptoms
        symptom_section += '<p style="margin-top: 15px; font-weight: bold; color: var(--danger-color);"><em>*** Disclaimer: This AI cannot diagnose. Symptom analysis is informational only based on common patterns. Please consult a healthcare professional for any health concerns or diagnosis. ***</em></p>'
        symptom_section += '</div>'
        formatted_response_parts.append(symptom_section)

    # 5. Proactive Wellness Tip (Occasionally)
    # Give a tip every ~5 interactions or if user asks for advice
    should_give_tip = (session.health_analytics["interaction_count"] % 5 == 1) or \
                      any(kw in user_message.lower() for kw in ["advice", "tip", "recommend", "improve", "help with"])

    if should_give_tip and WELLNESS_TIPS:
        try:
            # Try to select a category relevant to the conversation or profile
            relevant_category = None
            topics = extract_health_topics(user_message)
            if topics:
                topic_to_category = {"nutrition": "Nutrition", "exercise": "Physical Activity", "sleep": "Sleep", "mental_health": "Mental Wellbeing"}
                for topic in topics:
                     if topic in topic_to_category and topic_to_category[topic] in WELLNESS_TIPS:
                         relevant_category = topic_to_category[topic]
                         break
            # Fallback: cycle through categories or pick random
            if not relevant_category:
                 available_categories = list(WELLNESS_TIPS.keys())
                 relevant_category = random.choice(available_categories)


            if relevant_category and relevant_category in WELLNESS_TIPS:
                 selected_tip = random.choice(WELLNESS_TIPS[relevant_category])
                 tip_html = f'''
                 <div class="health-tips">
                     <div class="tip-title">Wellness Tip: {relevant_category}</div>
                     <p><strong>{selected_tip["tip"]}</strong></p>
                     <p style="font-size: 0.9em; color: #555;"><em>Benefit: {selected_tip["benefit"]}</em></p>
                 </div>
                 '''
                 formatted_response_parts.append(tip_html)
                 session.add_recommendation(selected_tip["tip"], relevant_category.lower())
        except Exception as e:
             logging.error(f"Error generating wellness tip: {e}")


    # 6. Relevant Health Resources
    resources = suggest_health_resources(user_message, reported_symptoms)
    if resources:
        resource_html = '<div class="resource-card"><strong>Helpful Resources (Informational Only):</strong><ul style="padding-left: 20px; margin-top: 10px;">'
        resource_count = 0
        max_resources_total = 4 # Limit total number of links

        for category, links in resources:
             if resource_count >= max_resources_total: break
             resource_html += f'<li class="resource-category" style="margin-bottom: 10px;">{category}'
             resource_html += '<ul style="list-style-type: disc; padding-left: 20px; margin-top: 5px;">'
             links_in_category = 0
             for link in links:
                  if resource_count >= max_resources_total: break
                  if isinstance(link, dict):
                     resource_html += f'<li style="margin-bottom: 3px;"><a href="{link["url"]}" target="_blank" rel="noopener noreferrer">{link["name"]}</a>: {link.get("description", "")}</li>'
                     resource_count += 1
                     links_in_category += 1
                  if links_in_category >= 2: break # Max 2 links per category
             resource_html += '</ul></li>'
        resource_html += '</ul></div>'
        formatted_response_parts.append(resource_html)

    return "".join(formatted_response_parts)


# --- Main Chatbot Logic ---

def health_chatbot(message: str, history: list, user_id: str = "default_user"):
    """Handles user message, interacts with LLM, formats response, updates session."""
    logging.info(f"Received message from user {user_id}: '{message[:50]}...'")

    if model is None and GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
         return "API Key not configured. Please set the GOOGLE_API_KEY environment variable."
    elif model is None:
         return "Chatbot model is not available due to configuration error. Please check logs."


    # Get or create user session
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
        logging.info(f"New session started for user {user_id}")
    session = user_sessions[user_id]

    # Preprocess message for emergencies and initial data extraction
    try:
        processed_message, is_emergency, detected_health_data = preprocess_health_query(message)
    except Exception as e:
        logging.error(f"Error during preprocessing for user {user_id}: {e}")
        # Fallback safely
        processed_message, is_emergency, detected_health_data = message, False, {}

    # Add user message to session history (internal)
    session.add_message("user", processed_message)

    # Extract and store health data mentioned in the message
    try:
        health_data_extracted = extract_health_data(session, processed_message, detected_health_data)
    except Exception as e:
        logging.error(f"Error during health data extraction for user {user_id}: {e}")
        health_data_extracted = False # Proceed without assuming data extraction worked


    # Build context for the LLM
    context = [
        # {"role": "system", "content": SYSTEM_PROMPT}, # System prompt often handled differently by genai models
    ]
    # Add profile context if available
    profile_items = []
    if session.user_profile["age"]: profile_items.append(f"Age: {session.user_profile['age']}")
    if session.user_profile["gender"]: profile_items.append(f"Gender: {session.user_profile['gender']}")
    if session.user_profile["chronic_conditions"]: profile_items.append(f"Conditions: {', '.join(session.user_profile['chronic_conditions'])}")
    if session.user_profile["allergies"]: profile_items.append(f"Allergies: {', '.join(session.user_profile['allergies'])}")
    if profile_items:
         profile_summary = "User Context: " + ". ".join(profile_items) + "."
         # Add as a user message for context, or potentially system if API supports it well
         context.append({"role": "user", "content": profile_summary})


    # Add conversation history (last ~5 exchanges) - map roles correctly
    history_limit = 10 # Number of individual messages (5 pairs)
    start_index = max(0, len(session.conversation_history) - history_limit)
    for item in session.conversation_history[start_index:-1]: # Exclude the latest user message already added
        role = "model" if item["role"] == "bot" else item["role"] # Map 'bot' to 'model'
        # Simple way to include history, might need more sophisticated context management
        context.append({"role": role, "content": item["message"]})


    # Add the current processed user message
    context.append({"role": "user", "content": processed_message})

    # --- Call the Generative AI Model ---
    bot_response_text = ""
    try:
        # Construct the prompt including the system instructions
        # Gemini API uses generate_content which takes the history directly
        # The system prompt needs to be integrated, often as the first 'user' or 'model' turn,
        # or via specific API parameters if available. Let's prepend it simply.

        # Prepending system prompt as a user message instructing the bot
        # Note: The effectiveness of this depends on the model's training.
        # Sometimes it's better placed differently or using API-specific features.
        full_context = [{"role": "user", "content": SYSTEM_PROMPT}]

        # Convert internal history roles ('bot') to 'model' for the API
        api_history = []
        for item in session.conversation_history:
             role = "model" if item["role"] == "bot" else "user"
             # Basic sanitization/shortening might be needed for long messages in history
             # For now, just append content
             api_history.append({"role": role, "content": item["message"]})


        # Use the internal session history for the API call
        response = model.generate_content(api_history) # Pass the converted history

        # Check for safety ratings or blocks if necessary (response.prompt_feedback)
        if response.prompt_feedback and response.prompt_feedback.block_reason:
             logging.warning(f"Prompt blocked for user {user_id}. Reason: {response.prompt_feedback.block_reason}")
             bot_response_text = "I cannot respond to that request due to safety guidelines."
        elif not response.candidates or not response.candidates[0].content.parts:
             logging.warning(f"No valid response candidate received from API for user {user_id}.")
             bot_response_text = "I'm sorry, I couldn't generate a response for that."
        else:
             bot_response_text = response.text
             logging.info(f"LLM response received for user {user_id}: '{bot_response_text[:50]}...'")

    except Exception as e:
        logging.error(f"Error calling Gemini API for user {user_id}: {e}")
        bot_response_text = f"I'm sorry, I encountered an error trying to generate a response. Please try again. (Error: {type(e).__name__})"

    # Format the raw text response with additional context
    try:
        formatted_response_html = format_health_response(
            session, bot_response_text, processed_message, is_emergency, health_data_extracted, detected_health_data
        )
    except Exception as e:
        logging.error(f"Error formatting health response for user {user_id}: {e}")
        formatted_response_html = bot_response_text # Fallback to raw text on formatting error


    # Add formatted bot response to session history (internal)
    session.add_message("bot", formatted_response_html) # Store the final HTML

    # Update analytics
    topics = extract_health_topics(processed_message)
    session.health_analytics["topics_discussed"].update(topics)
    # Calculate score periodically
    if session.health_analytics["interaction_count"] % 3 == 0:
         session.calculate_health_score()


    # Update Gradio history - Gradio expects a list of [user_msg, bot_msg] pairs
    # We reconstruct this from our internal session history for Gradio's display
    gradio_history = []
    user_msg = None
    for msg in session.conversation_history:
        if msg["role"] == "user":
            user_msg = msg["message"] # Store the user message
        elif msg["role"] == "bot" and user_msg is not None:
            # Pair the preceding user message with this bot response
            gradio_history.append([user_msg, msg["message"]])
            user_msg = None # Reset user message holder
        elif msg["role"] == "bot" and user_msg is None:
             # This might happen if the first message is from the bot (unlikely here)
             # Or if there are consecutive bot messages (e.g. due to errors/retries)
             # Decide how to handle - maybe display bot message alone?
             gradio_history.append([None, msg["message"]])


    return gradio_history # Return the updated history for Gradio Chatbot component


def extract_health_topics(message):
    """Extracts potential health topics mentioned in a message."""
    message_lower = message.lower()
    topics = set()
    health_topic_keywords = {
        "nutrition": ["diet", "food", "eating", "nutrient", "vitamin", "meal", "calorie", "nutrition"],
        "exercise": ["workout", "fitness", "exercise", "gym", "run", "cardio", "strength", "activity"],
        "sleep": ["sleep", "insomnia", "rest", "tired", "fatigue", "nap", "bedtime"],
        "mental_health": ["stress", "anxiety", "depression", "mood", "mental", "therapy", "emotion", "feeling"],
        "chronic_disease": ["diabetes", "hypertension", "asthma", "arthritis", "cholesterol", "heart disease", "cancer"],
        "medication": ["drug", "medicine", "prescription", "pill", "dose", "medication", "pharmacy"],
        "preventive_care": ["checkup", "screening", "vaccination", "prevention", "exam", "doctor visit"],
        "symptoms": ["pain", "ache", "fever", "cough", "headache", "nausea", "dizzy", "symptom", "feel sick"],
        "vitals": ["blood pressure", "heart rate", "temperature", "glucose", "sugar", "spo2", "oxygen"],
        "allergy": ["allergy", "allergic", "reaction", "hives"]
    }
    for topic, keywords in health_topic_keywords.items():
        if any(re.search(r'\b' + keyword + r'\b', message_lower, re.IGNORECASE) for keyword in keywords):
            topics.add(topic)
    return list(topics)


# --- Dashboard & Report Functions ---

def view_health_data(user_id: str = "default_user"):
    """Generates HTML for the Health Dashboard tab."""
    logging.info(f"Generating health data view for user {user_id}")
    if user_id not in user_sessions:
        return """
        <div class="health-report">
            <div class="report-header"><h2>Health Data Dashboard</h2></div>
            <p>No health data available yet. Start chatting to log information.</p>
        </div>"""

    session = user_sessions[user_id]

    # Check if *any* data relevant to the dashboard exists
    has_vitals = bool(session.vital_signs)
    has_symptoms = bool(session.symptom_log)
    has_meds = bool(session.medication_reminders)
    has_profile = any(v for k, v in session.user_profile.items() if k not in ['name', 'gender', 'last_checkup']) # Check for actual data points

    if not (has_vitals or has_symptoms or has_meds or has_profile):
        return """
        <div class="health-report">
            <div class="report-header"><h2>Health Data Dashboard</h2></div>
            <p>No health data has been recorded yet. Try sharing information like:</p>
            <ul>
                <li>"My blood pressure is 120/80."</li>
                <li>"I felt a slight headache yesterday."</li>
                <li>"I take Vitamin D daily."</li>
                <li>"My height is 170cm and weight is 65kg."</li>
            </ul>
        </div>"""

    # Calculate health score & analyze trends
    health_score = session.calculate_health_score()
    trends = analyze_health_trends(session)

    output = f"""
    <div class="health-report">
        <div class="report-header">
            <h2>Health Data Dashboard</h2>
            <p>Summary as of {get_current_timestamp()}</p>
        </div>"""

    # --- Health Score Section ---
    score_html = f"""
        <div class="report-section">
            <h3 class="report-section-title">Overall Health Score (Estimate)</h3>
            <div class="health-score">{health_score if health_score is not None else 'N/A'}/100</div>
            <div class="health-score-label">Based on recorded health data</div>
            <div class="progress-bar">
                <div class="progress-bar-inner" style="width: {health_score if health_score is not None else 0}%"></div>
            </div>"""
    if trends and "health_score" in trends:
        score_trend = trends["health_score"]
        direction = score_trend["direction"]
        if direction != 'stable':
            direction_icon = "↑" if direction == "improving" else "↓"
            direction_color = "var(--success-color)" if direction == "improving" else "var(--danger-color)"
            score_html += f"""
                <p style="text-align: center; color: {direction_color}; font-weight: 600; margin-top: 5px;">
                    {direction_icon} Trend: {direction.capitalize()} (Score changed by {score_trend.get('change', 0)})
                </p>"""
    score_html += "<p style='font-size: 0.8rem; color: #777; text-align: center;'>Note: This score is a simple estimate, not a clinical assessment.</p></div>"
    output += score_html


    # --- Vital Signs Section ---
    if has_vitals:
        vitals_html = """
        <div class="report-section">
            <h3 class="report-section-title">Recent Vital Signs</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">""" # Use grid layout

        for vital_type, measurements in session.vital_signs.items():
            if not measurements: continue
            latest = measurements[-1]
            formatted_type = vital_type.replace('_', ' ').title()
            in_range = is_vital_in_normal_range(vital_type, latest["value"])
            range_color = "var(--success-color)" if in_range is True else "var(--danger-color)" if in_range is False else "var(--text-color)" # Default color if range unknown

            card_html = f"""
            <div class="vital-card">
                <div class="vital-title">{formatted_type}</div>
                <div class="vital-value" style="color: {range_color};">{latest["value"]} {latest.get("unit", "")}</div>
                <div class="vital-timestamp">Recorded: {latest["timestamp"]}</div>"""

            # Add trend if available
            if trends and vital_type in trends:
                trend_data = trends[vital_type]
                if trend_data["direction"] != "stable":
                    trend_improving = trend_data.get("improving") # True, False, None
                    trend_icon = "↑" if trend_data["direction"] == "increasing" else "↓"
                    trend_text = trend_data["direction"].capitalize()
                    trend_color = "var(--success-color)" if trend_improving is True else "var(--danger-color)" if trend_improving is False else "var(--primary-color)"
                    card_html += f"""
                    <div style="margin-top: 8px; color: {trend_color}; font-weight: 500;">
                        Trend: {trend_icon} {trend_text}
                    </div>"""

            # Generate chart if enough data
            if len(measurements) >= 2:
                 chart_values = [m['value'] for m in measurements]
                 chart_dates = [m['timestamp'].split()[0] for m in measurements] # Use only date for x-axis labels
                 # Handle BP separately for plotting (e.g., plot systolic)
                 plot_type = formatted_type
                 plot_unit = latest.get("unit", "")
                 if vital_type == "blood_pressure":
                      try:
                          chart_values = [int(str(v).split('/')[0]) for v in chart_values] # Plot systolic
                          plot_type = "Systolic Blood Pressure"
                          plot_unit = "mmHg"
                      except Exception: chart_values = [] # Cannot plot BP if format wrong

                 chart_url = generate_health_chart(plot_type, chart_values, chart_dates, unit=plot_unit)
                 if chart_url:
                     card_html += f'<div class="health-chart"><img src="{chart_url}" alt="{formatted_type} Chart"></div>'

            card_html += "</div>"
            vitals_html += card_html

        vitals_html += "</div></div>"
        output += vitals_html


    # --- Symptoms Section ---
    if has_symptoms:
        symptoms_html = """
        <div class="report-section">
            <h3 class="report-section-title">Recent Symptoms Log</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Symptom</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Severity</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Date</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Notes</th>
                    </tr>
                </thead>
                <tbody>"""
        # Show last 5 symptoms max
        for symptom in reversed(session.symptom_log[-5:]):
            severity = symptom.get("severity", "N/A")
            severity_color = "var(--danger-color)" if severity == "severe" else "var(--warning-color)" if severity == "moderate" else "var(--success-color)" if severity == "mild" else "var(--text-color)"
            symptoms_html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{symptom["symptom"]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">
                        <span style="color: {severity_color}; font-weight: 500;">{severity.capitalize()}</span>
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{symptom["timestamp"]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{symptom.get("related_factors", "")}</td>
                </tr>"""
        symptoms_html += """</tbody></table>"""

        # Add symptom trend analysis if available
        if trends and "symptoms" in trends and trends["symptoms"]["most_frequent"]:
             symptom_trends = trends["symptoms"]
             symptoms_html += """
                <div style="margin-top: 16px;">
                    <h4>Most Frequent Symptoms (Last 14 Days)</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;">"""
             for symptom, count in symptom_trends["most_frequent"]:
                 symptoms_html += f"""
                    <div style="background-color: var(--light-bg); padding: 6px 12px; border-radius: 16px; font-size: 0.9rem;">
                        {symptom} <span style="font-weight: 600; color: var(--primary-color); font-size: 0.8rem;">({count} reports)</span>
                    </div>"""
             symptoms_html += """</div></div>"""

        symptoms_html += "</div>"
        output += symptoms_html

    # --- Medications Section ---
    if has_meds:
        meds_html = """
        <div class="report-section">
            <h3 class="report-section-title">Medications Logged</h3>"""
        for med in session.medication_reminders:
             meds_html += f"""
            <div class="medication-reminder">
                <div class="medication-name">{med.get("medication", "N/A")} ({med.get("dosage", "N/A")})</div>
                <div class="medication-schedule">Schedule: {med.get("schedule", "N/A")}</div>
                {f'<div style="font-size: 0.85rem; color: #666;">Duration: {med["duration"]}</div>' if med.get("duration") else ''}
                {f'<div style="font-size: 0.85rem; color: #666;">Notes: {med["notes"]}</div>' if med.get("notes") else ''}
            </div>"""
        meds_html += "</div>"
        output += meds_html


     # --- User Profile Summary ---
    if has_profile:
        profile_html = """
        <div class="report-section">
            <h3 class="report-section-title">User Profile Summary</h3>
            <table style="width: 100%; border-collapse: collapse;">"""
        profile_data_to_show = {
             "Age": session.user_profile.get("age"),
             "Height": f"{session.user_profile.get('height_cm')} cm" if session.user_profile.get('height_cm') else None,
             "Weight": f"{session.user_profile.get('weight_kg')} kg" if session.user_profile.get('weight_kg') else None,
             "BMI": session.calculate_bmi(),
             "BMI Category": session.get_bmi_category() if session.calculate_bmi() is not None else None,
             "Allergies": ', '.join(session.user_profile.get("allergies", [])),
             "Chronic Conditions": ', '.join(session.user_profile.get("chronic_conditions", [])),
             # "Current Medications": ', '.join(session.user_profile.get("current_medications", [])) # Redundant with meds section?
        }
        for key, value in profile_data_to_show.items():
            if value and value != 'N/A' and value != '': # Only show if there's data
                 profile_html += f"""
                    <tr>
                        <td style="padding: 6px 0; font-weight: 500; width: 35%;">{key}:</td>
                        <td style="padding: 6px 0;">{value}</td>
                    </tr>"""
        profile_html += "</table></div>"
        output += profile_html


    # --- Footer/Export ---
    output += """
        <div style="margin-top: 24px; text-align: center;">
            <button class="pdf-export-btn" onclick="alert('PDF export functionality is not implemented in this demo.')">
                <span class="pdf-icon">📄</span> Export Dashboard as PDF
            </button>
        </div>
    </div>""" # Close health-report div

    return output


def generate_health_report(user_id: str = "default_user"):
    """Generates a more comprehensive HTML health report."""
    logging.info(f"Generating comprehensive health report for user {user_id}")
    if user_id not in user_sessions:
        return """
        <div class="health-report">
            <div class="report-header"><h2>Comprehensive Health Report</h2></div>
            <p>No health data available to generate a report. Please chat first.</p>
        </div>"""

    session = user_sessions[user_id]
    report = f"""
    <div class="health-report">
        <div class="report-header">
            <h2>Comprehensive Health Report</h2>
            <p>Generated for User {session.user_id} on {get_current_timestamp()}</p>
        </div>"""

    # --- User Profile Section ---
    report += """
        <div class="report-section">
            <h3 class="report-section-title">Personal Information</h3>
            <table style="width: 100%; border-collapse: collapse;">"""
    profile_details = {
         "Age": session.user_profile.get("age"),
         "Gender": session.user_profile.get("gender"),
         "Height": f"{session.user_profile.get('height_cm')} cm" if session.user_profile.get('height_cm') else None,
         "Weight": f"{session.user_profile.get('weight_kg')} kg" if session.user_profile.get('weight_kg') else None,
         "BMI": session.calculate_bmi(),
         "BMI Category": session.get_bmi_category() if session.calculate_bmi() is not None else None,
         "Allergies": ', '.join(session.user_profile.get("allergies", []) or ["None reported"]),
         "Chronic Conditions": ', '.join(session.user_profile.get("chronic_conditions", []) or ["None reported"]),
         "Current Medications": ', '.join(session.user_profile.get("current_medications", []) or ["None reported"]),
         # "Last Checkup": session.user_profile.get("last_checkup") # Often not captured
    }
    for key, value in profile_details.items():
        if value is not None and value != '':
             report += f"""
                <tr>
                    <td style="padding: 8px; font-weight: 500; width: 30%; border-bottom: 1px solid #eee;">{key}:</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{value}</td>
                </tr>"""
    report += "</table></div>"

    # --- Health Score & Trend ---
    health_score = session.calculate_health_score()
    trends = analyze_health_trends(session) # Analyze trends
    report += f"""
        <div class="report-section">
            <h3 class="report-section-title">Health Score Assessment</h3>
            <div class="health-score">{health_score if health_score is not None else 'N/A'}/100</div>
            <div class="progress-bar">
                <div class="progress-bar-inner" style="width: {health_score if health_score is not None else 0}%"></div>
            </div>"""
    # Add interpretation
    score_interp = "This score is a simplified estimate based on logged data."
    if health_score is not None:
        if health_score >= 85: score_interp = "Indicates generally positive health indicators based on available data."
        elif health_score >= 70: score_interp = "Indicates fair health indicators; some areas might warrant attention."
        else: score_interp = "Suggests potential areas for health focus or review with a professional."
    report += f"<p style='text-align: center; font-size: 0.9rem; margin-top: 5px;'>{score_interp}</p>"

    # Add trend if available
    if trends and "health_score" in trends:
        score_trend = trends["health_score"]
        direction = score_trend["direction"]
        if direction != 'stable':
            direction_icon = "↑" if direction == "improving" else "↓"
            direction_color = "var(--success-color)" if direction == "improving" else "var(--danger-color)"
            report += f"""
                <p style="text-align: center; color: {direction_color}; font-weight: 600; margin-top: 5px;">
                    {direction_icon} Trend is {direction.capitalize()} compared to previous score.
                </p>"""
    report += "</div>"


    # --- Vital Signs Summary & Trends ---
    if session.vital_signs:
        report += """
            <div class="report-section">
                <h3 class="report-section-title">Vital Signs Summary</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Measurement</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Latest Value</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Date</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Status</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Trend</th>
                        </tr>
                    </thead>
                    <tbody>"""
        for vital_type, measurements in session.vital_signs.items():
            if not measurements: continue
            latest = measurements[-1]
            formatted_type = vital_type.replace('_', ' ').title()
            in_range = is_vital_in_normal_range(vital_type, latest["value"])
            status = "Normal" if in_range is True else "Check Range" if in_range is False else "N/A"
            status_color = "var(--success-color)" if status == "Normal" else "var(--warning-color)" if status == "Check Range" else "var(--text-color)"

            trend_text = "Stable"
            trend_color = "var(--text-color)"
            if trends and vital_type in trends:
                 trend_data = trends[vital_type]
                 if trend_data["direction"] != "stable":
                      trend_improving = trend_data.get("improving")
                      trend_icon = "↑" if trend_data["direction"] == "increasing" else "↓"
                      trend_text = f'{trend_icon} {trend_data["direction"].capitalize()}'
                      trend_color = "var(--success-color)" if trend_improving is True else "var(--danger-color)" if trend_improving is False else "var(--primary-color)"


            report += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{formatted_type}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{latest["value"]} {latest.get("unit", "")}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{latest["timestamp"]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: {status_color};">{status}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: {trend_color};">{trend_text}</td>
                </tr>"""
        report += "</tbody></table></div>"


    # --- Symptom Summary ---
    if session.symptom_log:
        report += """
            <div class="report-section">
                <h3 class="report-section-title">Symptom Log Summary (Last 10)</h3>
                <table style="width: 100%; border-collapse: collapse;">
                     <thead>
                        <tr>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Symptom</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Severity</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Date</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Notes</th>
                        </tr>
                    </thead>
                    <tbody>"""
        for symptom in reversed(session.symptom_log[-10:]):
            severity = symptom.get("severity", "N/A")
            severity_color = "var(--danger-color)" if severity == "severe" else "var(--warning-color)" if severity == "moderate" else "var(--success-color)" if severity == "mild" else "var(--text-color)"
            report += f"""
                 <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{symptom["symptom"]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;"><span style="color: {severity_color};">{severity.capitalize()}</span></td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{symptom["timestamp"]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{symptom.get("related_factors", "")}</td>
                 </tr>"""
        report += "</tbody></table>"
        # Add frequency analysis if available
        if trends and "symptoms" in trends and trends["symptoms"]["most_frequent"]:
            report += "<h4 style='margin-top: 15px; margin-bottom: 5px;'>Most Frequent Symptoms (Last 14 Days):</h4>"
            report += ", ".join([f"{s} ({c})" for s, c in trends["symptoms"]["most_frequent"]])
        report += "</div>"

    # --- Medication Summary ---
    if session.medication_reminders:
        report += """
            <div class="report-section">
                <h3 class="report-section-title">Medication Log</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                         <tr>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Medication</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Dosage</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Schedule</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid var(--primary-color);">Notes</th>
                        </tr>
                    </thead>
                    <tbody>"""
        for med in session.medication_reminders:
             report += f"""
                 <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{med.get("medication", "N/A")}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{med.get("dosage", "N/A")}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{med.get("schedule", "N/A")}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{med.get("notes", "")}</td>
                 </tr>"""
        report += "</tbody></table></div>"


    # --- Wellness Activity Summary ---
    if session.wellness_activities:
         report += """
             <div class="report-section">
                 <h3 class="report-section-title">Wellness Activities Logged</h3>"""
         # Simple list for report
         activity_summary = {}
         for activity in session.wellness_activities:
              atype = activity["activity_type"].replace('_', ' ').title()
              activity_summary[atype] = activity_summary.get(atype, 0) + 1
         if activity_summary:
              report += "<ul style='list-style: disc; padding-left: 20px;'>"
              for atype, count in activity_summary.items():
                  report += f"<li>{atype}: {count} times logged</li>"
              report += "</ul>"
         else: report += "<p>No wellness activities logged recently.</p>"
         report += "</div>"

    # --- General Recommendations (Placeholder) ---
        # --- General Recommendations (Placeholder) ---
    report += """
        <div class="report-section">
            <h3 class="report-section-title">General Recommendations</h3>
            <p>Based on the logged data, consider these general points. <strong>Always consult your doctor for personalized medical advice.</strong></p>
            <ul>"""

    # Simple logic based on report sections
    action_items_report = []

    # Check vital signs for out-of-range values
    vital_out_of_range = False
    try: # Add try-except for safety during iteration
        for vital_type, measurements in session.vital_signs.items():
            if measurements: # Check if list is not empty
                latest = measurements[-1]
                # Ensure 'value' key exists before checking range
                if 'value' in latest:
                    in_range_status = is_vital_in_normal_range(vital_type, latest['value'])
                    if in_range_status is False: # Explicitly check for False (out of range)
                        vital_out_of_range = True
                        logging.debug(f"Vital sign '{vital_type}' found out of range: {latest['value']}")
                        break # Found one, no need to check further
    except Exception as e:
        logging.error(f"Error checking vital sign ranges for report recommendations: {e}")

    if vital_out_of_range:
        action_items_report.append("Review vital signs potentially outside normal ranges with your healthcare provider.")

    # Check symptoms for severity
    try:
        if any(s.get("severity") == "severe" for s in session.symptom_log[-10:]): # Check last 10 symptoms
             action_items_report.append("Discuss any 'Severe' symptoms logged with your healthcare provider.")
    except Exception as e:
        logging.error(f"Error checking symptom severity for report recommendations: {e}")

    # Check wellness activities
    try:
        # Check activities logged in the last 14 days
        recent_activities_count = sum(1 for a in session.wellness_activities
                                    if (datetime.now() - datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")).days <= 14)
        if recent_activities_count < 3: # Arbitrary threshold for recent activity
             action_items_report.append("Consider incorporating regular wellness activities like exercise or mindfulness into your routine.")
    except Exception as e:
         logging.error(f"Error checking wellness activities for report recommendations: {e}")


    # Check if basic profile info is missing
    if not session.user_profile.get("age") or not session.user_profile.get("height_cm") or not session.user_profile.get("weight_kg"):
         action_items_report.append("Consider sharing basic profile information (age, height, weight) for better context during chats.")

    # Add generic good advice if few specific items were generated
    if len(action_items_report) < 2:
        action_items_report.append("Maintain a balanced diet and stay hydrated.")
        action_items_report.append("Ensure adequate sleep (typically 7-9 hours for adults).")
        action_items_report.append("Schedule regular check-ups with your healthcare provider for preventive care.")

    # Add the generated action items to the report HTML
    for item in action_items_report[:5]: # Limit suggestions shown in report
         report += f"<li style='margin-bottom: 8px;'>{item}</li>"
    report += "</ul></div>" # Close the recommendations list and section div


    # --- Disclaimer ---
    report += """
        <div class="report-section" style="margin-top: 32px; padding: 16px; background-color: #fff8e1; border-left: 4px solid var(--warning-color); border-radius: 4px;">
            <h3 class="report-section-title" style="color: var(--warning-color);">Important Disclaimer</h3>
            <p>This report is automatically generated based on information shared during your chat session. <strong>It is NOT a medical diagnosis or a substitute for professional medical advice, diagnosis, or treatment.</strong> Information may be incomplete or misinterpreted by the AI. Always consult with a qualified healthcare provider regarding any medical conditions or treatment options.</p>
        </div>

        <div style="margin-top: 24px; text-align: center;">
            <button class="pdf-export-btn" onclick="alert('PDF export functionality is not implemented in this demo.')">
                <span class="pdf-icon">📄</span> Export Report as PDF
            </button>
        </div>
    </div>""" # Close health-report div

    return report


def process_uploaded_file(file: gr.File, user_id: str = "default_user"):
    """Handles uploaded files (placeholder implementation)."""
    if file is None:
        return "<p>No file uploaded. Please select a file.</p>"

    logging.info(f"Processing uploaded file '{file.name}' for user {user_id}")

    # In a real application, you would:
    # 1. Securely save the file temporarily.
    # 2. Use libraries like PyPDF2, python-docx, openpyxl, Pillow, OCR (like Tesseract via pytesseract)
    #    to extract text or data based on file type.
    # 3. Parse the extracted content (potentially using complex regex or another AI model pass)
    #    to identify relevant health information (labs, diagnoses, meds).
    # 4. Update the user's session data (e.g., add extracted vitals, conditions).
    # 5. Provide a summary of extracted information or analysis results.
    # 6. Securely delete the temporary file.

    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()

    output = f"""
    <div class="health-report">
        <div class="report-header"><h2>File Processing Result</h2></div>
        <p>Received file: <strong>{os.path.basename(file_path)}</strong> (Type: {file_ext})</p>"""

    if file_ext in [".pdf", ".txt", ".md", ".docx", ".jpg", ".png"]:
         output += """
            <div class="report-section">
                <h3 class="report-section-title">Analysis (Simulated)</h3>
                <p><strong>Demo Note:</strong> File content analysis is not implemented in this version.</p>
                <p>In a full implementation, the system would attempt to extract key health information like:</p>
                <ul>
                    <li>Diagnoses and Conditions</li>
                    <li>Lab Results (e.g., Cholesterol, Blood Counts)</li>
                    <li>Medications and Dosages</li>
                    <li>Vital Signs Records</li>
                    <li>Procedure Notes</li>
                </ul>
                <p>Extracted data would be added to your health dashboard and report.</p>
                <p>You could then ask questions like: <em>"Summarize the key findings from the uploaded report"</em> or <em>"Explain the lab results in the document."</em></p>
            </div>"""
    elif file_ext in [".csv", ".xlsx", ".xls"]:
         output += """
            <div class="report-section">
                <h3 class="report-section-title">Data File Analysis (Simulated)</h3>
                 <p><strong>Demo Note:</strong> Spreadsheet analysis is not implemented in this version.</p>
                 <p>In a full implementation, the system would parse the data to identify trends, generate visualizations, and integrate the information (e.g., blood sugar readings over time) into your dashboard.</p>
            </div>"""
    else:
        output += f"""
            <p style="color: var(--danger-color);">File type '{file_ext}' is not currently supported for analysis in this demo.</p>
            <p>Supported types (simulation): PDF, TXT, DOCX, JPG, PNG, CSV, XLSX.</p>
            """

    output += "</div>"
    return output


# --- Gradio Interface Definition ---

# JS function to set textbox value and click submit
# (Using _js parameter for simple cases, more complex JS might need different approach)
set_and_submit_js = """
(txt) => {
    const gradio_app_el = document.querySelector('gradio-app'); // Find the top-level Gradio element
    const msg_textbox = gradio_app_el.querySelector('#message-input textarea'); // Find the textarea by ID
    const submit_button = gradio_app_el.querySelector('#health-submit-btn'); // Find button by ID

    if (msg_textbox && submit_button) {
        msg_textbox.value = txt; // Set the text

        // Dispatch an 'input' event to ensure Gradio recognizes the change
        const inputEvent = new Event('input', { bubbles: true });
        msg_textbox.dispatchEvent(inputEvent);

        // Short delay seems to help Gradio UI catch up sometimes before click
        setTimeout(() => { submit_button.click(); }, 100);
    } else {
        console.error('Could not find message textbox or submit button.');
    }
    return txt; // Return the original text (or updated text) back to Gradio if needed
}
"""


with gr.Blocks(css=custom_css, title="MediGuide AI", theme=gr.themes.Soft()) as demo:
    user_id_state = gr.State("user_" + str(random.randint(1000, 9999))) # Simple unique ID per session

    # Header
    gr.HTML("""
    <div class="health-header">
        <div style="display: flex; align-items: center;">
            <svg class="health-logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21Z" stroke="white" stroke-width="2"/><path d="M10 8H14V16H10V8Z" fill="white"/><path d="M8 10H16V14H8V10Z" fill="white"/></svg>
            <h1>MediGuide AI Health Assistant</h1>
        </div>
        <div><span style="background-color: rgba(255, 255, 255, 0.2); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;">Powered by Gemini</span></div>
    </div>""")

    # Intro Message
    gr.HTML("""
    <div style="background-color: var(--primary-color); color: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: var(--card-shadow);">
        <h2 style="margin-top: 0; color: white;">Welcome to MediGuide</h2>
        <p>Your AI-powered healthcare assistant for personalized health information, guidance, and tracking.</p>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
            <div style="background-color: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px; flex: 1; min-width: 200px;">
                <h3 style="margin-top: 0; font-size: 1rem; color: white;">🔍 Ask health questions</h3>
                <p style="font-size: 0.9rem; margin-bottom: 0;">Get reliable information about symptoms, conditions, medications, and wellness.</p>
            </div>
            <div style="background-color: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px; flex: 1; min-width: 200px;">
                <h3 style="margin-top: 0; font-size: 1rem; color: white;">📊 Track your health</h3>
                <p style="font-size: 0.9rem; margin-bottom: 0;">MediGuide automatically logs vital signs, symptoms, and medications you mention.</p>
            </div>
            <div style="background-color: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px; flex: 1; min-width: 200px;">
                <h3 style="margin-top: 0; font-size: 1rem; color: white;">📋 Generate reports</h3>
                <p style="font-size: 0.9rem; margin-bottom: 0;">Visualize your health data and get personalized insights and recommendations.</p>
            </div>
        </div>
        <p style="color: rgba(255,255,255,0.8); font-style: italic; margin-top: 15px; font-size: 0.9rem;">Important: MediGuide is not a replacement for professional medical care. For serious concerns, please consult a healthcare provider.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Chat Interface
            chatbot_display = gr.Chatbot(
                height=550,
                label="MediGuide Chat",
                elem_id="health-chatbot", # Keep ID for CSS/JS
                avatar_images=("./static/user_avatar.png", "./static/bot_avatar.png"), # Provide paths to avatar images if available
                bubble_full_width=False,
                show_copy_button=True,
               
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a health question or log info (e.g., 'My temp is 99.1 F', 'I have a headache')...",
                    container=False,
                    scale=7,
                    elem_id="message-input", # Keep ID
                    show_label=False,
                    lines=1,
                    max_lines=5
                )
                submit_btn = gr.Button("Send", variant="primary", elem_classes="health-submit-btn", min_width=80) # Use elem_classes
                # clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes="health-clear-btn", min_width=80) # Clear button (optional)

            # Example buttons using the JS trick
            with gr.Row():
                 ex1 = gr.Button("What are allergy symptoms?", elem_classes="example-btn")
                 ex2 = gr.Button("My BP is 120/80", elem_classes="example-btn")
                 ex3 = gr.Button("Tips to improve sleep?", elem_classes="example-btn")
                 ex4 = gr.Button("Log mild headache", elem_classes="example-btn")


        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("📊 Health Dashboard"):
                    health_data_output = gr.HTML("<p style='text-align: center; padding: 20px; color: #777;'>Click 'View/Update Dashboard' to load your health summary.</p>")
                    view_data_btn = gr.Button("🔄 View/Update Dashboard", variant="secondary") # Use icon

                with gr.TabItem("📋 Health Report"):
                    report_output = gr.HTML("<p style='text-align: center; padding: 20px; color: #777;'>Click 'Generate Report' for a detailed summary.</p>")
                    generate_report_btn = gr.Button("📄 Generate Comprehensive Report", variant="secondary")

                with gr.TabItem("⬆️ Upload Records"):
                     gr.HTML("""
                    <div style="margin-bottom: 15px; padding: 10px;">
                        <h4 style="margin-top: 0;">Upload Medical Document (Demo)</h4>
                        <p style="font-size: 0.9rem; color: #555;">Upload PDF, TXT, JPG, PNG files. (Analysis is simulated).</p>
                    </div>""")
                     # Accept common document/image types
                     file_upload = gr.File(label="Select file", file_types=["pdf", "txt", "jpg", "jpeg", "png", "csv", "xlsx"], height=100)
                     upload_button = gr.Button("⚙️ Process Uploaded File", variant="secondary")
                     upload_output = gr.HTML("<p style='text-align: center; padding: 20px; color: #777;'>Upload a file and click 'Process'.</p>")

                with gr.TabItem("ℹ️ Resources & Emergency"):
                    gr.HTML("""
                    <div style="padding: 15px;">
                        <h3 style="margin-top: 0; color: var(--danger-color);">🚨 Emergency Contacts 🚨</h3>
                         <div class="resource-card" style="border-left-color: var(--danger-color);">
                            <ul>
                                <li><strong>USA:</strong> Call <strong>911</strong> immediately for emergencies.</li>
                                <li><strong>Crisis & Suicide Lifeline (USA):</strong> Call or Text <strong>988</strong>.</li>
                                <li><strong>Poison Control (USA):</strong> Call <strong>1-800-222-1222</strong>.</li>
                                <li><em>(Know your local emergency numbers outside the USA)</em></li>
                            </ul>
                        </div>

                        <h3 style="margin-top: 20px; color: var(--primary-color);">Trusted Health Resources</h3>
                        <div class="resource-card">
                            <h4 class="resource-category">General Health</h4>
                            <ul>
                                <li><a href="https://www.cdc.gov/" target="_blank" rel="noopener noreferrer">CDC (USA)</a></li>
                                <li><a href="https://www.who.int/" target="_blank" rel="noopener noreferrer">WHO (Global)</a></li>
                                <li><a href="https://www.mayoclinic.org/" target="_blank" rel="noopener noreferrer">Mayo Clinic</a></li>
                                <li><a href="https://medlineplus.gov/" target="_blank" rel="noopener noreferrer">MedlinePlus</a></li>
                            </ul>
                        </div>
                         <div class="resource-card">
                            <h4 class="resource-category">Mental Health</h4>
                            <ul>
                                <li><a href="https://www.nimh.nih.gov/" target="_blank" rel="noopener noreferrer">NIMH (USA)</a></li>
                                <li><a href="https://www.samhsa.gov/find-help/national-helpline" target="_blank" rel="noopener noreferrer">SAMHSA Helpline (USA)</a></li>
                                <li><a href="https://www.nami.org/" target="_blank" rel="noopener noreferrer">NAMI (USA)</a></li>
                            </ul>
                        </div>
                         <!-- Add more resource categories as needed -->
                    </div>""")

                # Settings Tab - Placeholder
                with gr.TabItem("⚙️ Settings (Demo)"):
                     gr.HTML("""
                    <div style="padding: 15px;">
                        <h3 style="margin-top: 0; color: var(--primary-color);">Preferences (Demo)</h3>
                        <p>Settings for notifications and data are not functional in this demo.</p>
                        <div style="margin: 20px 0;">
                            <input type="checkbox" id="demo-cb1" checked disabled><label for="demo-cb1" style="margin-left: 5px;"> Medication Reminders</label><br>
                            <input type="checkbox" id="demo-cb2" checked disabled><label for="demo-cb2" style="margin-left: 5px;"> Checkup Reminders</label><br>
                            <input type="checkbox" id="demo-cb3" checked disabled><label for="demo-cb3" style="margin-left: 5px;"> Health Tips</label>
                        </div>
                        <h3 style="color: var(--primary-color); margin-top: 30px;">Data Management (Demo)</h3>
                        <button class="gradio-button health-clear-btn" onclick="alert('Clearing data is not implemented in this demo.')" style="background-color: var(--warning-color) !important; color: white !important;">Clear My Chat Data (Simulated)</button>
                        <p style="font-size: 0.8rem; color: #777; margin-top: 5px;">In a real app, this would clear your session.</p>
                    </div>""")


    # --- Event Handlers ---

    # Function to clear message input and return updated history
    def clear_input_and_update_history(history_list):
        return "", history_list # Return empty string for msg_input, and the history for chatbot

    # When user submits message (Enter key)
    msg_input.submit(
        health_chatbot,
        inputs=[msg_input, chatbot_display, user_id_state],
        outputs=[chatbot_display] # Only update chatbot
    ).then(
        lambda: "", # Function to return empty string
        inputs=None,
        outputs=[msg_input], # Clear the input textbox
        queue=False # Run immediately after chatbot update
    )

    # When user clicks Send button
    submit_btn.click(
        health_chatbot,
        inputs=[msg_input, chatbot_display, user_id_state],
        outputs=[chatbot_display]
     ).then(
        lambda: "",
        inputs=None,
        outputs=[msg_input],
        queue=False
    )

    # Dashboard and Report buttons
    view_data_btn.click(view_health_data, inputs=[user_id_state], outputs=[health_data_output])
    generate_report_btn.click(generate_health_report, inputs=[user_id_state], outputs=[report_output])

    # File upload button
    upload_button.click(process_uploaded_file, inputs=[file_upload, user_id_state], outputs=[upload_output])

    click_submit_js = """
    () => {
        // Find the Gradio root element first to scope the query
        const gradio_app_el = document.querySelector('gradio-app');
        if (!gradio_app_el) {
            console.error('Gradio root element not found.');
            return;
        }
        const submit_button = gradio_app_el.querySelector('#health-submit-btn'); // Find button by ID within the app
        if (submit_button) {
             // Use a small timeout to allow Gradio to process the textbox update first
            setTimeout(() => { submit_button.click(); }, 100);
        } else {
            console.error('Could not find submit button (#health-submit-btn).');
        }
    }
    """

    # Example buttons - using fn -> outputs, then js trigger
    ex1.click(
        fn=lambda: "What are the symptoms of seasonal allergies?", # Python fn returns the text
        inputs=None,                                            # Lambda takes no input
        outputs=[msg_input],                                    # Update the msg_input textbox
        js=click_submit_js                                      # Run JS to click submit AFTER update
    )
    ex2.click(
        fn=lambda: "My blood pressure is 120/80. Is that normal?",
        inputs=None,
        outputs=[msg_input],
        js=click_submit_js
    )
    ex3.click(
        fn=lambda: "How can I improve my sleep quality?",
        inputs=None,
        outputs=[msg_input],
        js=click_submit_js
    )
    ex4.click(
        fn=lambda: "I have a mild headache today.",
        inputs=None,
        outputs=[msg_input],
        js=click_submit_js
    )

    # Optional: Clear button functionality
    # def clear_chat_history(user_id):
    #     if user_id in user_sessions:
    #         user_sessions[user_id] = UserSession(user_id) # Reset session
    #         logging.info(f"Chat history cleared for user {user_id}")
    #     return [], "" # Return empty history for chatbot, empty string for message box
    # clear_btn.click(clear_chat_history, inputs=[user_id_state], outputs=[chatbot_display, msg_input], queue=False)


    # Footer
    gr.HTML("""
    <div class="health-footer">
        <p>MediGuide AI Health Assistant Demo | For Informational Purposes Only</p>
        <p><strong>Disclaimer:</strong> Not a substitute for professional medical advice. Consult a healthcare provider for medical concerns. In emergencies, call 911 or your local emergency number.</p>
    </div>""")


# --- Launch the App ---
if __name__ == "__main__":
    # Create dummy static files if they don't exist (for Gradio avatar paths)
    os.makedirs("./static", exist_ok=True)
    if not os.path.exists("./static/user_avatar.png"):
        # Create a simple placeholder image
        try:
            img = Image.new('RGB', (60, 60), color = (73, 109, 137))
            img.save("./static/user_avatar.png")
        except Exception as e:
            logging.warning(f"Could not create placeholder user avatar: {e}")
    if not os.path.exists("./static/bot_avatar.png"):
         try:
            img = Image.new('RGB', (60, 60), color = (0, 105, 179)) # Use primary color
            img.save("./static/bot_avatar.png")
         except Exception as e:
            logging.warning(f"Could not create placeholder bot avatar: {e}")


    logging.info("Launching Gradio Interface...")
    demo.queue().launch(
        # share=True, # Creates a public link - Use with caution due to API key/data
        debug=True, # Set to True for more detailed Gradio errors
        share=True,
        allowed_paths=["./static"] # Allow access to the static folder for avatars
    )
