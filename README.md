# MediGuide AI Health Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MediGuide** is a conversational AI health assistant demo built with Python, Gradio, and powered by Google's Gemini language model. It aims to provide general health information, track user-mentioned health data (like vital signs and symptoms), visualize trends, and generate health summaries *for informational purposes only*.

**[Insert Screenshot Here - Recommended]**
*Replace this text with a screenshot of the running application. You can drag & drop images into GitHub's markdown editor.*
`![MediGuide Screenshot](link_to_your_screenshot.png)`

---

## 🚨 IMPORTANT DISCLAIMER 🚨

**MediGuide is an AI demonstration application and is NOT a substitute for professional medical advice, diagnosis, or treatment.**

*   **Do NOT use this application for medical emergencies.** If you are experiencing a medical emergency, call 911 (or your local emergency number) immediately or go to the nearest emergency room.
*   The information provided by MediGuide is for general knowledge and informational purposes only, and does not constitute medical advice.
*   MediGuide **cannot** diagnose medical conditions or recommend specific treatments. AI can make mistakes and misinterpret information.
*   Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or heard from this application.
*   Data entered into this demo application is stored locally within the application's session during runtime (in the `user_sessions` dictionary) and is **not intended for secure, long-term storage** of sensitive health information in this basic implementation.

---

## ✨ Features

*   **Conversational Interface:** Chat with the AI about health topics, symptoms, and wellness using a Gradio-based UI.
*   **Powered by Gemini:** Utilizes the `gemini-flash` model for generating responses.
*   **Health Data Extraction:** Attempts to automatically extract vital signs (blood pressure, temperature, heart rate, etc.), symptoms, medications, and basic profile information (age, height, weight) mentioned in the chat using regular expressions.
*   **Symptom Analysis (Informational):** Identifies mentioned symptoms and suggests potentially related common conditions based on keywords (*not a diagnosis*).
*   **Vital Sign Tracking & Visualization:** Stores logged vital signs and generates simple trend charts using Matplotlib.
*   **Health Dashboard:** Provides a summarized view of logged vitals, recent symptoms, medications, profile data, and an estimated health score (heuristic, *not clinical*).
*   **Comprehensive Health Report:** Generates a more detailed report summarizing all logged information and trends.
*   **Wellness Tips:** Offers general wellness tips periodically or upon request.
*   **Resource Suggestions:** Recommends links to reliable health organizations (CDC, WHO, Mayo Clinic, etc.).
*   **Basic Emergency Keyword Detection:** Identifies keywords suggesting a potential emergency and strongly advises seeking immediate professional help.
*   **(Simulated) Document Upload:** Includes a placeholder UI for uploading medical documents (analysis is not implemented in this demo).

---

## 🛠️ Technology Stack

*   **Language:** Python 3.x
*   **AI Model:** Google Gemini API (`gemini-flash`)
*   **Web Framework/UI:** Gradio
*   **Data Handling:** Basic Python Dictionaries, Regex (`re`)
*   **Plotting:** Matplotlib
*   **Utilities:** NumPy, Pillow (for placeholder images)

---

## ⚙️ Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/mediguide-ai.git # Replace with your repo URL
    cd mediguide-ai
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    gradio
    google-generativeai
    matplotlib
    numpy
    Pillow
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *(Alternatively, run `pip install gradio google-generativeai matplotlib numpy Pillow`)*

4.  **Get Google Gemini API Key:**
    *   Obtain an API key from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    *   Make sure the Gemini API is enabled for your project.

5.  **Set Environment Variable:**
    Set the `GOOGLE_API_KEY` environment variable to your obtained key. **Do not hardcode your key directly into the script.**
    *   **Linux/macOS:**
        ```bash
        export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
        ```
    *   **Windows (Command Prompt):**
        ```bash
        set GOOGLE_API_KEY=YOUR_API_KEY_HERE
        ```
    *   **Windows (PowerShell):**
        ```powershell
        $env:GOOGLE_API_KEY='YOUR_API_KEY_HERE'
        ```

---

## ▶️ Running the Application

1.  Ensure your virtual environment is activated and the `GOOGLE_API_KEY` environment variable is set.
2.  Navigate to the project directory in your terminal.
3.  Run the main Python script (assuming it's named `app.py` - rename if necessary):
    ```bash
    python app.py
    ```
4.  The application will start, and Gradio will output a local URL (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`). Open this URL in your web browser.

---

## 📖 Usage Guide

1.  **Chat:** Interact with the MediGuide bot by typing questions or statements into the message box and pressing Enter or clicking "Send".
2.  **Log Data:** Mention vital signs (`My BP is 120/80`), symptoms (`I have a mild headache`), medications (`I take lisinopril 10mg daily`), or profile details (`I am 30 years old`) in your chat messages. The assistant will attempt to extract and log this information.
3.  **Dashboard:** Click the "📊 Health Dashboard" tab and then "🔄 View/Update Dashboard" to see a summary of your logged data and visualizations.
4.  **Report:** Click the "📋 Health Report" tab and then "📄 Generate Comprehensive Report" for a more detailed summary.
5.  **Resources:** Explore the "ℹ️ Resources & Emergency" tab for links to trusted health websites and emergency contact information.
6.  **Examples:** Use the example buttons below the chat input to quickly send predefined messages.

---

## ⚠️ Limitations

*   **Not Medical Advice:** As stated prominently, this is an informational tool only.
*   **No Diagnosis:** The AI cannot diagnose conditions. Symptom analysis is purely based on keyword matching against common patterns.
*   **Data Extraction Accuracy:** Relies on regular expressions, which may not capture all variations of user input or could extract data incorrectly.
*   **AI Hallucinations:** Like all LLMs, Gemini can sometimes generate incorrect or nonsensical information. Always verify critical information.
*   **Session-Based Storage:** Health data is stored in memory *only for the current session* and is lost when the application restarts. This demo does not include persistent database storage.
*   **Security:** This demo implementation lacks robust security features for handling sensitive health data. Do not deploy in a production environment handling real patient data without significant security enhancements and compliance considerations.
*   **Emergency Handling:** Keyword detection is basic and not foolproof. **Always call emergency services directly.**
*   **File Upload:** The file upload feature is simulated and does not perform actual document parsing or analysis.

---

## 🚀 Potential Future Enhancements

*   Implement robust document parsing (PDF, DOCX, Images via OCR) for the upload feature.
*   Integrate a database (e.g., SQLite, PostgreSQL) for persistent user data storage (requires handling authentication and privacy).
*   Add user authentication and profiles.
*   Improve Natural Language Understanding (NLU) for more accurate data extraction beyond simple regex.
*   Implement a functional notification system (reminders, tips).
*   Refine the health score algorithm (though it will always remain non-clinical).
*   Enhance UI/UX with more interactive elements.
*   Add more sophisticated data analysis and trend detection.
*   Internationalization and localization support.

---

## 🤝 Contributing

Contributions are welcome! If you find a bug or have an idea for improvement:

1.  **Check Issues:** See if the issue or feature request already exists.
2.  **Open an Issue:** If not, open a new issue detailing the bug or suggestion.
3.  **Fork & Pull Request:** Fork the repository, create a new branch for your changes, make your modifications, and submit a pull request with a clear description of your changes.

Please adhere to standard coding practices and ensure any health-related information remains general and is accompanied by appropriate disclaimers.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You should create a file named `LICENSE` in your repository containing the MIT License text).
