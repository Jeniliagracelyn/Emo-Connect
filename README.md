# Emotion-Aware Chatbot 🤖💬

This project is a web-based chatbot application integrated with an emotion detection model. The chatbot uses machine learning to identify the emotional tone of user inputs and respond accordingly.

---

## 📁 Project Structure

├── pycache/ # Python cache files ├── templates/ # HTML templates for the frontend │ └── index.html # Main chatbot UI (assumed) ├── README.md # Project documentation ├── app.py # Flask app to run the server ├── chatbot_logic.py # Main chatbot response logic ├── emotion_model.hdf5 # Pre-trained emotion detection model ├── requirements.txt # Python dependencies


---

## 🚀 How to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/emotion-chatbot.git
cd emotion-chatbot

2.Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3.Install dependencies
pip install -r requirements.txt

4.Run the Flask app

python app.py

5.Open your browser

Navigate to http://127.0.0.1:5000 to interact with the chatbot.

Features
Emotion detection using a deep learning model (emotion_model.hdf5)

Dynamic chatbot responses based on detected emotion

User-friendly web interface built with HTML (via templates/ folder)

Future Improvements
Add voice input/output

Extend emotion model to handle more complex emotions

Store chat history and analytics

Deploy to cloud (Heroku, Render, etc.)

Author
Created by Jenilia Gracelyn.s
Feel free to reach out for collaborations!