import cv2
import numpy as np
from keras.models import load_model
import cohere
from keras.optimizers import Adam

# Initialize Cohere client
co = cohere.Client("4g4XdeCQh82t7BKXyVxsqmOI31NyotsB8u5SO10C")

# Load the pre-trained emotion detection model
emotion_model = load_model(r"C:\Users\jenil\OneDrive\Desktop\emotion analyzer\emotion_model.hdf5", compile=False)
emotion_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Global variable for emotion
current_emotion = "Neutral"

def capture_and_detect_emotion(frame):
    """
    Detect faces in the frame, predict emotions, and overlay the results on the frame.
    """
    global current_emotion  # Ensure it's updated globally
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_emotion = "Neutral"

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (64, 64)) / 255.0
        face_reshaped = np.expand_dims(face_resized, axis=(0, -1))

        # Predict emotion
        emotion_prediction = emotion_model.predict(face_reshaped)
        predicted_emotion = np.argmax(emotion_prediction)
        detected_emotion = emotion_labels[predicted_emotion]

        # Update the global emotion variable
        current_emotion = detected_emotion  

        # Draw rectangle around face and display emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, detected_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame, detected_emotion  # Ensure the detected emotion is returned

def get_chatbot_response(user_message, emotion):
    """
    Generate a chatbot response based on the user's message and detected emotion.
    """
    print(f"Emotion received in chatbot: {emotion}")  # Debugging statement

    emotion_responses = {
        "Happy": "You look absolutely radiant with happiness, and it's truly heartwarming to see! If you’d like to share, I’d love to hear what’s been making you so joyful—whether it’s something exciting that happened, good news you received, or just a wonderful moment that brightened your day!" ,
        "Sad": "You’re not alone, and I’m here for you. If you want to talk or just need some company, I’m happy to listen.",
        "Angry": "I understand that you're feeling frustrated, and I truly appreciate you sharing your concerns with me. I'm here to assist you, and I want to make sure we find the best possible solution together. Please let me know more details about the issue, and I'll do my best to help resolve it as quickly as possible.",
        "Surprise": "Well, it looks like something caught you off guard! Was it something I said, or did something unexpected just happen? Either way, surprises can be exciting!",
        "Fear": "I can see that you're feeling afraid, and that's completely okay. Fear is a natural response, but you don’t have to go through it alone. If you’re comfortable, would you like to talk about what’s on your mind? Sometimes, sharing your thoughts can make things feel a little lighter. No matter what, I want you to know that you're not alone, and you have the strength to face this. I'm here for you.",
        "Disgust": "It seems like something may have made you uncomfortable or uneasy. Could you share what specifically bothered you? I’d like to understand your feelings better and see if there’s anything I can do to improve the situation.",
        "Neutral": "Hello! How can I assist you today?"
    }

    # If the user hasn't typed anything, respond based on detected emotion
    if not user_message.strip():
        return emotion_responses.get(emotion, "Hello! How can I assist you today?")

    # Modify the prompt to consider emotion context
    prompt = f"You are a helpful chatbot. The user is feeling {emotion}. Respond accordingly.\nUser: {user_message}\nChatbot:"
    
    try:
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=250,  # Increased token limit
            temperature=0.7,
            stop_sequences=["\n"],  # Prevents abrupt cut-off
            truncate='END' 
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Sorry, I couldn't process that. {e}"


