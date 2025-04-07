from flask import Flask, render_template, Response, request, jsonify
from chatbot_logic import capture_and_detect_emotion, get_chatbot_response
import cv2

app = Flask(__name__)

# Global variable to store the current emotion
current_emotion = "Neutral"

@app.route("/")
def index():
    return render_template("index.html")

def generate_frames():
    global current_emotion
    cap = cv2.VideoCapture(0)  # Use the default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect emotion and overlay on frame
            frame, emotion = capture_and_detect_emotion(frame)
            current_emotion = emotion  # Update detected emotion

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Stream the video frame
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/chat", methods=["POST"])
def chat():
    global current_emotion
    user_message = request.json.get("message", "")

    # Debugging: Print the detected emotion
    print(f"Detected Emotion: {current_emotion}")  

    chatbot_response = get_chatbot_response(user_message, current_emotion)
    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(debug=True)




