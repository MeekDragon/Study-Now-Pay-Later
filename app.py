import cv2
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_session import Session
from tensorflow.keras.models import load_model
import mediapipe as mp
from datetime import datetime
import matplotlib
from config.quiz_config import QUESTION_BANK
import html

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
import base64
import os
import json
import threading
from datetime import datetime as dt
import logging

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_PERMANENT'] = False
Session(app)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    """Format ISO datetime string"""
    if isinstance(value, str):
        try:
            value = dt.fromisoformat(value)
        except ValueError:
            return "Invalid Date"
    if isinstance(value, dt):
         return value.strftime(format)
    return "Invalid Date"

app.jinja_env.filters['datetimeformat'] = datetimeformat


try:
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    logging.info("MediaPipe Face Detection initialized.")
except Exception as e:
    logging.error(f"Failed to initialize MediaPipe Face Detection: {e}", exc_info=True)
    face_detection = None


model_path = 'static/models/emotion_model.h5'
if not os.path.exists(model_path):
    logging.error(f"CRITICAL: Emotion model not found at {model_path}. Analysis will fail.")
    model = None
else:
    try:
        model = load_model(model_path)
        logging.info(f"Emotion model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load emotion model from {model_path}: {e}", exc_info=True)
        model = None

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


analysis_lock = threading.Lock()

def calculate_stress(emotion_probs):
    """Calculate stress level from emotion probabilities"""
    if not isinstance(emotion_probs, dict):
        logging.warning(f"Invalid input for calculate_stress: {emotion_probs}")
        return 0

    stress_weights = {
        'Angry': 0.7, 'Fear': 0.6, 'Sad': 0.5,
        'Neutral': 0.1, 'Happy': -0.4, 'Disgust': 0.8,
        'Surprise': 0.0
    }
    stress_score = sum(
        emotion_probs.get(emo, 0) * weight
        for emo, weight in stress_weights.items()
    ) * 100
    return min(100, max(0, stress_score))

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/start_quiz', methods=['POST'])
def start_quiz():
    session.clear()
    form_topic = request.form.get('topic', 'general')
    user_name = request.form.get('name', 'Anonymous')
    user_email = request.form.get('email', '')


    if not form_topic:
        logging.warning("Start quiz attempted without selecting a topic.")
        form_topic = 'general'


    if form_topic.upper() == "DSA":
        session_topic_key = "Data Structures & Algorithm"

    elif form_topic.lower() == "oop":
        session_topic_key = "oop"

    else:
        session_topic_key = html.unescape(form_topic)


    if session_topic_key not in QUESTION_BANK:
         logging.error(f"Topic '{session_topic_key}' (from form: '{form_topic}') not found in QUESTION_BANK during start_quiz.")

         return redirect(url_for('index'))
    # ---------------------------------------------------------------------

    session.update({
        'user_info': {
            'name': user_name,
            'email': user_email,
            'topic': session_topic_key,
            'start_time': datetime.now().isoformat()
        },
        'emotion_data': [],
        'quiz_answers': {},
        'quiz_results': None
    })
    session.modified = True
    logging.info(f"Quiz started for topic key: '{session_topic_key}', User: {user_name}")
    # Redirect using the CANONICAL topic key stored in the session
    return redirect(url_for('quiz', topic=session_topic_key))



@app.route('/quiz/<topic>')
def quiz(topic):

    canonical_topic_key = html.unescape(topic)


    session_topic = session.get('user_info', {}).get('topic')
    if not session_topic or session_topic != canonical_topic_key:
        logging.warning(f"Access denied or session topic mismatch for quiz. URL key: '{canonical_topic_key}', Session key: '{session_topic}'. Redirecting to index.")
        return redirect(url_for('index'))
    # --------------------

    logging.info(f"Attempting to find quiz data for canonical key: '{canonical_topic_key}'")
    quiz_data = QUESTION_BANK.get(canonical_topic_key)
    # ----------------------------------------------------

    if not quiz_data:

         logging.error(f"Quiz data not found for key: '{canonical_topic_key}' (validated at start). Configuration issue? Redirecting.")
         return redirect(url_for('index'))

    # Prepare data for template
    questions = quiz_data.get("questions", {})

    quiz_title = quiz_data.get("title", canonical_topic_key)

    logging.info(f"Rendering quiz page for topic key: '{canonical_topic_key}', title: '{quiz_title}'")

    return render_template(
        'quiz.html',
        topic=canonical_topic_key,
        quiz_title=quiz_title,
        questions=questions
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    # for Checking if model and face detection loaded correctly
    if model is None or face_detection is None:
         logging.error("Analysis endpoint called, but model or face detector not loaded.")
         return jsonify({'error': 'Analysis components unavailable'}), 503

    try:
        if 'user_info' not in session:
            logging.warning("Analyze endpoint called without active session.")
            return jsonify({'error': 'No active quiz session'}), 403

        if 'frame' not in request.files:
            logging.error('No frame file in request')
            return jsonify({'error': 'Missing image data'}), 400

        frame_file = request.files['frame']
        if not frame_file or frame_file.filename == '' or not hasattr(frame_file, 'read'):
            logging.error('Empty or invalid frame file object')
            return jsonify({'error': 'Empty or invalid image data'}), 400

        frame_bytes = frame_file.read()
        if not frame_bytes or len(frame_bytes) < 500:
            logging.error(f'Invalid image size: {len(frame_bytes)} bytes')
            return jsonify({'error': 'Invalid image data (too small)'}), 400

        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            logging.error('Failed to decode image using cv2.imdecode')
            return jsonify({'error': 'Invalid image format or corrupted data'}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with the initialized detector instance
        results = face_detection.process(img_rgb)

        if not results.detections:
            logging.warning('No faces detected in submitted frame')
            return jsonify({'error': 'Face not detected'}), 404

        detection = results.detections[0].location_data.relative_bounding_box
        h, w = img.shape[:2]

        xmin, ymin, width_rel, height_rel = detection.xmin, detection.ymin, detection.width, detection.height
        padding = 0.1
        x = int(max(0, (xmin - padding * width_rel) * w))
        y = int(max(0, (ymin - padding * height_rel) * h))
        width = int(min(w - x, (width_rel + 2 * padding * width_rel) * w))
        height = int(min(h - y, (height_rel + 2 * padding * height_rel) * h))

        if width <= 0 or height <= 0 or x >= w or y >= h:
             logging.error(f"Invalid face ROI coordinates calculated: x={x}, y={y}, width={width}, height={height}")
             return jsonify({'error': 'Face coordinate calculation failed'}), 400

        face_roi = img[y:y + height, x:x + width]
        if face_roi.size == 0:
            logging.error('Empty face region after slicing')
            return jsonify({'error': 'Face extraction failed (empty ROI)'}), 400

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        normalized = resized.astype('float32') / 255.0
        input_tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

        predictions = model.predict(input_tensor)[0]
        emotion_probs = {label: float(prob) for label, prob in zip(emotion_labels, predictions)}
        dominant_emotion_index = np.argmax(predictions)
        dominant_emotion = emotion_labels[dominant_emotion_index]
        confidence = float(predictions[dominant_emotion_index])
        current_stress = calculate_stress(emotion_probs)

        with analysis_lock:
             if 'emotion_data' not in session: session['emotion_data'] = []
             session['emotion_data'].append({
                 'timestamp': datetime.now().isoformat(), 'emotions': emotion_probs,
                 'dominant_emotion': dominant_emotion, 'confidence': confidence, 'stress': current_stress
             })
             session.modified = True

        return jsonify({
            'dominant_emotion': dominant_emotion, 'confidence': confidence,
            'emotion_probs': emotion_probs, 'stress': current_stress
        })

    except cv2.error as e:
        logging.error(f"OpenCV error during analysis: {str(e)}", exc_info=True)
        return jsonify({'error': 'Image processing failed (OpenCV)'}), 500
    except Exception as e:
        logging.error(f"Unexpected analysis error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during analysis'}), 500

@app.route('/submit-quiz', methods=['POST'])
def submit_quiz():
    try:
        logging.info("Received /submit-quiz request")
        if 'user_info' not in session:
            logging.warning("Submit quiz endpoint called without active session.")
            return jsonify({'error': 'No active quiz session'}), 403

        if not request.is_json:
            logging.error("Invalid content type: Request is not JSON")
            return jsonify({'error': 'Invalid content type, JSON required'}), 400

        data = request.get_json()
        logging.info(f"Received JSON data: {data}")

        topic_from_request = data.get('topic')
        user_answers = data.get('answers', {})

        if not topic_from_request:
             logging.error("Missing 'topic' key in submitted JSON data")
             return jsonify({'error': 'Missing quiz topic in request'}), 400
        if not isinstance(user_answers, dict):
             logging.error("Invalid 'answers' format in submitted JSON data (must be a dictionary)")
             return jsonify({'error': 'Invalid answers format'}), 400


        canonical_topic_key = html.unescape(topic_from_request)


        logging.info(f"Processing submission for topic key: '{canonical_topic_key}', User answers: {user_answers}")


        if canonical_topic_key not in QUESTION_BANK:
            logging.error(f"Invalid topic key received in submission: '{canonical_topic_key}' (Original from request: '{topic_from_request}') - Key not found in QUESTION_BANK.")
            return jsonify({'error': 'Invalid quiz topic specified'}), 400

        quiz_details = QUESTION_BANK[canonical_topic_key]
        questions_data = quiz_details.get('questions', {})
        if not questions_data:
             logging.error(f"No questions found for topic '{canonical_topic_key}' in QUESTION_BANK configuration.")
             return jsonify({'error': 'Internal configuration error: No questions for topic'}), 500

        correct_answers = {q_id: details['correct'] for q_id, details in questions_data.items() if 'correct' in details}
        total_questions = len(correct_answers)

        score = 0
        for q_id, submitted_answer in user_answers.items():

            if q_id in correct_answers and correct_answers[q_id] == submitted_answer:
                score += 1
            elif q_id not in correct_answers:
                 logging.warning(f"Received answer for unknown question ID '{q_id}' for topic '{canonical_topic_key}'.")


        logging.info(f"Quiz for topic '{canonical_topic_key}' completed. Score: {score}/{total_questions}")

        session['quiz_results'] = {
            'score': score, 'total': total_questions,
            'accuracy': round((score / total_questions) * 100, 2) if total_questions > 0 else 0,
            'topic': canonical_topic_key # Store the canonical topic key
        }
        session['quiz_answers'] = user_answers
        session.modified = True

        logging.info("Quiz submission processed successfully")
        return jsonify({'success': True, 'message': 'Quiz submitted successfully'})

    except Exception as e:
        logging.error(f"Unexpected error during quiz submission: {str(e)}", exc_info=True)
        return jsonify({'error': 'Quiz processing failed due to an internal error'}), 500


def calculate_metrics(emotion_data, user_info, quiz_results):
    if not emotion_data:
        logging.warning("Calculating metrics with no emotion data.")
        return {
            'avg_stress': 0, 'peak_stress': 0, 'test_accuracy': quiz_results.get('accuracy', 0),
            'emotion_counts': {emo: 0 for emo in emotion_labels},
            'quiz_score': quiz_results.get('score', 0), 'total_questions': quiz_results.get('total', 0),
            'positivity_index': 0, 'overall_score': 0, 'duration_seconds': 0
        }

    stress_levels = [float(entry.get('stress', 0)) for entry in emotion_data]
    dominant_emotions = [entry.get('dominant_emotion') for entry in emotion_data if entry.get('dominant_emotion')]

    duration_seconds = 0
    try:
        start_time_str = user_info.get('start_time')
        end_time_str = emotion_data[-1].get('timestamp') if emotion_data else None
        if start_time_str and end_time_str:
             start_time = dt.fromisoformat(start_time_str)
             end_time = dt.fromisoformat(end_time_str)
             duration_seconds = max(0, (end_time - start_time).total_seconds())
        else:
             logging.warning("Could not calculate duration due to missing timestamps.")
    except ValueError:
        logging.warning("Could not parse timestamps for duration calculation.")

    metrics = {
        'avg_stress': np.mean(stress_levels) if stress_levels else 0,
        'peak_stress': max(stress_levels) if stress_levels else 0,
        'test_accuracy': quiz_results.get('accuracy', 0),
        'emotion_counts': {emo: dominant_emotions.count(emo) for emo in emotion_labels},
        'quiz_score': quiz_results.get('score', 0),
        'total_questions': quiz_results.get('total', 0),
        'duration_seconds': round(duration_seconds)
    }

    total_valid_emotions = sum(metrics['emotion_counts'].values())
    if total_valid_emotions == 0:
        metrics['positivity_index'] = 0
    else:
        metrics['positivity_index'] = (metrics['emotion_counts'].get('Happy', 0) / total_valid_emotions) * 100


    weights = {'accuracy': 0.5, 'stress': 0.3, 'positivity': 0.2}
    metrics['overall_score'] = (
        weights['accuracy'] * metrics['test_accuracy'] +
        weights['stress'] * max(0, 100 - metrics['avg_stress']) +
        weights['positivity'] * metrics['positivity_index']
    )

    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in metrics.items()}


@app.route('/results')
def results():

    if 'user_info' not in session:
        logging.warning("Results page accessed without user info in session.")
        return redirect(url_for('index'))

    user_info = session['user_info']
    emotion_data = session.get('emotion_data', [])
    quiz_results = session.get('quiz_results')

    if not quiz_results:
         logging.warning("Results page accessed without quiz results in session.")

         quiz_results = {}

    try:
        metrics = calculate_metrics(emotion_data, user_info, quiz_results)

        plot_data_url = None
        if emotion_data:

             plt.switch_backend('Agg')
             fig, ax = plt.subplots(figsize=(10, 4))
             timestamps_str = [e.get('timestamp') for e in emotion_data if e.get('timestamp')]
             stress_levels = [e.get('stress', 0) for e in emotion_data]
             timestamps_dt = []
             valid_stress_levels = []
             for ts_str, stress in zip(timestamps_str, stress_levels):
                  try:
                       timestamps_dt.append(dt.fromisoformat(ts_str))
                       valid_stress_levels.append(stress)
                  except (ValueError, TypeError): continue
             if timestamps_dt:
                  ax.plot(timestamps_dt, valid_stress_levels, marker='o', linestyle='-', color='#ff6384', label='Stress Level')
                  ax.set_title('Stress Level During Quiz'); ax.set_ylabel('Stress Level (%)'); ax.set_xlabel('Time')
                  ax.set_ylim(0, 100); ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
                  from matplotlib.dates import AutoDateFormatter, AutoDateLocator
                  locator = AutoDateLocator(); formatter = AutoDateFormatter(locator)
                  ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
                  plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                  buffer = BytesIO(); plt.savefig(buffer, format='png', bbox_inches='tight'); buffer.seek(0)
                  plot_data = base64.b64encode(buffer.read()).decode('utf-8')
                  plot_data_url = f"data:image/png;base64,{plot_data}"; plt.close(fig)
             else: logging.warning("No valid timestamp data for plotting stress.")
        else: logging.info("No emotion data, skipping stress plot.")

        return render_template(
            'results.html', plot_url=plot_data_url, user_info=user_info,
            quiz_results=quiz_results, metrics=metrics
       )

    except Exception as e:
        logging.error(f"Unexpected error generating results page: {str(e)}", exc_info=True)
        return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)