from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import json
import asyncio
from transformers import pipeline
from PIL import Image
import io
import tempfile
import os
from collections import deque
import logging
import traceback
import wave
import soundfile as sf
import librosa

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logging.getLogger("transformers").setLevel(logging.ERROR)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and configs
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
SMOOTH_WINDOW = 5
SAMPLE_RATE = 16000

# Voice emotion mapping - Updated for the wav2vec2 model
VOICE_TO_FACE_MAP = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'pleasant surprised': 'surprise',
    'sad': 'sad'
}

# Initialize models with error handling
face_emotion_pipe = None
voice_emotion_pipe = None
face_cascade = None

print("Loading models...")
try:
    face_emotion_pipe = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection",
        device=-1
    )
    print("Face emotion model loaded")
except Exception as e:
    print(f"Error loading face emotion model: {e}")

try:
    # Using the wav2vec2 model from the Tkinter app
    voice_emotion_pipe = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=-1
    )
    print("Voice emotion model loaded")
except Exception as e:
    print(f"Error loading voice emotion model: {e}")

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Face cascade loaded")
except Exception as e:
    print(f"Error loading face cascade: {e}")

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.emotion_buffers = {}
        self.emotion_counters = {}
        self.session_active = {}
        self.audio_buffers = {}  # Store audio chunks

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        connection_id = id(websocket)
        self.emotion_buffers[connection_id] = deque(maxlen=SMOOTH_WINDOW)
        self.emotion_counters[connection_id] = {
            'face': {e: 0 for e in EMOTION_LABELS},
            'voice': {e: 0 for e in EMOTION_LABELS}
        }
        self.session_active[connection_id] = False
        self.audio_buffers[connection_id] = []
        print(f"Client connected: {connection_id}")

    def disconnect(self, websocket: WebSocket):
        connection_id = id(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Clean up connection data
        for dict_obj in [self.emotion_buffers, self.emotion_counters, self.session_active, self.audio_buffers]:
            if connection_id in dict_obj:
                del dict_obj[connection_id]
        print(f"Client disconnected: {connection_id}")

    def start_session(self, connection_id):
        self.session_active[connection_id] = True
        # Reset counters and audio buffer
        self.emotion_counters[connection_id] = {
            'face': {e: 0 for e in EMOTION_LABELS},
            'voice': {e: 0 for e in EMOTION_LABELS}
        }
        self.audio_buffers[connection_id] = []

    def stop_session(self, connection_id):
        self.session_active[connection_id] = False

manager = ConnectionManager()

def analyze_face_emotion(image_data):
    """Analyze emotion from base64 encoded image"""
    if not face_emotion_pipe or not face_cascade:
        return {'success': False, 'error': 'Models not loaded'}

    try:
        # Handle base64 data URL
        if ',' in image_data:
            img_data = base64.b64decode(image_data.split(',')[1])
        else:
            img_data = base64.b64decode(image_data)

        img = Image.open(io.BytesIO(img_data))

        # Convert to OpenCV format for face detection
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = img.crop((x, y, x + w, y + h))

            # Get emotion predictions
            result = face_emotion_pipe(face_roi)
            emotion_scores = {}

            # Normalize emotion labels and scores
            for item in result:
                label = item['label'].lower()
                if label in EMOTION_LABELS:
                    emotion_scores[label] = item['score']

            # Ensure all emotions have a score
            for emotion in EMOTION_LABELS:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0

            return {
                'success': True,
                'emotions': emotion_scores,
                'face_detected': True,
                'face_coords': [int(x), int(y), int(w), int(h)]
            }
        else:
            return {'success': True, 'emotions': {}, 'face_detected': False}

    except Exception as e:
        print(f"Face analysis error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def save_audio_chunk_as_wav(audio_data, filename):
    """Save raw audio data as WAV file"""
    try:
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}")
        return False


import soundfile as sf
import librosa

def analyze_voice_emotion_from_chunks(connection_id):
    """Analyze accumulated audio chunks using voice emotion pipeline without FFmpeg"""
    if not voice_emotion_pipe:
        return {'success': False, 'error': 'Voice model not loaded'}
    try:
        audio_chunks = manager.audio_buffers[connection_id]
        if not audio_chunks:
            return {'success': True, 'emotions': {}, 'no_audio': True}

        # Concatenate all audio chunks into one NumPy array
        full_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32) / 32768.0  # Normalize int16 to float
        # Skip if volume is too low
        if np.abs(full_audio).mean() < 0.01:
            print("Audio too quiet, skipping voice analysis")
            return {'success': True, 'emotions': {'neutral': 1.0}, 'silent': True}

        # Resample if needed
        target_sample_rate = voice_emotion_pipe.feature_extractor.sampling_rate
        if SAMPLE_RATE != target_sample_rate:
            full_audio = librosa.resample(full_audio, orig_sr=SAMPLE_RATE, target_sr=target_sample_rate)

        # Run inference directly with raw audio data
        result = voice_emotion_pipe(full_audio, sampling_rate=target_sample_rate)

        # Map results to standard emotions
        mapped_emotions = {e: 0.0 for e in EMOTION_LABELS}
        if result:
            for item in result:
                label = item['label'].lower()
                face_label = VOICE_TO_FACE_MAP.get(label, 'neutral')
                if face_label in mapped_emotions:
                    mapped_emotions[face_label] += item['score']
        else:
            mapped_emotions['neutral'] = 1.0

        return {'success': True, 'emotions': mapped_emotions}

    except Exception as e:
        print(f"Voice analysis error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def smooth_emotions(connection_id, new_scores):
    """Smooth emotion scores over time"""
    if connection_id not in manager.emotion_buffers:
        return new_scores

    buffer = manager.emotion_buffers[connection_id]
    buffer.append(new_scores)

    # Calculate moving average
    smoothed = {}
    for emotion in EMOTION_LABELS:
        scores = [frame.get(emotion, 0) for frame in buffer if frame.get(emotion) is not None]
        smoothed[emotion] = sum(scores) / len(scores) if scores else 0

    return smoothed

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    connection_id = id(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'start_session':
                manager.start_session(connection_id)
                await websocket.send_text(json.dumps({
                    'type': 'session_started',
                    'message': 'Recording started'
                }))
                print(f"Session started for {connection_id}")

            elif message['type'] == 'stop_session':
                manager.stop_session(connection_id)

                # Analyze final audio chunks before stopping
                if manager.audio_buffers[connection_id]:
                    voice_result = analyze_voice_emotion_from_chunks(connection_id)
                    if voice_result['success'] and voice_result.get('emotions'):
                        emotions = voice_result['emotions']
                        if any(score > 0 for score in emotions.values()):
                            dominant = max(emotions, key=emotions.get)
                            manager.emotion_counters[connection_id]['voice'][
                                dominant] += 5  # Give final analysis more weight

                # Send final summary
                counters = manager.emotion_counters[connection_id]
                face_total = sum(counters['face'].values())
                voice_total = sum(counters['voice'].values())

                face_percentages = {
                    e: (count / face_total * 100) if face_total > 0 else 0
                    for e, count in counters['face'].items()
                }
                voice_percentages = {
                    e: (count / voice_total * 100) if voice_total > 0 else 0
                    for e, count in counters['voice'].items()
                }

                await websocket.send_text(json.dumps({
                    'type': 'final_summary',
                    'face_emotions': face_percentages,
                    'voice_emotions': voice_percentages
                }))
                print(f"Session stopped for {connection_id}")

            elif message['type'] == 'video_frame' and manager.session_active.get(connection_id, False):
                # Process video frame
                result = analyze_face_emotion(message['data'])

                if result['success'] and result.get('face_detected'):
                    # Smooth emotions
                    smoothed = smooth_emotions(connection_id, result['emotions'])

                    # Find dominant emotion
                    if smoothed:
                        dominant = max(smoothed, key=smoothed.get)
                        manager.emotion_counters[connection_id]['face'][dominant] += 1

                        await websocket.send_text(json.dumps({
                            'type': 'face_emotion',
                            'emotions': smoothed,
                            'dominant': dominant,
                            'face_coords': result.get('face_coords', [])
                        }))

            elif message['type'] == 'audio_chunk' and manager.session_active.get(connection_id, False):
                try:
                    # Decode audio data (expecting 16-bit PCM)
                    audio_bytes = base64.b64decode(message['data'])

                    # Convert to numpy array (16-bit integers)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                    # Store chunk for later analysis
                    manager.audio_buffers[connection_id].append(audio_array)

                    print(f"Audio chunk received: {len(audio_array)} samples")

                    # Analyze every few chunks (similar to Tkinter approach)
                    if len(manager.audio_buffers[connection_id]) >= 1:  # Analyze every 3 chunks
                        voice_result = analyze_voice_emotion_from_chunks(connection_id)

                        if voice_result['success'] and voice_result.get('emotions'):
                            emotions = voice_result['emotions']
                            if any(score > 0 for score in emotions.values()):
                                dominant = max(emotions, key=emotions.get)
                                manager.emotion_counters[connection_id]['voice'][dominant] += 1

                                await websocket.send_text(json.dumps({
                                    'type': 'voice_emotion',
                                    'emotions': emotions,
                                    'dominant': dominant
                                }))

                        # Clear processed chunks but keep the last one for continuity
                        if len(manager.audio_buffers[connection_id]) > 1:
                            manager.audio_buffers[connection_id] = manager.audio_buffers[connection_id][-1:]

                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
                    traceback.print_exc()

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {connection_id}")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        manager.disconnect(websocket)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def get_frontend():
    html_path = os.path.join(PROJECT_DIR, "static", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=404
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "face_model_loaded": face_emotion_pipe is not None,
        "voice_model_loaded": voice_emotion_pipe is not None,
        "face_cascade_loaded": face_cascade is not None
    }

if __name__ == "__main__":
    import uvicorn

    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
