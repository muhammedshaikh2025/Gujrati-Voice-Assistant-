
# gujarati_assistant.py
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pandas as pd
from rapidfuzz import fuzz, process
import subprocess
import sys
import os

# ---------- CONFIG ----------
VOSK_MODEL_PATH = "/home/pi/Downloads/gujrati/guj/vosk-model-small-gu-0.42"   # path to the Gujarati Vosk model
SAMPLE_RATE = 16000
CSV_QA = "/home/pi/Downloads/gujrati/guj/qa_gujarati.csv"           # CSV with columns: question,answer (Gujarati text)
MIN_CONFIDENCE = 0.6                 # confidence threshold for fuzzy match (0-1 scale)
TOP_K = 3                            # top k fuzzy candidates to consider
# ----------------------------

if not os.path.exists(VOSK_MODEL_PATH):
    print("ERROR: Vosk model not found at", VOSK_MODEL_PATH)
    sys.exit(1)

# load model
model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(False)

# load QA CSV
df = pd.read_csv(CSV_QA)
# ensure columns 'question' and 'answer' present
if 'question' not in df.columns or 'answer' not in df.columns:
    raise ValueError("CSV must have 'question' and 'answer' columns")

questions = df['question'].astype(str).tolist()
answers = df['answer'].astype(str).tolist()

q_indexed = list(enumerate(questions))  # (idx, question)

# audio callback
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print("Audio status:", status, file=sys.stderr)
    q.put(bytes(indata))

def text_to_speech_espeak(text):
    # eSpeak NG: specify voice lang 'gu' if available; else default.
    # On many systems, 'gu' will work. You can adjust speed with -s and pitch with -p
   
    try:
        subprocess.run(["espeak-ng", text])

    except FileNotFoundError:
        print("espeak-ng not found. Install or change TTS function.")

def find_best_answer(transcript):
    # fast fuzzy match using RapidFuzz process.extract
    # returns best matching answer and score (0-100)
    results = process.extract(transcript, questions, scorer=fuzz.token_sort_ratio, limit=TOP_K)
    # results = [(match, score, idx), ...]
    best_match, score, idx = results[0]
    return answers[idx], score/100.0, best_match

print("Starting Gujarati offline assistant. Speak now... (press Ctrl+C to stop)")

with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize = 8000, dtype='int16',
                       channels=1, callback=callback):
    try:
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                res = rec.Result()
                j = json.loads(res)
                text = j.get("text", "").strip()
                if text:
                    print("You said:", text)
                    ans, score, matched_q = find_best_answer(text)
                    print(f"Matched (score={score:.2f}): '{matched_q}' -> Answer: {ans}")
                    # If score low, optionally say fallback
                    if score < MIN_CONFIDENCE:
                        fallback = "માફ કરજો, મને સારું સમજાતું નથી. ફરી કહીશ?"
                        print("Low confidence, responding fallback.")
                        text_to_speech_espeak(fallback)
                    else:
                        text_to_speech_espeak(ans)
            else:
                # partial result (optional)
                # print(rec.PartialResult())
                pass
    except KeyboardInterrupt:
        print("\nStopped by user")
