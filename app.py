import os
import json
import random
from datetime import datetime
from functools import wraps
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import cv2
import numpy as np
from spellchecker import SpellChecker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
import io
import re
from werkzeug.security import generate_password_hash

# ─── Flask Setup ─────────────────
app = Flask(__name__)
app.secret_key = 'smart-dyslexia-detector-secret-key-2025'
CORS(app)

# ─── Database Config ─────────────
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'dyslexia_db'
}

# ─── Upload Config ───────────────
UPLOAD_FOLDER = os.path.join('static', 'uploads')
SOUND_FOLDER = os.path.join('static', 'sounds')
WORKSHEET_FOLDER = os.path.join('static', 'worksheets')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ─── OCR Setup ─────────────────
OCR_AVAILABLE = False
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
    print("✅ Tesseract OCR loaded successfully!")
except Exception as e:
    print(f"⚠️ Tesseract OCR not available: {e}")

# ─── YOLO Model ─────────────────
MODEL_PATH = os.path.join('model', 'best.pt')
model = None

try:
    from ultralytics import YOLO
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("✅ YOLOv11 model loaded for dyslexia detection!")
    else:
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")

# ─── Spell Checker ───────────────
spell = SpellChecker()

# ═══════════════ CONSTANTS ═══════════════

# YOLO class indices from training
CLASS_NORMAL    = 0   # letter written correctly
CLASS_REVERSAL  = 1   # letter reversed/mirrored (dyslexia pattern)
CLASS_CORRECTED = 2   # letter written then corrected (also a dyslexia signal)

# Reversal pairs — used ONLY for labelling the reversal in the result,
# NOT for detecting dyslexia (YOLO does that visually)
REVERSAL_PAIRS = {
    'b': ['d', 'p'],
    'd': ['b', 'q'],
    'p': ['q', 'b'],
    'q': ['p', 'd'],
    'm': ['w'],
    'w': ['m'],
    'n': ['u'],
    'u': ['n'],
}

# OCR confusion pairs — characters OCR commonly misreads
# Used to filter false positives when OCR identity check is needed
OCR_CONFUSIONS = {
    'g': 'e', 'e': 'g', 'a': 'o', 'o': 'a',
    'c': 'e', 'l': 'i', 'i': 'l', 'h': 'n',
    'r': 'v', 'v': 'r', 't': 'f', 'f': 't',
    's': 'z', 'z': 's', 'u': 'v',
    'k': 'x', 'x': 'k', 'j': 'i', 'y': 'v'
}

THEMED_WORDS = {
    'animals': {
        'easy': ['duck', 'pig', 'bird', 'bear', 'deer', 'bull', 'pony', 'frog', 'eel', 'bee'],
        'medium': ['badger', 'dolphin', 'penguin', 'rabbit', 'butterfly', 'ladybug', 'woodpecker', 'hummingbird'],
        'hard': ['dinosaur', 'pterodactyl', 'panda', 'peacock', 'butterflyfish', 'dragonfly']
    },
    'food': {
        'easy': ['bun', 'jam', 'pie', 'egg', 'milk', 'peas', 'beans', 'bread', 'butter', 'pudding'],
        'medium': ['burger', 'pancake', 'waffle', 'donut', 'pumpkin', 'broccoli', 'cucumber', 'radish'],
        'hard': ['pineapple', 'blueberry', 'raspberry', 'blackberry', 'pomegranate', 'cauliflower']
    },
    'colors': {
        'easy': ['red', 'blue', 'pink', 'brown', 'gold', 'plum', 'lime', 'mint', 'beige'],
        'medium': ['purple', 'bronze', 'silver', 'copper', 'maroon', 'crimson', 'indigo', 'violet'],
        'hard': ['turquoise', 'magenta', 'lavender', 'chartreuse', 'cerulean', 'vermillion']
    },
    'actions': {
        'easy': ['run', 'jump', 'skip', 'dip', 'bob', 'dab', 'mop', 'dig', 'pop', 'tap'],
        'medium': ['bounce', 'dangle', 'wiggle', 'paddle', 'splash', 'climb', 'dive', 'bury'],
        'hard': ['balance', 'dribble', 'compute', 'disappear', 'appreciate', 'demonstrate']
    },
    'nature': {
        'easy': ['sun', 'mud', 'dew', 'bud', 'pod', 'bush', 'pond', 'wood', 'peak', 'bark'],
        'medium': ['branch', 'pebble', 'boulder', 'mountain', 'desert', 'river', 'waterfall', 'canyon'],
        'hard': ['volcano', 'earthquake', 'hurricane', 'tornado', 'tsunami', 'avalanche']
    }
}

DEFAULT_WORDS = {
    'easy': ['bed', 'big', 'bad', 'dad', 'dab', 'bun', 'bud', 'bug', 'dug', 'pub', 
             'mud', 'sun', 'bus', 'cup', 'pup', 'sub'],
    'medium': ['bird', 'brown', 'blue', 'black', 'bread', 'dirty', 'puppy', 'bunny',
               'mummy', 'daddy', 'pond', 'under', 'bubble', 'purple', 'dragon'],
    'hard': ['bridge', 'butterfly', 'dolphin', 'penguin', 'mountain', 'umbrella',
             'bedroom', 'cupboard', 'upstairs', 'downstairs', 'dinosaur', 'backpack']
}

MALAY_WORDS = {
    'easy': ['buku', 'bapa', 'dua', 'dapur', 'padi', 'mata', 'budi', 'pura', 'diri', 'muka'],
    'medium': ['badan', 'bulat', 'pukul', 'dunia', 'bumi', 'buruk', 'putih', 'dalam', 'malam', 'pagi'],
    'hard': ['bangunan', 'pemerhati', 'dermawan', 'budiman', 'pertama', 'berdiri', 'membaca', 'mendengar']
}

LETTER_FORMATION = {
    'b': "1. Start at top line\n2. Draw straight down\n3. Go back up to middle\n4. Draw circle to the right",
    'd': "1. Start at middle line\n2. Draw circle going left\n3. Continue straight down\n4. Go back up",
    'p': "1. Start at middle line\n2. Draw straight down below line\n3. Go back up\n4. Draw circle to right",
    'q': "1. Start at middle line\n2. Draw circle going left\n3. Continue down below line\n4. Add curve at bottom",
    'm': "1. Start at middle line\n2. Draw straight down\n3. Go up, make hump\n4. Go down, make another hump",
    'w': "1. Start at middle line\n2. Go down diagonally\n3. Go up diagonally\n4. Repeat for second peak",
    'n': "1. Start at middle line\n2. Draw straight down\n3. Go back up\n4. Make hump to the right",
    'u': "1. Start at middle line\n2. Draw down curving right\n3. Go back up curving right\n4. Short line down"
}

SHORT_PHRASES = ['big red dog', 'blue bird sings', 'brown bear sleeps',
                 'queen waves hand', 'baby duck swims', 'dark night sky']


# ═══════════════ HELPERS ═══════════════

def get_db():
    return mysql.connector.connect(**db_config)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def cleanup_temp_files():
    """Remove any leftover temp files from YOLO processing"""
    upload_dir = app.config['UPLOAD_FOLDER']
    count = 0
    for filename in os.listdir(upload_dir):
        if filename.startswith('temp_') and filename.endswith('.png'):
            try:
                os.remove(os.path.join(upload_dir, filename))
                count += 1
            except:
                pass
    if count > 0:
        print(f"🗑️ Cleaned up {count} temp files")

def save_uploaded_image(file):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')    
    filename = secure_filename(f"{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath, filename


def convert_letter_for_model(letter_img):
    """Preprocess a cropped letter image into the 64x64 format the YOLO model expects."""
    if letter_img is None:
        return None
    if hasattr(letter_img, 'size') and letter_img.size == 0:
        return None

    if len(letter_img.shape) == 3:
        letter_gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
    else:
        letter_gray = letter_img.copy()

    # Add padding
    h, w = letter_gray.shape[:2]
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    letter_padded = cv2.copyMakeBorder(
        letter_gray, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT, value=255
    )

    # Ensure dark ink on white background
    letter_mean = float(np.mean(letter_padded))
    if letter_mean > 127:
        letter_inverted = cv2.bitwise_not(letter_padded)
    else:
        letter_inverted = letter_padded

    # Threshold
    white_pixels = np.sum(letter_inverted > 128)
    total_pixels = letter_inverted.size
    if white_pixels < total_pixels * 0.02:
        _, letter_bw = cv2.threshold(letter_inverted, 100, 255, cv2.THRESH_BINARY)
    else:
        _, letter_bw = cv2.threshold(letter_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels_after = np.sum(letter_bw > 128)
    if white_pixels_after < total_pixels * 0.01:
        letter_bw = cv2.adaptiveThreshold(
            letter_inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    letter_final = cv2.cvtColor(letter_bw, cv2.COLOR_GRAY2BGR)
    letter_resized = cv2.resize(letter_final, (64, 64), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    rh, rw = letter_resized.shape[:2]
    start_y = max(0, (64 - rh) // 2)
    start_x = max(0, (64 - rw) // 2)
    end_y = min(64, start_y + rh)
    end_x = min(64, start_x + rw)
    canvas[start_y:end_y, start_x:end_x] = letter_resized[:end_y - start_y, :end_x - start_x]
    return canvas


# ═══════════════ OCR — IDENTITY ONLY ═══════════════

def ocr_read_raw(image):
    """
    OCR reads the image to identify WHAT WORD was written.
    This is used purely for word identity (did the child write the right word?).
    Dyslexia detection is NOT done here — that is YOLO's job.
    Returns: (text, confidence)
    """
    if not OCR_AVAILABLE:
        return None, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ocr_results = []
    configs = [
        '--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        '--psm 8 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
    ]
    for config in configs:
        for img_version in [gray, enhanced]:
            text = pytesseract.image_to_string(img_version, config=config).strip().lower()
            cleaned = ''.join(c for c in text if c.isalpha())
            if cleaned:
                try:
                    conf_data = pytesseract.image_to_data(
                        img_version,
                        output_type=pytesseract.Output.DICT,
                        config=config
                    )
                    conf_values = [int(c) for c in conf_data['conf'] if c != '-1']
                    avg_conf = np.mean(conf_values) if conf_values else 0
                except Exception:
                    avg_conf = 30
                ocr_results.append({'text': cleaned, 'confidence': avg_conf})

    if not ocr_results:
        return None, 0

    best = max(ocr_results, key=lambda x: x['confidence'])
    print(f"   OCR (identity): '{best['text']}' (confidence: {best['confidence']:.0f}%)")
    return best['text'], best['confidence']


# ═══════════════ LETTER SEGMENTATION ═══════════════

def segment_letters(word_img):
    """
    Extract individual letter bounding boxes from a word image.
    Tries multiple thresholding methods and picks the one that finds
    the most plausible letter count.
    Returns list of dicts with 'bbox': (x, y, w, h), sorted left to right.
    """
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    candidates = []

    # Method 1: Otsu
    _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(gray)) > 127:
        binary1 = cv2.bitwise_not(binary1)
    c1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.append(c1)

    # Method 2: Adaptive Gaussian
    binary2 = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    c2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.append(c2)

    # Method 3: Simple fixed threshold
    _, binary3 = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    c3, _ = cv2.findContours(binary3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.append(c3)

    best_letters, best_count = [], 0
    for contours in candidates:
        letters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 4 and h > 6 and w < 150 and h < 150:
                aspect = w / h if h > 0 else 0
                if 0.1 < aspect < 3.0:
                    letters.append({'bbox': (x, y, w, h)})
        letters.sort(key=lambda l: l['bbox'][0])
        if len(letters) > best_count:
            best_count = len(letters)
            best_letters = letters

    return best_letters


# ═══════════════ YOLO LETTER CLASSIFICATION (PRIMARY DETECTOR) ═══════════════

def classify_letters_with_yolo(word_img, expected_word, threshold=0.7):
    """
    PRIMARY dyslexia detection function.

    For each letter position in the word image:
      - Crop the letter
      - Feed it to the YOLO model
      - Get class: 0=normal, 1=reversal, 2=corrected

    Returns:
      letters_found  — list of all letter results
      reversals      — list of positions where reversal or corrected was detected
      letter_details — full per-letter breakdown for reporting
    """
    if model is None:
        print("❌ YOLO model not loaded — cannot classify letters")
        return [], [], []

    letters_found = segment_letters_guaranteed(word_img, expected_word)
    expected_chars = list(expected_word.lower())

    print(f"      Segments found: {len(letters_found)} | Expected letters: {len(expected_chars)}")

    letter_details = []
    reversals = []

    for idx, letter_info in enumerate(letters_found):
        if idx >= len(expected_chars):
            break

        x, y, w, h = letter_info['bbox']
        expected_char = expected_chars[idx]

        # Crop with padding
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(word_img.shape[1], x + w + pad)
        y2 = min(word_img.shape[0], y + h + pad)
        letter_crop = word_img[y1:y2, x1:x2]

        if letter_crop is None or letter_crop.size == 0:
            print(f"      Letter {idx+1} ('{expected_char}'): empty crop, skipped")
            continue

        # Preprocess for model
        letter_for_model = convert_letter_for_model(letter_crop)
        if letter_for_model is None:
            print(f"      Letter {idx+1} ('{expected_char}'): preprocessing failed, skipped")
            continue

        # Save temp file and run YOLO
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{idx}.png')
        cv2.imwrite(temp_path, letter_for_model)

        try:
            result = model(temp_path, conf=0.15, iou=0.5, imgsz=64)[0]
        except Exception as e:
            print(f"      Letter {idx+1} ('{expected_char}'): YOLO error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue

        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Parse YOLO output
        detected_class = CLASS_NORMAL
        confidence = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs   = result.boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)
            detected_class = int(cls_ids[best_idx])
            confidence = float(confs[best_idx])

        class_names = {CLASS_NORMAL: 'normal', CLASS_REVERSAL: 'reversal', CLASS_CORRECTED: 'corrected'}
        class_label = class_names.get(detected_class, 'unknown')

        print(f"      Letter {idx+1} ('{expected_char}'): YOLO → {class_label} ({confidence*100:.1f}%)")

        detail = {
            'position': idx + 1,
            'expected': expected_char,
            'yolo_class': class_label,
            'confidence': round(confidence * 100, 1),
        }

        # Flag reversal and corrected as dyslexia signals
        if detected_class == CLASS_REVERSAL and confidence > threshold:
            reversed_to = None
            if expected_char in REVERSAL_PAIRS:
                reversed_to = REVERSAL_PAIRS[expected_char][0]

            detail['type'] = 'reversal'
            detail['reversed_to'] = reversed_to
            reversals.append(detail)                       # only reversals added here
            print(f"         ⚠️ REVERSAL detected at position {idx+1}")

        elif detected_class == CLASS_CORRECTED and confidence > 0.25:
            detail['type'] = 'corrected'
            # DO NOT append to reversals – correction is not a dyslexia indicator
            print(f"         ℹ️ CORRECTED letter at position {idx+1} (not counted as dyslexia)")

        else:
            detail['type'] = 'normal'

        letter_details.append(detail)

    return letters_found, reversals, letter_details

# ═══════════════ GUARANTEED SEGMENTATION (always returns expected number of boxes) ═══════════════

def segment_letters_guaranteed(word_img, expected_word):
    """
    Always returns a list of bounding boxes with length = len(expected_word).
    Tries contour detection first; if it finds at least 60% of expected letters, uses it.
    Otherwise falls back to equal‑width splitting (guaranteed).
    """
    expected_len = len(expected_word)
    h, w = word_img.shape[:2]
    
    # Try contour detection
    boxes = _segment_letters_contour(word_img)
    if len(boxes) >= expected_len * 0.6:
        print(f"      Segmentation: contour → {len(boxes)} boxes")
        # Trim if too many (due to broken letters)
        if len(boxes) > expected_len:
            boxes = boxes[:expected_len]
        return boxes
    
    # Fallback: equal‑width split (guaranteed to produce exactly expected_len boxes)
    piece_w = w // expected_len
    boxes = []
    for i in range(expected_len):
        x_start = i * piece_w
        # Small overlap to avoid cutting letters
        x_start = max(0, x_start - 2)
        w_adj = min(piece_w + 4, w - x_start)
        boxes.append({'bbox': (x_start, 0, w_adj, h)})
    print(f"      Segmentation: forced equal split → {len(boxes)} boxes")
    return boxes

def _segment_letters_contour(word_img):
    """Same contour‑based function you already have – keep as is."""
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    candidates = []
    # Otsu
    _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary1) > 127:
        binary1 = cv2.bitwise_not(binary1)
    c1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.append(c1)
    # Adaptive
    binary2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    c2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.append(c2)
    # Fixed
    _, binary3 = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    c3, _ = cv2.findContours(binary3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.append(c3)

    best_boxes = []
    best_count = 0
    for contours in candidates:
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 4 and h > 6 and w < 150 and h < 150:
                aspect = w / h if h > 0 else 0
                if 0.1 < aspect < 3.0:
                    boxes.append((x, y, w, h))
        if len(boxes) > best_count:
            best_count = len(boxes)
            best_boxes = sorted(boxes, key=lambda b: b[0])
    return [{'bbox': box} for box in best_boxes]


# ═══════════════ MAIN DETECTION FUNCTION ═══════════════

def is_only_reversal_difference(written, expected):
    """Return True if written differs from expected only by reversal pairs (b/d, p/q, m/w, n/u)."""
    if len(written) != len(expected):
        return False
    reversal_map = {'b':'d', 'd':'b', 'p':'q', 'q':'p', 'm':'w', 'w':'m', 'n':'u', 'u':'n'}
    for wc, ec in zip(written, expected):
        if wc != ec:
            if reversal_map.get(wc) != ec and reversal_map.get(ec) != wc:
                return False
    return True


def analyze_single_word(image_path, expected_word, source='upload'):
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            return {
                'error': 'Could not read image.',
                'expected_word': expected_word,
                'written_word': '(no image)',
                'is_correct': False,
                'has_dyslexia': False,
                'dyslexia_confidence': 0,
                'reversal_details': [],
                'letter_details': [],
                'fun_feedback': '📷 No image found. Please upload a photo!',
                'method': 'no_image'
            }

        # Handle RGBA transparency
        if len(original_img.shape) == 3 and original_img.shape[2] == 4:
            white_bg = np.ones((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8) * 255
            alpha = original_img[:, :, 3] / 255.0
            for c in range(3):
                white_bg[:, :, c] = (original_img[:, :, c] * alpha + white_bg[:, :, c] * (1 - alpha))
            original_img = white_bg

        # Resize
        h, w = original_img.shape[:2]
        if max(h, w) > 1000:
            scale = 800 / max(h, w)
            original_img = cv2.resize(original_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        if max(h, w) < 100:
            scale = 400 / max(h, w)
            original_img = cv2.resize(original_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        expected_word = expected_word.lower().strip()
        print(f"\n{'='*60}")
        print(f"🔍 WORD: '{expected_word}'")
        print(f"{'='*60}")

        # ── OCR (only for reference, not for early decision) ──
        written_word, ocr_confidence = ocr_read_raw(original_img)
        ocr_similarity = SequenceMatcher(None, written_word or '', expected_word).ratio() if written_word else 0.0
        print(f"   OCR result: '{written_word}' (conf: {ocr_confidence:.0f}%, similarity: {ocr_similarity:.2f})")

        # ── YOLO (always runs) ─────────────────────────────────
        print("🧠 YOLO visual letter classification (primary detector)...")
        threshold = 0.5 if source == 'upload' else 0.65
        letters_found, reversals, letter_details = classify_letters_with_yolo(original_img, expected_word, threshold)

        if not letters_found:
            return {
                'expected_word': expected_word,
                'written_word': written_word or '(unreadable)',
                'is_correct': False,
                'has_dyslexia': False,
                'dyslexia_confidence': 0,
                'reversal_details': [],
                'letter_details': [],
                'fun_feedback': '🤔 I could not see any letters. Try writing larger with dark ink!',
                'method': 'yolo_no_segments',
                'result_level': 'Cannot Read'
            }

        count_ratio = len(letters_found) / max(len(expected_word), 1)
        if count_ratio < 0.4:
            return {
                'expected_word': expected_word,
                'written_word': written_word or '(partial)',
                'is_correct': False,
                'has_dyslexia': False,
                'dyslexia_confidence': 0,
                'reversal_details': [],
                'letter_details': letter_details,
                'fun_feedback': f'🤔 I only see {len(letters_found)} of {len(expected_word)} letters. Write each letter clearly with small gaps!',
                'method': 'yolo_partial',
                'result_level': 'Cannot Read Clearly'
            }

        # ── NEW: Decision logic – YOLO reversals take priority ──
        has_dyslexia = len(reversals) > 0
        total_letters_analysed = len(letter_details)
        dyslexia_count = len(reversals)
        dyslexia_confidence = round((dyslexia_count / max(total_letters_analysed, 1)) * 100, 1)

        if has_dyslexia:
            # YOLO found reversals → Dyslexia Detected
            reversal_letters = [r['expected'] for r in reversals if r.get('type') == 'reversal']
            fun_feedback = random.choice([
                f'🔍 The letter "{reversal_letters[0]}" looks reversed. Let\'s practice!',
                f'🔄 I spotted reversed letters – keep working on it!'
            ]) if reversal_letters else '🔍 Reversal patterns detected.'
            result_level = 'Dyslexia Detected'
            is_correct = False
        else:
            # No reversals – now check if the word is wrong or correct
            if written_word and written_word != expected_word:
                fun_feedback = f"I read '{written_word}' but the word was '{expected_word}'. Please write the correct word."
                result_level = 'Incorrect Word'
                is_correct = False
            else:
                fun_feedback = random.choice([
                    '🌟 All letters look correct!',
                    '⭐ Great writing — no reversals found!'
                ])
                result_level = 'No Dyslexia'
                is_correct = True

        print(f"\n📊 YOLO Summary:")
        print(f"   Letters analysed : {total_letters_analysed}")
        print(f"   Reversals found  : {dyslexia_count}")
        print(f"   Dyslexia confidence: {dyslexia_confidence}%")
        print(f"   Result: {result_level}")
        print(f"{'='*60}\n")

        return {
            'expected_word': expected_word,
            'written_word': written_word or '(YOLO analysed)',
            'is_correct': is_correct,
            'has_dyslexia': has_dyslexia,
            'dyslexia_confidence': dyslexia_confidence,
            'reversal_details': reversals,
            'letter_details': letter_details,
            'total_letters': total_letters_analysed,
            'dyslexia_count': dyslexia_count,
            'fun_feedback': fun_feedback,
            'method': 'yolo_primary',
            'result_level': result_level
        }

    except Exception as e:
        print(f"❌ Error in analyze_single_word: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

# ═══════════════ ROUTES ═══════════════

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    # Input validation
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters long.'}), 400

    # Email validation (if provided)
    if email:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return jsonify({'error': 'Please provide a valid email address.'}), 400

    # Password strength validation
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long.'}), 400
    if not re.search(r'[A-Z]', password):
        return jsonify({'error': 'Password must contain at least one uppercase letter.'}), 400
    if not re.search(r'[a-z]', password):
        return jsonify({'error': 'Password must contain at least one lowercase letter.'}), 400
    if not re.search(r'[0-9]', password):
        return jsonify({'error': 'Password must contain at least one number.'}), 400
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return jsonify({'error': 'Password must contain at least one special character.'}), 400

    hashed = generate_password_hash(password)

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, hashed)
        )
        conn.commit()
        return jsonify({'success': True, 'message': 'Registration successful! Please login.'})
    except mysql.connector.IntegrityError:
        return jsonify({'error': 'Username already exists. Please choose another.'}), 409
    finally:
        cursor.close()
        conn.close()


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, password_hash, username FROM users WHERE username = %s",
        (data['username'],)
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user and check_password_hash(user['password_hash'], data['password']):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid username or password'}), 401


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')


@app.route('/canvas')
@login_required
def canvas_page():
    return render_template('canvas.html')


@app.route('/practice')
@login_required
def practice_page():
    return render_template('practice.html')


@app.route('/history')
@login_required
def history_page():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM screenings WHERE user_id = %s ORDER BY created_at DESC LIMIT 20",
        (session['user_id'],)
    )
    screenings = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('history.html', screenings=screenings)


@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/letter-guide')
@login_required
def letter_guide():
    return render_template('letter_guide.html', guides=LETTER_FORMATION)


@app.route('/progress')
@login_required
def progress_page():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT COUNT(*) as total FROM screenings WHERE user_id = %s",
        (session['user_id'],)
    )
    total = cursor.fetchone()['total']

    cursor.execute("""
        SELECT COUNT(*) as correct FROM screenings
        WHERE user_id = %s AND (risk_level LIKE '%Correct%' OR risk_level LIKE '%No Dyslexia%')
    """, (session['user_id'],))
    correct = cursor.fetchone()['correct']

    cursor.execute("""
        SELECT created_at, risk_level,
               JSON_EXTRACT(result_json, '$.written_word') as written,
               JSON_EXTRACT(result_json, '$.expected_word') as expected
        FROM screenings WHERE user_id = %s ORDER BY created_at DESC LIMIT 20
    """, (session['user_id'],))
    recent = cursor.fetchall()

    cursor.execute("""
        SELECT JSON_EXTRACT(result_json, '$.expected_word') as word, COUNT(*) as count
        FROM screenings WHERE user_id = %s AND risk_level LIKE '%Dyslexia%'
        GROUP BY word ORDER BY count DESC LIMIT 5
    """, (session['user_id'],))
    mistakes = cursor.fetchall()

    cursor.close()
    conn.close()

    stars = min(5, (correct // 5) + 1) if total > 0 else 0
    accuracy = round((correct / total * 100), 1) if total > 0 else 0
    return render_template(
        'progress.html',
        total=total, correct=correct,
        stars=stars, accuracy=accuracy,
        recent=recent, mistakes=mistakes
    )


@app.route('/api/user/stats', methods=['GET'])
@login_required
def get_user_stats():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT COUNT(*) as total FROM screenings WHERE user_id = %s",
        (session['user_id'],)
    )
    total = cursor.fetchone()['total']
    cursor.execute(
        """SELECT COUNT(*) as correct FROM screenings
           WHERE user_id = %s
           AND (risk_level LIKE '%Correct%' OR risk_level LIKE '%No Dyslexia%')""",
        (session['user_id'],)
    )
    correct = cursor.fetchone()['correct']
    cursor.close()
    conn.close()
    stars = min(5, (correct // 5) + 1) if total > 0 else 0
    accuracy = round((correct / total * 100), 1) if total > 0 else 0
    return jsonify({
        'total_words': total,
        'correct_words': correct,
        'stars': stars,
        'accuracy': accuracy
    })


@app.route('/api/worksheet')
@login_required
def generate_worksheet():
    level = request.args.get('level', 'easy')
    theme = request.args.get('theme', 'animals')
    lang  = request.args.get('lang', 'english')

    if lang == 'malay':
        words = MALAY_WORDS.get(level, MALAY_WORDS['easy'])[:8]
    elif theme in THEMED_WORDS:
        words = THEMED_WORDS[theme].get(level, THEMED_WORDS[theme]['easy'])[:8]
    else:
        words = DEFAULT_WORDS.get(level, DEFAULT_WORDS['easy'])[:8]

    buffer = io.BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "✏️ Handwriting Practice Worksheet")
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 80, f"Level: {level.title()} | Theme: {theme.title()} | Language: {lang.title()}")
    c.drawString(50, height - 100, "Write each word 3 times:")
    y = height - 150
    c.setFont("Helvetica-Bold", 18)
    for word in words:
        c.drawString(50, y, word)
        c.setFont("Helvetica", 12)
        for i in range(3):
            c.setStrokeColorRGB(0.8, 0.8, 0.8)
            c.line(200, y - 5, 500, y - 5)
            y -= 25
        y -= 15
        c.setFont("Helvetica-Bold", 18)
    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/pdf',
        download_name=f'worksheet_{level}_{theme}.pdf',
        as_attachment=True
    )


@app.route('/phrase')
@login_required
def phrase_page():
    return render_template('phrase.html')


@app.route('/api/get_phrase', methods=['GET'])
def get_phrase():
    level = request.args.get('level', 'easy')
    if level == 'easy':
        phrase = random.choice(SHORT_PHRASES[:3])
    elif level == 'medium':
        phrase = random.choice(SHORT_PHRASES[3:5])
    else:
        phrase = random.choice(SHORT_PHRASES)
    return jsonify({'phrase': phrase, 'level': level})


@app.route('/game/bingo')
@login_required
def game_bingo():
    return render_template('game_bingo.html')


@app.route('/game/lotto')
@login_required
def game_lotto():
    return render_template('game_lotto.html')


@app.route('/game/reversal')
@login_required
def game_reversal():
    return render_template('game_reversal.html')


@app.route('/game/words')
@login_required
def game_words():
    return render_template('game_words.html')


@app.route('/game/quiz')
@login_required
def game_quiz():
    return render_template('game_quiz.html')


@app.route('/game/memory')
@login_required
def game_memory():
    return render_template('game_memory.html')


@app.route('/api/game/get_challenge', methods=['GET'])
def get_challenge():
    pairs = [('b', 'd'), ('p', 'q'), ('m', 'w'), ('n', 'u')]
    pair = random.choice(pairs)
    show = random.choice(pair)
    correct = pair[0] if show == pair[1] else show
    return jsonify({'letter': show, 'correct_answer': correct, 'options': list(pair)})


@app.route('/api/game/check_letter', methods=['POST'])
def check_letter():
    data = request.json
    is_correct = (data.get('letter', '').lower() == data.get('expected', '').lower())
    return jsonify({
        'correct': is_correct,
        'feedback': '✅ Great!' if is_correct else '❌ Try again!'
    })


@app.route('/api/game/submit_score', methods=['POST'])
@login_required
def submit_game_score():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO game_scores (user_id, game_type, score, total_questions) VALUES (%s,%s,%s,%s)",
        (session['user_id'], data['game_type'], data['score'], data['total'])
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/get_word', methods=['GET'])
def get_word():
    level = request.args.get('level', 'easy')
    theme = request.args.get('theme', None)
    lang  = request.args.get('lang', 'english')

    if lang == 'malay':
        word_list = MALAY_WORDS.get(level, MALAY_WORDS['easy'])
    elif theme and theme in THEMED_WORDS:
        word_list = THEMED_WORDS[theme].get(level, THEMED_WORDS[theme]['easy'])
    else:
        word_list = DEFAULT_WORDS.get(level, DEFAULT_WORDS['easy'])

    word = random.choice(word_list)
    return jsonify({'word': word, 'level': level, 'theme': theme, 'lang': lang})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filepath, filename = save_uploaded_image(file)
        expected_word = request.form.get('expected_word', 'dog').lower().strip()
    
        source = request.form.get('source', 'upload')
        result = analyze_single_word(filepath, expected_word, source=source)

        # ─── KEEP IMAGES (PERMANENTLY SAVED) ──────────────────────────
        # Image is now stored permanently in static/uploads/
        # DO NOT delete – it will be displayed in History page
        # try:
        #     if os.path.exists(filepath):
        #         os.remove(filepath)
        #         print(f"🗑️ Deleted uploaded image: {filename}")
        # except Exception as e:
        #     print(f"⚠️ Could not delete image: {e}")

        # ✅ Clean up any leftover temp files from YOLO
        cleanup_temp_files()

        # Save to database
        if 'user_id' in session and 'error' not in result:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO screenings (user_id, image_path, total_letters, normal_count,
                 reversal_count, corrected_count, risk_score, risk_level, result_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (session['user_id'], filename,
                  result.get('total_letters', len(expected_word)),
                  result.get('total_letters', len(expected_word)) - result.get('dyslexia_count', 0),
                  sum(1 for r in result.get('reversal_details', []) if r.get('type') == 'reversal'),
                  sum(1 for r in result.get('reversal_details', []) if r.get('type') == 'corrected'),
                  result.get('dyslexia_confidence', 0),
                  result.get('result_level', 'Unknown'),
                  json.dumps(result)))
            conn.commit()
            cursor.close()
            conn.close()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🦉 Smart Dyslexia Detector — YOLO Primary Edition")
    print("=" * 60)
    print(f"🔍 OCR    : {'Available (identity check only)' if OCR_AVAILABLE else 'Not Available'}")
    print(f"🧠 YOLO   : {'Loaded — PRIMARY detector' if model else 'Not Loaded ❌'}")
    print()
    print("Pipeline:")
    print("  1. OCR  → What word did the child write? (identity only)")
    print("  2. YOLO → Is each letter normal / reversed / corrected? (detection)")
    print("  3. Result based on YOLO visual classification")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
