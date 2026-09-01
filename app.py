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

import cv2
import numpy as np
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
from spellchecker import SpellChecker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
import io
import re

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

# ─── Word Lists ───────────────────────────────────────────────
WORD_LISTS = {
    'easy': {
        'default': ['dog'],
        'animals': ['dog'],
        'food': ['egg'],
        'colors': ['red']
    },
    'medium': {
        'default': ['rabbit'],
        'animals': ['rabbit'],
        'food': ['apple'],
        'colors': ['purple']
    }
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
        
        # Check if there's a reversal detection
        reversal_confidence = 0.0
        has_reversal = False
        
        # Check if there's a corrected detection
        corrected_confidence = 0.0
        has_corrected = False

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            # Get the highest confidence overall
            best_idx = np.argmax(confs)
            detected_class = int(cls_ids[best_idx])
            confidence = float(confs[best_idx])
            
            # Check for reversal detections
            for cls_id, conf in zip(cls_ids, confs):
                if int(cls_id) == CLASS_REVERSAL:
                    has_reversal = True
                    reversal_confidence = max(reversal_confidence, float(conf))
                elif int(cls_id) == CLASS_CORRECTED:
                    has_corrected = True
                    corrected_confidence = max(corrected_confidence, float(conf))

        class_names = {CLASS_NORMAL: 'normal', CLASS_REVERSAL: 'reversal', CLASS_CORRECTED: 'corrected'}
        class_label = class_names.get(detected_class, 'unknown')

        print(f"      Letter {idx+1} ('{expected_char}'): YOLO → {class_label} ({confidence*100:.1f}%)")
        if has_reversal:
            print(f"         → Reversal detected with {reversal_confidence*100:.1f}% confidence")
        if has_corrected:
            print(f"         → Corrected detected with {corrected_confidence*100:.1f}% confidence")

        detail = {
            'position': idx + 1,
            'expected': expected_char,
            'yolo_class': class_label,
            'confidence': round(confidence * 100, 1),
        }

        # Flag reversal and corrected as dyslexia signals
        # Check if ANY reversal detection meets the threshold
        if has_reversal and reversal_confidence > threshold:
            reversed_to = None
            if expected_char in REVERSAL_PAIRS:
                reversed_to = REVERSAL_PAIRS[expected_char][0]

            detail['type'] = 'reversal'
            detail['reversed_to'] = reversed_to
            detail['reversal_confidence'] = round(reversal_confidence * 100, 1)
            reversals.append(detail)  # Count as dyslexia
            print(f"         ⚠️ REVERSAL counted at position {idx+1} with {reversal_confidence*100:.1f}% confidence")

        elif has_corrected and corrected_confidence > 0.25:
            detail['type'] = 'corrected'
            detail['corrected_confidence'] = round(corrected_confidence * 100, 1)
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
    """
    Main dyslexia detection function.
    
    Flow:
    1. Load and preprocess image
    2. OCR reads the word (identity)
    3. RUN YOLO on the DETECTED word (what the user actually wrote)
    4. Compare with expected word
    5. Return results with transparency including 'detected' field
    
    Args:
        image_path: Path to the uploaded image
        expected_word: The word the user was supposed to write
        source: 'upload' or 'canvas' (affects confidence threshold)
    
    Returns:
        Dict with analysis results including transparency data
    """
    try:
        # ======================================================================
        # STEP 1: Load Image
        # ======================================================================
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
                'method': 'no_image',
                'result_level': 'Error',
                'show_letter_table': False
            }

        # Handle RGBA transparency (for canvas images)
        if len(original_img.shape) == 3 and original_img.shape[2] == 4:
            white_bg = np.ones((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8) * 255
            alpha = original_img[:, :, 3] / 255.0
            for c in range(3):
                white_bg[:, :, c] = (original_img[:, :, c] * alpha + white_bg[:, :, c] * (1 - alpha))
            original_img = white_bg

        # Resize for better processing
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

        # ======================================================================
        # STEP 2: OCR - What word did the user actually write?
        # ======================================================================
        written_word, ocr_confidence = ocr_read_raw(original_img)
        
        # Convert to Python native types (JSON serialization fix)
        ocr_confidence = float(ocr_confidence) if ocr_confidence is not None else 0.0
        ocr_similarity = float(SequenceMatcher(None, written_word or '', expected_word).ratio()) if written_word else 0.0
        
        print(f"   OCR result: '{written_word}' (conf: {ocr_confidence:.0f}%, similarity: {ocr_similarity:.2f})")

        # Store OCR data for transparency display
        ocr_data = {
            'text': str(written_word or ''),
            'confidence': float(round(ocr_confidence, 1)),
            'similarity': float(round(ocr_similarity, 2)),
            'is_trustworthy': bool(ocr_confidence >= 30)
        }

        # ======================================================================
        # STEP 3: Determine which word to analyze with YOLO
        # ======================================================================
        # We ALWAYS analyze the DETECTED word (what the user actually wrote)
        # Because we want to check THEIR writing for dyslexia patterns
        #
        # However, if OCR confidence is very low, we may not trust the detected word
        # In that case, we use the expected word but warn the user
        
        use_detected_word = False
        analysis_word = expected_word  # Default to expected word
        
        if written_word and ocr_confidence >= 30:
            # OCR is trustworthy - use the detected word
            use_detected_word = True
            analysis_word = written_word.lower()
            print(f"   ✅ Using DETECTED word for analysis: '{analysis_word}'")
        elif written_word and 20 <= ocr_confidence < 30:
            # OCR is partially trustworthy - use detected word but warn
            use_detected_word = True
            analysis_word = written_word.lower()
            print(f"   ⚠️ Using DETECTED word with low confidence: '{analysis_word}'")
        elif written_word and ocr_confidence < 20:
            # OCR is very unreliable - use expected word
            use_detected_word = False
            analysis_word = expected_word
            print(f"   ⚠️ OCR confidence too low ({ocr_confidence:.0f}%) - using EXPECTED word for analysis")
        else:
            # No OCR result - use expected word
            analysis_word = expected_word
            print(f"   ⚠️ No OCR result - using EXPECTED word for analysis")

        # ======================================================================
        # STEP 4: RUN YOLO on the ANALYSIS word
        # ======================================================================
        print(f"🧠 Running YOLO on: '{analysis_word}'")
        
        # Set threshold based on source
        threshold = 0.4 if source == 'upload' else 0.65
        
        # Run YOLO classification on the analysis word
        letters_found, reversals, letter_details = classify_letters_with_yolo(
            original_img, 
            analysis_word,  # ← The word we're analyzing (detected OR expected)
            threshold
        )

        # Check if any letters were found
        if not letters_found:
            return {
                'expected_word': str(expected_word),
                'written_word': str(written_word or ''),
                'analysis_word': str(analysis_word),
                'is_correct': False,
                'has_dyslexia': False,
                'dyslexia_confidence': 0,
                'reversal_details': [],
                'letter_details': [],
                'ocr_data': ocr_data,
                'fun_feedback': '🤔 I could not see any letters. Try writing larger with dark ink!',
                'method': 'yolo_no_segments',
                'result_level': 'Cannot Read',
                'show_letter_table': False,
                'word_mismatch': bool(written_word and written_word.lower() != expected_word)
            }

        # Check if enough letters were detected (at least 40%)
        count_ratio = len(letters_found) / max(len(analysis_word), 1)
        if count_ratio < 0.4:
            return {
                'expected_word': str(expected_word),
                'written_word': str(written_word or ''),
                'analysis_word': str(analysis_word),
                'is_correct': False,
                'has_dyslexia': False,
                'dyslexia_confidence': 0,
                'reversal_details': [],
                'letter_details': letter_details,
                'ocr_data': ocr_data,
                'fun_feedback': f'🤔 I only see {len(letters_found)} of {len(analysis_word)} letters. Write each letter clearly with small gaps!',
                'method': 'yolo_partial',
                'result_level': 'Cannot Read Clearly',
                'show_letter_table': True,
                'word_mismatch': bool(written_word and written_word.lower() != expected_word)
            }

        # ======================================================================
        # STEP 5: Filter Reversals - ONLY Valid Dyslexia Reversal Pairs
        # ======================================================================
        VALID_REVERSAL_PAIRS = ['b', 'd', 'p', 'q', 'm', 'w', 'n', 'u']
        
        valid_reversals = []
        for r in reversals:
            expected_letter = r.get('expected', '').lower()
            if expected_letter in VALID_REVERSAL_PAIRS:
                valid_reversals.append(r)
            else:
                print(f"   ⚠️ Ignoring reversal for '{expected_letter}' - not in valid reversal pairs")
                for detail in letter_details:
                    if detail.get('position') == r.get('position'):
                        detail['ignored_reversal'] = True

        # ======================================================================
        # STEP 6: Calculate Results
        # ======================================================================
        has_dyslexia = bool(len(valid_reversals) > 0)
        total_letters_analysed = int(len(letter_details))
        dyslexia_count = int(len(valid_reversals))
        dyslexia_confidence = float(round((dyslexia_count / max(total_letters_analysed, 1)) * 100, 1))

        # Check if word matches expected
        word_matches = bool(written_word and written_word.lower() == expected_word)
        word_mismatch = bool(written_word and not word_matches)

        is_correct = False
        result_level = ''
        fun_feedback = ''

        # ── Build the feedback message ──
        
        # Part 1: Word mismatch warning (if applicable)
        word_warning = ""
        if word_mismatch:
            word_warning = f"📝 You wrote '{written_word}' but expected '{expected_word}'. "

        # Part 2: Dyslexia detection result
        if has_dyslexia:
            result_level = 'Dyslexia Detected'
            is_correct = False
            reversal_letters = [r['expected'] for r in valid_reversals]
            
            if len(valid_reversals) == 1:
                dyslexia_msg = f"🔍 Found 1 reversed letter: '{reversal_letters[0]}'."
            else:
                dyslexia_msg = f"🔍 Found {len(valid_reversals)} reversed letters: {', '.join(reversal_letters)}."
            
            dyslexia_msg += f" (Confidence: {dyslexia_confidence}%)"
            
            # Combine messages
            if word_mismatch:
                fun_feedback = word_warning + " " + dyslexia_msg + " Practice writing these letters correctly!"
            else:
                fun_feedback = dyslexia_msg + " Practice writing these letters correctly!"
                
        else:
            # No reversals found
            if word_mismatch:
                result_level = 'Incorrect Word - No Dyslexia'
                is_correct = False
                fun_feedback = word_warning + "✅ No dyslexia patterns detected in your writing. Try writing the correct word next time!"
            else:
                result_level = 'No Dyslexia'
                is_correct = True
                fun_feedback = random.choice([
                    '🌟 All letters look correct! Great writing!',
                    '⭐ Perfect! No reversals found. Keep up the good work!'
                ])

        # ======================================================================
        # STEP 7: Determine if we should show the letter table
        # ======================================================================
        show_table = bool(result_level not in [
            'Cannot Read', 
            'Cannot Read Clearly', 
            'Error'
        ])

        # ======================================================================
        # STEP 8: ENHANCE LETTER DETAILS WITH 'detected' FIELD
        # ======================================================================
        # This is the key addition - adds the 'detected' field to each letter
        # so the frontend can display what the system actually detected
        analysis_chars = list(analysis_word)
        enhanced_letter_details = []
        
        for idx, detail in enumerate(letter_details):
            # Get the detected character from the analysis word
            detected_char = analysis_chars[idx] if idx < len(analysis_chars) else '?'
            
            # Create enhanced detail with detected field
            enhanced_detail = {
                'position': detail.get('position', idx + 1),
                'expected': detail.get('expected', '?'),
                'detected': detected_char,  # ← NEW: What the system actually detected
                'yolo_class': detail.get('yolo_class', 'unknown'),
                'confidence': detail.get('confidence', 0),
                'type': detail.get('type', 'normal'),
            }
            
            # Copy over reversal confidence if present
            if detail.get('type') == 'reversal':
                enhanced_detail['reversal_confidence'] = detail.get('reversal_confidence', 0)
            elif detail.get('type') == 'corrected':
                enhanced_detail['corrected_confidence'] = detail.get('corrected_confidence', 0)
            
            enhanced_letter_details.append(enhanced_detail)

        # ======================================================================
        # STEP 9: Print Summary
        # ======================================================================
        print(f"\n📊 YOLO Summary:")
        print(f"   Analysis word  : '{analysis_word}'")
        print(f"   Letters analysed: {total_letters_analysed}")
        print(f"   Reversals found : {dyslexia_count}")
        print(f"   Valid reversals : {len(valid_reversals)}")
        print(f"   Dyslexia confidence: {dyslexia_confidence}%")
        print(f"   Word matches    : {word_matches}")
        print(f"   Result: {result_level}")
        print(f"{'='*60}\n")

        # ======================================================================
        # STEP 10: Return Results with enhanced letter details
        # ======================================================================
        return {
            'expected_word': str(expected_word),
            'written_word': str(written_word or ''),
            'analysis_word': str(analysis_word),
            'is_correct': bool(is_correct),
            'has_dyslexia': bool(has_dyslexia),
            'dyslexia_confidence': float(dyslexia_confidence),
            'reversal_details': valid_reversals,
            'letter_details': enhanced_letter_details,  # ← ENHANCED WITH 'detected' FIELD
            'total_letters': int(total_letters_analysed),
            'dyslexia_count': int(dyslexia_count),
            'ocr_data': ocr_data,
            'method': 'yolo_primary',
            'result_level': str(result_level),
            'fun_feedback': str(fun_feedback),
            'show_letter_table': show_table,
            'word_mismatch': bool(word_mismatch)
        }

    except Exception as e:
        print(f"❌ Error in analyze_single_word: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'show_letter_table': False
        }
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

@app.route('/worksheets')
@login_required
def worksheets_page():
    return render_template('worksheet.html')

# ═══════════════ GAME ROUTES ═══════════════

@app.route('/game/spelling-bee')
@login_required
def game_spelling_bee():
    return render_template('spelling_bee.html')

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
    # Get only the theme parameter (no level or lang)
    theme = request.args.get('theme', 'animals')
    
    # Get words from BOTH easy and medium levels for the selected theme
    words = []
    
    # Get Easy words
    if theme in WORD_LISTS.get('easy', {}):
        easy_words = WORD_LISTS['easy'].get(theme, WORD_LISTS['easy']['default'])
        words.extend(easy_words[:8])  # Take first 8
    
    # Get Medium words
    if theme in WORD_LISTS.get('medium', {}):
        medium_words = WORD_LISTS['medium'].get(theme, WORD_LISTS['medium']['default'])
        words.extend(medium_words[:8])  # Take first 8
    
    # Fallback: if no words found (shouldn't happen)
    if not words:
        words = WORD_LISTS['easy']['default'] + WORD_LISTS['medium']['default']
    
    # Limit to 16 words total (8 easy + 8 medium)
    words = words[:16]
    
    # Create PDF
    buffer = io.BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.setFillColorRGB(0.18, 0.23, 0.55)  # Dark blue
    c.drawString(50, height - 50, "✏️ Handwriting Practice Worksheet")
    
    c.setFont("Helvetica", 14)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(50, height - 80, f"Theme: {theme.title()} (16 words)")
    
    # Draw a line under header
    c.setStrokeColorRGB(0.18, 0.23, 0.55)
    c.setLineWidth(2)
    c.line(50, height - 110, width - 50, height - 110)
    
    # Start word list
    y = height - 150
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0, 0, 0)
    
    # Track which words are easy vs medium for labeling
    easy_count = len(WORD_LISTS['easy'].get(theme, WORD_LISTS['easy']['default'])) if theme in WORD_LISTS.get('easy', {}) else 8
    easy_count = min(easy_count, 8)
    
    for idx, word in enumerate(words):
        # Add level label for first easy and first medium word
        if idx == 0:
            c.setFont("Helvetica-Bold", 12)
            c.setFillColorRGB(0, 0.5, 0)  # Green
            c.drawString(50, y + 5, "🔵 EASY WORDS")
            c.setFont("Helvetica-Bold", 18)
            c.setFillColorRGB(0, 0, 0)
            y -= 20
        elif idx == easy_count:
            c.setFont("Helvetica-Bold", 12)
            c.setFillColorRGB(0.8, 0.5, 0)  # Orange
            c.drawString(50, y + 5, "🟠 MEDIUM WORDS")
            c.setFont("Helvetica-Bold", 18)
            c.setFillColorRGB(0, 0, 0)
            y -= 20
        
        # Draw the word
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, y, word)
        
        # Draw 3 writing lines
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.8, 0.8, 0.8)
        line_start_x = 180
        line_end_x = 530
        
        for i in range(3):
            c.setStrokeColorRGB(0.8, 0.8, 0.8)
            c.setLineWidth(1)
            # Draw dashed line for writing
            c.setDash(3, 3)  # Dashed line
            c.line(line_start_x, y - 5 - (i * 25), line_end_x, y - 5 - (i * 25))
            c.setDash()  # Reset to solid
        
        y -= 75  # Move down for next word
        
        # Check if we need a new page
        if y < 80:
            c.showPage()
            y = height - 80
            c.setFont("Helvetica-Bold", 18)
    
    # Footer
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.6, 0.6, 0.6)
    c.drawString(50, 30, f"Generated by DyslexiAI - Theme: {theme.title()}")
    c.drawString(50, 20, "Practice each word 3 times. Take a photo and upload for analysis!")
    
    c.save()
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        download_name=f'worksheet_{theme}.pdf',
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
    lang = request.args.get('lang', 'english')  # Keep for compatibility

    # Use WORD_LISTS instead of old variables
    level_words = WORD_LISTS.get(level, WORD_LISTS['easy'])
    
    # Get theme-specific words or fallback to default
    word_list = level_words.get(theme, level_words['default']) if theme else level_words['default']

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
