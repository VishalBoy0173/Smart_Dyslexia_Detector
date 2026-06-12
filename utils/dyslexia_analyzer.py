import cv2
import numpy as np
import os

class DyslexiaAnalyzer:
    """Analyze handwriting for dyslexia indicators"""
    
    REVERSAL_PAIRS = {'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p', 'm': 'w', 'w': 'm', 'n': 'u', 'u': 'n'}
    NO_REVERSAL = set('acefghijklmorstvxyz')
    
    def __init__(self, model, spell):
        self.model = model
        self.spell = spell
    
    def analyze_word(self, word_img, expected_word, word_index, preprocessor, letter_segmenter):
        """Analyze a single word for dyslexia patterns"""
        try:
            # Segment letters
            letters = letter_segmenter.segment_letters(word_img)
            
            if not letters or len(letters) < 2:
                return None
            
            expected_chars = list(expected_word)
            letter_analyses = []
            reversal_count = 0
            
            for i, letter_data in enumerate(letters[:len(expected_chars)]):
                letter_img = letter_data['image']
                expected_char = expected_chars[i] if i < len(expected_chars) else ''
                
                # Analyze single letter
                analysis = self.analyze_letter(letter_img, expected_char, i)
                letter_analyses.append(analysis)
                
                if analysis['is_reversal']:
                    reversal_count += 1
            
            total_letters = len(letter_analyses)
            reversal_rate = (reversal_count / total_letters * 100) if total_letters > 0 else 0
            
            return {
                'word_index': word_index,
                'expected_word': expected_word,
                'letter_analysis': letter_analyses,
                'reversal_count': reversal_count,
                'total_letters': total_letters,
                'reversal_rate': round(reversal_rate, 1),
                'is_dyslexic': reversal_count > 0,
                'avg_confidence': round(np.mean([l['confidence'] for l in letter_analyses]) * 100, 1)
            }
        except Exception as e:
            print(f"Word analysis error: {e}")
            return None
    
    def analyze_letter(self, letter_img, expected_char, position):
        """Analyze a single letter using YOLOv11"""
        try:
            # Convert to model format
            letter_gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
            letter_mean = np.mean(letter_gray)
            
            if letter_mean > 127:
                letter_inverted = cv2.bitwise_not(letter_gray)
            else:
                letter_inverted = letter_gray
            
            _, letter_bw = cv2.threshold(letter_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            letter_final = cv2.cvtColor(letter_bw, cv2.COLOR_GRAY2BGR)
            letter_resized = cv2.resize(letter_final, (64, 64), interpolation=cv2.INTER_CUBIC)
            
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            canvas[:64, :64] = letter_resized
            
            # Save temp and run model
            temp_path = f'temp_letter_{position}.png'
            cv2.imwrite(temp_path, canvas)
            result = self.model(temp_path, conf=0.15, iou=0.5, imgsz=64)[0]
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            detected_class = 0
            confidence = 0.0
            
            if result.boxes is not None and len(result.boxes) > 0:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                best_idx = np.argmax(confs)
                detected_class = int(cls_ids[best_idx])
                confidence = float(confs[best_idx])
            
            # Check if reversal
            is_reversal = (detected_class == 1 and expected_char in self.REVERSAL_PAIRS)
            
            return {
                'position': position + 1,
                'expected_char': expected_char,
                'detected_class': detected_class,
                'confidence': confidence,
                'is_reversal': is_reversal,
                'reversal_partner': self.REVERSAL_PAIRS.get(expected_char, '') if is_reversal else ''
            }
        except Exception as e:
            return {
                'position': position + 1,
                'expected_char': expected_char,
                'detected_class': 0,
                'confidence': 0.0,
                'is_reversal': False,
                'reversal_partner': ''
            }
    
    def calculate_risk_score(self, word_analyses, total_words):
        """Calculate overall risk score"""
        valid_analyses = [w for w in word_analyses if w is not None]
        dyslexic_words = [w for w in valid_analyses if w['is_dyslexic']]
        
        if total_words == 0:
            return 0, 'No words detected', 'none'
        
        risk_score = round((len(dyslexic_words) / total_words * 100), 2)
        
        if risk_score < 10:
            return risk_score, 'Very Low Risk', 'very_low'
        elif risk_score < 20:
            return risk_score, 'Low Risk', 'low'
        elif risk_score < 40:
            return risk_score, 'Moderate Risk', 'moderate'
        elif risk_score < 60:
            return risk_score, 'High Risk', 'high'
        else:
            return risk_score, 'Very High Risk', 'very_high'