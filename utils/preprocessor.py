import cv2
import numpy as np

class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    def check_image_quality(self, img):
        """Check if image quality is acceptable"""
        issues = []
        score = 100
        
        h, w = img.shape[:2]
        
        # Check resolution
        if w < 400 or h < 300:
            issues.append("Low resolution")
            score -= 30
        
        # Check brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50:
            issues.append("Too dark")
            score -= 20
        elif brightness > 230:
            issues.append("Too bright/washed out")
            score -= 20
        
        # Check blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            issues.append("Image is blurry")
            score -= 30
        
        return {
            'quality': 'good' if score >= 60 else 'poor',
            'score': max(0, score),
            'issues': issues
        }
    
    def enhance_image(self, img):
        """Enhance image for better OCR and detection"""
        result = {}
        result['original'] = img.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result['gray'] = gray
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result['enhanced'] = clahe.apply(gray)
        
        # Binary threshold
        avg_brightness = np.mean(gray)
        if avg_brightness > 127:
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        result['binary'] = binary
        
        # Cleaned version
        kernel = np.ones((2, 2), np.uint8)
        result['cleaned'] = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def correct_skew(self, img):
        """Correct image skew/tilt"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) < 100:
                return img
            
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) < 0.3:
                return img
            
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        except:
            return img


class WordDetector:
    """Detect word regions in an image"""
    
    def detect_words(self, img, binary):
        """Find word bounding boxes"""
        h, w = img.shape[:2]
        
        # Dilate to connect letters into words
        kernel = np.ones((5, 20), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        words = []
        for contour in contours:
            x, y, ww, wh = cv2.boundingRect(contour)
            if ww < 20 or wh < 10:
                continue
            words.append({
                'bbox': (x, y, ww, wh),
                'area': ww * wh,
                'aspect_ratio': ww / wh if wh > 0 else 0
            })
        
        words.sort(key=lambda w: (w['bbox'][1] // 20, w['bbox'][0]))
        return words


class LetterSegmenter:
    """Segment individual letters from a word image"""
    
    def segment_letters(self, word_img):
        """Extract individual letters from a word image"""
        if word_img is None or word_img.size == 0:
            return []
        
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 127:
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        letters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 5 or h < 5:
                continue
            
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(word_img.shape[1], x + w + pad)
            y2 = min(word_img.shape[0], y + h + pad)
            
            letter_img = word_img[y1:y2, x1:x2]
            if letter_img.size > 0:
                letters.append({
                    'image': letter_img,
                    'bbox': (x, y, w, h),
                    'position': (x1, y1, x2, y2)
                })
        
        letters.sort(key=lambda l: l['bbox'][0])
        return letters