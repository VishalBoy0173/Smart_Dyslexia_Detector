import re

class OCRHandler:
    """Handle OCR text extraction and comparison"""
    
    def __init__(self, tesseract_path=None):
        self.tesseract_path = tesseract_path
    
    def extract_text(self, img):
        """Extract text from image using OCR"""
        try:
            import pytesseract
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            text = pytesseract.image_to_string(img)
            return text.strip()
        except:
            return None
    
    def compare_with_expected(self, extracted_text, expected_text):
        """Compare extracted text with expected sentence"""
        extracted_words = re.findall(r'\b\w+\b', extracted_text.lower())
        expected_words = re.findall(r'\b\w+\b', expected_text.lower())
        
        matches = []
        mismatches = []
        
        for i, expected in enumerate(expected_words):
            if i < len(extracted_words):
                extracted = extracted_words[i]
                if extracted == expected:
                    matches.append({'position': i + 1, 'word': expected, 'status': 'match'})
                else:
                    mismatches.append({
                        'position': i + 1,
                        'expected': expected,
                        'extracted': extracted,
                        'status': 'mismatch'
                    })
            else:
                mismatches.append({
                    'position': i + 1,
                    'expected': expected,
                    'extracted': None,
                    'status': 'missing'
                })
        
        total = max(len(expected_words), 1)
        match_rate = (len(matches) / total) * 100
        
        return {
            'match_rate': round(match_rate, 1),
            'matches': matches,
            'mismatches': mismatches,
            'total_expected': len(expected_words),
            'total_extracted': len(extracted_words)
        }