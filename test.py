from spellchecker import SpellChecker
spell = SpellChecker()

# Test words
print("'dog' is a word:", 'dog' in spell)
print("'bog' is a word:", 'bog' in spell)
print("'the' is a word:", 'the' in spell)
print("'quick' is a word:", 'quick' in spell)
print("'brown' is a word:", 'brown' in spell)