from num2words import num2words
import re

def normalize_nums(text, language='ar'):
    def replace(match):
        number = int(match.group())
        return num2words(number, lang=language)
    return re.sub(r"\d+", replace, text)