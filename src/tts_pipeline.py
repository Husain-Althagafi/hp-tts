from num2words import num2words
import re
import unicodedata

def normalize_nums(text, language='ar'):
    def replace(match):
        number = int(match.group())
        return num2words(number, lang=language)
    return re.sub(r"\d+", replace, text)


def normalize_text(t):
    t = re.sub(r"[^\w\s]", "", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip()

