import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources exist
resources = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
]

for path, package in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)

# Optional: download if your NLTK version needs it
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

STOP_WORDS = set(stopwords.words("english"))
PUNCT = string.punctuation


def clean_text(text):
    """Lowercase, remove punctuation and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS and not w.isnumeric()]
    return " ".join(tokens)


def parse_catalog_content(text):
    """Extract item_name, brand_name, bullet_points, product_description,
    value, unit from the raw catalog_content string — same regexes used
    during training."""
    if pd.isna(text) or text is None:
        return {
            "item_name": "",
            "brand_name": "Unknown",
            "bullet_points": "",
            "product_description": "",
            "value": None,
            "unit": None,
        }

    result = {}

    item = re.search(
        r"Item Name:\s*(.*?)(?=Bullet Point\s*\d*:|Product Description:|Value:|Unit:|$)",
        text, flags=re.DOTALL,
    )
    bullets = re.findall(
        r"Bullet Point\s*\d*:\s*(.*?)(?=Bullet Point\s*\d*:|Product Description:|Value:|Unit:|$)",
        text, flags=re.DOTALL,
    )
    desc = re.search(
        r"Product Description:\s*(.*?)(?=Value:|Unit:|$)",
        text, flags=re.DOTALL,
    )
    value = re.search(r"Value:\s*([\d.]+)", text)
    unit = re.search(r"Unit:\s*(.*)$", text)

    if item:
        item_text = item.group(1).strip()
        result["item_name"] = clean_text(item_text)
        words = item_text.split()
        result["brand_name"] = words[0] if words else "Unknown"
    else:
        result["item_name"] = ""
        result["brand_name"] = "Unknown"

    result["bullet_points"] = " ".join(clean_text(x) for x in bullets) if bullets else ""
    result["product_description"] = clean_text(desc.group(1)) if desc else ""
    result["value"] = float(value.group(1)) if value else None
    result["unit"] = unit.group(1).strip() if unit else None

    return result