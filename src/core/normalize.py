"""Text normalization utilities for multilingual search (EN/AR)."""
import re

# Arabic diacritics pattern
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

# Alef variants
ALEF = re.compile(r'[\u0622\u0623\u0625]')

# Arabic-Indic to Western digits
AR_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def normalize_text(s: str, remove_diacritics: bool = True) -> str:
    """
    Normalize text for search (EN/AR).
    
    Steps:
    - Lowercase
    - Convert Arabic-Indic digits to Western
    - Remove Arabic diacritics (optional)
    - Normalize Alef variants
    - Normalize Alif Maqsura to Yaa
    - Strip whitespace
    
    Args:
        s: Input text
        remove_diacritics: Whether to remove Arabic diacritics
    
    Returns:
        Normalized text
    """
    if not s:
        return ""
    
    # Lowercase and strip
    s = s.strip().lower()
    
    # Convert Arabic-Indic digits to Western
    s = s.translate(AR_DIGITS)
    
    # Remove diacritics
    if remove_diacritics:
        s = AR_DIAC.sub('', s)
    
    # Normalize Alef variants to base Alef (U+0627)
    s = ALEF.sub('\u0627', s)
    
    # Normalize Alif Maqsura (U+0649) to Yaa (U+064A)
    s = s.replace('\u0649', '\u064A')
    
    return s


def normalize_batch(texts: list[str], remove_diacritics: bool = True) -> list[str]:
    """Normalize a batch of texts."""
    return [normalize_text(t, remove_diacritics) for t in texts]
