import re
import unicodedata
import hashlib
from typing import Set, Tuple


class TextNormalizer:
    """
    Utility for canonicalizing question text to detect exact and near-exact duplicates
    regardless of casing, punctuation, Unicode variations, and whitespace differences.
    """

    CONTRACTIONS = {
        r"\bwhat's\b": "what is",
        r"\bthat's\b": "that is",
        r"\bthere's\b": "there is",
        r"\bwho's\b": "who is",
        r"\bhow's\b": "how is",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bit's\b": "it is",
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Produce a normalized string for exact comparison and hashing.
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Lowercase
        text = text.lower()

        # 3. Expand common contractions
        for pattern, replacement in cls.CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)

        # 4. Remove all punctuation and symbols, keep alphanumeric and spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # 5. Normalize whitespace (collapse multiple spaces, strip ends)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def compute_hash(cls, normalized_text: str) -> str:
        """
        Compute an MD5 hash of the normalized text for O(1) exact matching.
        """
        return hashlib.md5(normalized_text.encode("utf-8")).hexdigest()

    @classmethod
    def get_token_set(cls, normalized_text: str) -> Set[str]:
        """
        Return the set of unique words in normalized text.
        """
        tokens = normalized_text.split()
        return set(tokens)

    @classmethod
    def jaccard_similarity(cls, text_a: str, text_b: str) -> float:
        """
        Calculate Jaccard token overlap between two strings.
        """
        norm_a = cls.normalize(text_a)
        norm_b = cls.normalize(text_b)
        set_a = cls.get_token_set(norm_a)
        set_b = cls.get_token_set(norm_b)

        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union
