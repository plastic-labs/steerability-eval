import hashlib

def generate_short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]