import re
from typing import List


def clean_text(text: str) -> str:
    """
    텍스트 정제
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정제된 텍스트
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    간단한 키워드 추출 (빈도 기반)
    
    Args:
        text: 입력 텍스트
        top_k: 상위 k개 키워드
        
    Returns:
        키워드 리스트
    """
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'are', 'was'}
    words = [w for w in words if w not in stop_words]
    
    from collections import Counter
    word_freq = Counter(words)
    
    return [word for word, _ in word_freq.most_common(top_k)]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    텍스트 자르기
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 접미사
        
    Returns:
        자른 텍스트
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix