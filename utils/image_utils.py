import base64
from io import BytesIO
from PIL import Image
from typing import Tuple


def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    이미지를 최대 크기로 리사이즈 (비율 유지)
    
    Args:
        image: PIL Image 객체
        max_size: 최대 크기 (width, height)
        
    Returns:
        리사이즈된 Image 객체
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    이미지를 base64 문자열로 인코딩
    
    Args:
        image: PIL Image 객체
        format: 이미지 포맷 (PNG, JPEG 등)
        
    Returns:
        base64 인코딩된 문자열
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def create_thumbnail(image_path: str, output_path: str, size: Tuple[int, int] = (300, 300)):
    """
    썸네일 생성
    
    Args:
        image_path: 원본 이미지 경로
        output_path: 출력 경로
        size: 썸네일 크기
    """
    img = Image.open(image_path)
    img.thumbnail(size, Image.Resampling.LANCZOS)
    img.save(output_path)