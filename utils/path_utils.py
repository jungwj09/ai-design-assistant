import os
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> str:
    """
    디렉토리가 없으면 생성하고 경로 반환
    
    Args:
        path: 디렉토리 경로
        
    Returns:
        str: 절대 경로
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())


def get_output_path(
    original_path: Union[str, Path],
    suffix: str,
    output_dir: Union[str, Path] = None
) -> str:
    """
    출력 파일 경로 생성
    
    Args:
        original_path: 원본 파일 경로
        suffix: 추가할 접미사 (예: '_improved', '_comparison')
        output_dir: 출력 디렉토리 (None이면 원본 디렉토리 사용)
        
    Returns:
        str: 출력 파일 경로
    """
    original = Path(original_path)
    name_without_ext = original.stem
    ext = original.suffix
    
    if output_dir:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
    else:
        output_dir = original.parent
    
    output_filename = f"{name_without_ext}{suffix}{ext}"
    return str(output_dir / output_filename)


def validate_image_path(path: Union[str, Path]) -> bool:
    """
    이미지 파일 경로 유효성 검사
    
    Args:
        path: 이미지 파일 경로
        
    Returns:
        bool: 유효하면 True
    """
    path = Path(path)
    
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        return False
    
    return True


def get_file_size_mb(path: Union[str, Path]) -> float:
    """
    파일 크기를 MB 단위로 반환
    
    Args:
        path: 파일 경로
        
    Returns:
        float: 파일 크기 (MB)
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    
    size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)


if __name__ == "__main__":
    # 테스트
    test_dir = ensure_dir('test_samples')
    print(f"Created/verified directory: {test_dir}")
    
    test_output = get_output_path(
        'test_samples/design.png',
        '_improved',
        'output'
    )
    print(f"Output path: {test_output}")
    
    # 유효성 검사 테스트
    from PIL import Image
    test_img = Image.new('RGB', (100, 100), 'white')
    test_img.save('test_samples/test.png')
    
    print(f"Valid image: {validate_image_path('test_samples/test.png')}")
    print(f"Invalid path: {validate_image_path('nonexistent.png')}")
    
    size = get_file_size_mb('test_samples/test.png')
    print(f"File size: {size:.4f} MB")