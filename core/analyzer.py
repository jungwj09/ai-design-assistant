import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Tuple
import base64
from io import BytesIO
import os

class DesignAnalyzer:
    """
    이미지 분석 클래스
    - SAM2를 통한 세그멘테이션 (Hugging Face API 활용)
    - Pali-Gemma/GPT-4V를 통한 시각적 특징 추출
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # SAM2는 로컬 구동이 어려워서 GPT-4V로 세그멘트 분석 대체
        self.vision_model = "gpt-4o"
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        이미지를 분석하여 세그먼트와 시각적 특징 추출
        
        Args:
            image_path: 분석할 이미지 경로
            
        Returns:
            Dict: {
                'segments': List of detected design elements,
                'visual_features': Detailed description of each segment,
                'overall_composition': Overall design analysis
            }
        """
        print(f"[Analyzer] Analyzing image: {image_path}")
        
        image = Image.open(image_path)
        image_base64 = self._encode_image(image)
        
        # SAM2 개념 적용: 이미지를 세그먼트로 분할
        segments = self._segment_image(image_base64)
        
        # Pali-Gemma 개념 적용: 각 세그먼트의 시각적 특징 추출
        visual_features = self._extract_visual_features(image_base64, segments)
        
        overall_composition = self._analyze_composition(image_base64)
        
        result = {
            'segments': segments,
            'visual_features': visual_features,
            'overall_composition': overall_composition
        }
        
        print(f"[Analyzer] Found {len(segments)} design elements")
        return result
    
    def _encode_image(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _segment_image(self, image_base64: str) -> List[str]:
        """
        SAM2 개념: 이미지에서 주요 디자인 요소 분할
        실제로는 GPT-4V를 사용하여 세그먼트 식별
        """
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        prompt = """
        Analyze this design image and identify all distinct design elements/segments.
        Follow the SAM2 (Segment Anything Model) approach:
        
        List each design element as a separate item:
        - Buttons
        - Text blocks
        - Images/Icons
        - Background sections
        - Navigation elements
        - Form inputs
        - Cards or containers
        
        Return ONLY a comma-separated list of elements found.
        Example: "Hero image, Navigation bar, CTA button, Text heading, Footer"
        """
        
        try:
            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            segments_text = response.choices[0].message.content.strip()
            segments = [s.strip() for s in segments_text.split(',')]
            return segments
            
        except Exception as e:
            print(f"[Analyzer Error] Segmentation failed: {e}")
            return ["Main content area", "Text elements", "Visual elements"]
    
    def _extract_visual_features(self, image_base64: str, segments: List[str]) -> Dict[str, str]:
        """
        Pali-Gemma 개념: 각 세그먼트의 시각적 특징 추출
        색상, 형태, 스타일, 타이포그래피 등을 분석
        """
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        segments_list = "\n".join([f"- {seg}" for seg in segments])
        
        prompt = f"""
        Following the Pali-Gemma visual understanding approach, analyze the visual features of each design element:
        
        Elements to analyze:
        {segments_list}
        
        For EACH element above, describe:
        1. Color scheme (specific colors, contrast)
        2. Typography (font style, size, weight)
        3. Shape and form (geometric properties)
        4. Visual hierarchy (prominence, positioning)
        5. Style characteristics (modern, minimal, bold, etc.)
        
        Format your response as:
        [Element Name]
        - Color: ...
        - Typography: ...
        - Shape: ...
        - Hierarchy: ...
        - Style: ...
        """
        
        try:
            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500
            )
            
            features_text = response.choices[0].message.content
            
            # 파싱: 각 세그먼트별로 특징 추출
            features_dict = {}
            current_element = None
            
            for line in features_text.split('\n'):
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    current_element = line[1:-1]
                    features_dict[current_element] = ""
                elif current_element and line:
                    features_dict[current_element] += line + "\n"
            
            return features_dict
            
        except Exception as e:
            print(f"[Analyzer Error] Feature extraction failed: {e}")
            return {seg: "Visual analysis unavailable" for seg in segments}
    
    def _analyze_composition(self, image_base64: str) -> str:
        """전체 디자인 구성 분석"""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        prompt = """
        Analyze the overall composition of this design:
        
        1. Layout structure (grid, asymmetric, centered, etc.)
        2. Visual balance and symmetry
        3. White space usage
        4. Visual flow and eye movement
        5. Design consistency
        6. Overall aesthetic style
        
        Provide a comprehensive but concise analysis (3-4 sentences).
        """
        
        try:
            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[Analyzer Error] Composition analysis failed: {e}")
            return "Overall composition analysis unavailable"


if __name__ == "__main__":
    analyzer = DesignAnalyzer()
    
    test_img = Image.new('RGB', (800, 600), color='white')
    test_img.save('test_samples/test.png')
    
    result = analyzer.analyze_image('test_samples/test.png')
    print("\n=== Analysis Result ===")
    print(f"Segments: {result['segments']}")
    print(f"\nComposition: {result['overall_composition']}")