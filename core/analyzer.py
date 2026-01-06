import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Tuple
import base64
from io import BytesIO
import os
import json
import re


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
    
    def _extract_json_array(self, text: str) -> List[str]:
        """
        텍스트에서 JSON 배열을 안전하게 추출
        
        Args:
            text: GPT 응답 텍스트
            
        Returns:
            추출된 문자열 리스트
        """
        # 코드 블록 제거
        text = re.sub(r'```json\s*|\s*```', '', text)
        
        # JSON 배열 찾기
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"[Analyzer Warning] JSON decode failed: {e}")
        
        # 폴백: 쉼표로 구분된 텍스트 파싱
        # "Element1, Element2, Element3" 형식 처리
        lines = text.split('\n')
        elements = []
        for line in lines:
            # "- Element" 또는 "• Element" 형식
            if line.strip().startswith(('-', '•', '*')):
                element = line.strip().lstrip('-•*').strip()
                if element:
                    elements.append(element)
            # "1. Element" 형식
            elif re.match(r'^\d+\.\s+', line.strip()):
                element = re.sub(r'^\d+\.\s+', '', line.strip())
                if element:
                    elements.append(element)
        
        if elements:
            return elements
        
        # 최종 폴백: 쉼표로 분할
        return [s.strip() for s in text.split(',') if s.strip()]
    
    def _segment_image(self, image_base64: str) -> List[str]:
        """
        SAM2 개념: 이미지에서 주요 디자인 요소 분할
        실제로는 GPT-4V를 사용하여 세그먼트 식별
        """
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        prompt = """
        Analyze this design image and identify all distinct design elements/segments.
        Follow the SAM2 (Segment Anything Model) approach.
        
        List each design element as a separate item. Look for:
        - Buttons and CTAs
        - Text blocks (headings, paragraphs)
        - Images/Icons
        - Background sections
        - Navigation elements
        - Form inputs
        - Cards or containers
        - Headers/Footers
        
        Return ONLY a JSON array of strings, with NO additional text or explanation:
        ["Element 1", "Element 2", "Element 3", ...]
        
        Example: ["Hero image", "Navigation bar", "CTA button", "Text heading", "Footer"]
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
                max_tokens=500,
                temperature=0.3
            )
            
            segments_text = response.choices[0].message.content.strip()
            segments = self._extract_json_array(segments_text)
            
            if not segments:
                print("[Analyzer Warning] No segments extracted, using defaults")
                return ["Main content area", "Text elements", "Visual elements"]
            
            return segments
            
        except openai.APIError as e:
            print(f"[Analyzer Error] OpenAI API error during segmentation: {e}")
            return ["Main content area", "Text elements", "Visual elements"]
        except Exception as e:
            print(f"[Analyzer Error] Unexpected error during segmentation: {e}")
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
        
        Format your response EXACTLY as follows (use this structure):
        [Element Name]
        - Color: ...
        - Typography: ...
        - Shape: ...
        - Hierarchy: ...
        - Style: ...
        
        [Next Element Name]
        - Color: ...
        ...
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
                max_tokens=1500,
                temperature=0.3
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
            
            # 추출된 특징이 없으면 세그먼트에 기본값 할당
            if not features_dict:
                features_dict = {seg: "Visual features detected" for seg in segments}
            
            return features_dict
            
        except openai.APIError as e:
            print(f"[Analyzer Error] OpenAI API error during feature extraction: {e}")
            return {seg: "Visual analysis unavailable" for seg in segments}
        except Exception as e:
            print(f"[Analyzer Error] Unexpected error during feature extraction: {e}")
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
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.APIError as e:
            print(f"[Analyzer Error] OpenAI API error during composition analysis: {e}")
            return "Overall composition analysis unavailable"
        except Exception as e:
            print(f"[Analyzer Error] Unexpected error during composition analysis: {e}")
            return "Overall composition analysis unavailable"


if __name__ == "__main__":
    analyzer = DesignAnalyzer()
    
    test_img = Image.new('RGB', (800, 600), color='white')
    os.makedirs('test_samples', exist_ok=True)
    test_img.save('test_samples/test.png')
    
    result = analyzer.analyze_image('test_samples/test.png')
    print("\n=== Analysis Result ===")
    print(f"Segments: {result['segments']}")
    print(f"\nComposition: {result['overall_composition']}")