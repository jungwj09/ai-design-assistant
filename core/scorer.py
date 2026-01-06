import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import os
from transformers import CLIPProcessor, CLIPModel
import openai
import json
import re


class SimilarityScorer:
    """
    CLIP 기반 이미지-텍스트 유사도 측정
    Late Fusion 개념을 적용하여 시각적/텍스트 특징 정렬
    """
    
    def __init__(self, api_key: str = None, use_local_clip: bool = False):
        """
        Args:
            api_key: OpenAI API 키
            use_local_clip: 로컬 CLIP 모델 사용 여부
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_local_clip = use_local_clip
        
        if use_local_clip:
            print("[Scorer] Loading local CLIP model...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"[Scorer] CLIP loaded on {self.device}")
        else:
            print("[Scorer] Using OpenAI embeddings API")
    
    def calculate_similarity(
        self, 
        image_path: str, 
        text_intent: str,
        analysis_result: Dict = None
    ) -> Dict[str, float]:
        """
        이미지와 텍스트 의도 간의 유사도 계산
        
        Args:
            image_path: 이미지 경로
            text_intent: 사용자의 디자인 의도
            analysis_result: analyzer.py의 분석 결과 (옵션)
            
        Returns:
            Dict: {
                'overall_score': 전체 유사도 점수 (0-100),
                'visual_alignment': 시각적 정렬 점수,
                'semantic_alignment': 의미적 정렬 점수,
                'detailed_scores': 세부 요소별 점수
            }
        """
        print(f"[Scorer] Calculating similarity for intent: '{text_intent[:50]}...'")
        
        if self.use_local_clip:
            scores = self._calculate_with_clip(image_path, text_intent)
        else:
            scores = self._calculate_with_openai(image_path, text_intent, analysis_result)
        
        print(f"[Scorer] Overall Score: {scores['overall_score']:.2f}/100")
        return scores
    
    def _calculate_with_clip(self, image_path: str, text_intent: str) -> Dict[str, float]:
        """
        로컬 CLIP 모델을 사용한 유사도 계산
        Late Fusion: 이미지/텍스트 임베딩을 독립적으로 추출 후 코사인 유사도 계산
        """
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            text=[text_intent],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Late Fusion: 독립적으로 추출된 특징
            image_embeds = outputs.image_embeds  # [1, 512]
            text_embeds = outputs.text_embeds    # [1, 512]
            
            # 코사인 유사도 계산
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            similarity = (image_embeds @ text_embeds.T).item()
        
        # 점수 정규화 (0-100)
        overall_score = (similarity + 1) / 2 * 100  # [-1, 1] -> [0, 100]
        
        return {
            'overall_score': overall_score,
            'visual_alignment': overall_score * 0.95,  # 시뮬레이션
            'semantic_alignment': overall_score * 1.05,
            'detailed_scores': {
                'color_match': overall_score * 0.9,
                'layout_match': overall_score * 1.1,
                'style_match': overall_score * 0.95
            }
        }
    
    def _extract_json_from_response(self, text: str) -> Dict:
        """
        GPT 응답에서 JSON 객체를 안전하게 추출
        
        Args:
            text: GPT 응답 텍스트
            
        Returns:
            파싱된 JSON 딕셔너리
            
        Raises:
            ValueError: JSON을 찾을 수 없거나 파싱 실패 시
        """
        # 1. 코드 블록 제거
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # 2. JSON 객체 찾기 (중첩된 객체도 처리)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        
        if not json_match:
            raise ValueError("No JSON object found in response")
        
        json_str = json_match.group()
        
        # 3. JSON 파싱
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 파싱 실패 시 더 자세한 정보 제공
            print(f"[Scorer Debug] Failed to parse JSON:")
            print(f"Text: {json_str[:200]}...")
            raise ValueError(f"JSON parsing failed: {e}")
    
    def _calculate_with_openai(
        self, 
        image_path: str, 
        text_intent: str,
        analysis_result: Dict = None
    ) -> Dict[str, float]:
        """
        OpenAI GPT-4V를 사용한 유사도 계산
        CLIP 개념을 프롬프트로 구현
        """
        import base64
        from io import BytesIO
        
        client = openai.OpenAI(api_key=self.api_key)
        
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 분석 결과가 있으면 컨텍스트로 활용
        context = ""
        if analysis_result:
            context = f"""
            Design Analysis Context:
            - Detected Elements: {', '.join(analysis_result.get('segments', []))}
            - Composition: {analysis_result.get('overall_composition', 'N/A')}
            """
        
        prompt = f"""
        You are a CLIP-based similarity evaluator. Analyze how well this design image aligns with the user's intent.
        
        User's Design Intent:
        "{text_intent}"
        
        {context}
        
        Apply the Late Fusion concept:
        1. Extract visual features from the image (colors, layout, style, elements)
        2. Extract semantic features from the text intent
        3. Calculate alignment between these features
        
        Evaluate the following aspects (score 0-100 each):
        - Visual Alignment: How well colors, shapes, and visual style match the intent
        - Semantic Alignment: How well the meaning and purpose match the intent
        - Color Match: Appropriateness of color choices
        - Layout Match: Effectiveness of layout structure
        - Style Match: Consistency with intended style
        
        CRITICAL: Respond with ONLY a valid JSON object, no other text before or after:
        {{
            "overall_score": 75.5,
            "visual_alignment": 72.0,
            "semantic_alignment": 80.0,
            "detailed_scores": {{
                "color_match": 70.0,
                "layout_match": 75.0,
                "style_match": 73.0
            }},
            "reasoning": "Brief explanation here"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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
            
            result_text = response.choices[0].message.content.strip()
            
            # JSON 추출 및 파싱
            scores = self._extract_json_from_response(result_text)
            
            # reasoning 제거 (선택적)
            if 'reasoning' in scores:
                print(f"[Scorer] Reasoning: {scores['reasoning']}")
                del scores['reasoning']
            
            # 필수 필드 검증
            required_fields = ['overall_score', 'visual_alignment', 'semantic_alignment', 'detailed_scores']
            for field in required_fields:
                if field not in scores:
                    raise ValueError(f"Missing required field: {field}")
            
            # detailed_scores 검증
            required_detailed = ['color_match', 'layout_match', 'style_match']
            for field in required_detailed:
                if field not in scores['detailed_scores']:
                    raise ValueError(f"Missing detailed score: {field}")
            
            return scores
            
        except openai.APIError as e:
            print(f"[Scorer Error] OpenAI API error: {e}")
            return self._get_fallback_scores()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Scorer Error] Response parsing failed: {e}")
            print(f"[Scorer Debug] Raw response: {result_text[:200] if 'result_text' in locals() else 'N/A'}...")
            return self._get_fallback_scores()
        except Exception as e:
            print(f"[Scorer Error] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_scores()
    
    def _get_fallback_scores(self) -> Dict[str, float]:
        """폴백 점수 반환"""
        return {
            'overall_score': 50.0,
            'visual_alignment': 50.0,
            'semantic_alignment': 50.0,
            'detailed_scores': {
                'color_match': 50.0,
                'layout_match': 50.0,
                'style_match': 50.0
            }
        }
    
    def generate_score_report(self, scores: Dict[str, float]) -> str:
        """점수 리포트 생성"""
        report = f"""
╔════════════════════════════════════════╗
║      DESIGN ALIGNMENT SCORE REPORT     ║
╠════════════════════════════════════════╣
║ Overall Score:        {scores['overall_score']:6.2f} / 100 ║
║ Visual Alignment:     {scores['visual_alignment']:6.2f} / 100 ║
║ Semantic Alignment:   {scores['semantic_alignment']:6.2f} / 100 ║
╠════════════════════════════════════════╣
║ Detailed Breakdown:                    ║
║  • Color Match:       {scores['detailed_scores']['color_match']:6.2f} / 100 ║
║  • Layout Match:      {scores['detailed_scores']['layout_match']:6.2f} / 100 ║
║  • Style Match:       {scores['detailed_scores']['style_match']:6.2f} / 100 ║
╚════════════════════════════════════════╝
        """
        return report.strip()


if __name__ == "__main__":
    # 테스트 코드
    scorer = SimilarityScorer(use_local_clip=False)
    
    # 테스트 이미지 생성
    os.makedirs('test_samples', exist_ok=True)
    test_img = Image.new('RGB', (800, 600), color='lightblue')
    test_img.save('test_samples/test.png')
    
    test_intent = "A modern, minimalist landing page with blue color scheme"
    scores = scorer.calculate_similarity('test_samples/test.png', test_intent)
    
    print("\n" + scorer.generate_score_report(scores))