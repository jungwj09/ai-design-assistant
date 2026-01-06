import os
from PIL import Image
from typing import Dict, Optional
import openai
import base64
from io import BytesIO
import requests


class DesignGenerator:
    """
    Image-to-Image 기반 디자인 개선안 생성
    - Diffusion Model 개념 적용
    - 원본 이미지와 텍스트 프롬프트를 기반으로 개선된 디자인 생성
    """
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: OpenAI API 키 (DALL-E 3 사용)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def generate_improved_design(
        self,
        original_image_path: str,
        text_intent: str,
        feedback: str,
        analysis: Dict,
        scores: Dict
    ) -> str:
        """
        개선된 디자인 이미지 생성
        
        Args:
            original_image_path: 원본 이미지 경로
            text_intent: 사용자의 디자인 의도
            feedback: 생성된 피드백
            analysis: 이미지 분석 결과
            scores: 유사도 점수
            
        Returns:
            str: 생성된 이미지 저장 경로
        """
        print("[Generator] Generating improved design...")
        
        # Image-to-Image 프롬프트 구성
        generation_prompt = self._build_generation_prompt(
            text_intent,
            feedback,
            analysis,
            scores
        )
        
        print(f"[Generator] Prompt: {generation_prompt[:100]}...")
        
        # DALL-E 3로 이미지 생성
        # 참고: DALL-E 3는 Image-to-Image를 직접 지원하지 않으므로 텍스트 프롬프트로 개선 방향을 명확히 지시
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=generation_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            output_path = self._download_and_save_image(
                image_url,
                original_image_path
            )
            
            print(f"[Generator] Image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[Generator Error] Generation failed: {e}")
            
            # 폴백: 원본 이미지에 텍스트 오버레이
            return self._create_fallback_image(
                original_image_path,
                "Improved design generation unavailable"
            )
    
    def _build_generation_prompt(
        self,
        intent: str,
        feedback: str,
        analysis: Dict,
        scores: Dict
    ) -> str:
        """
        Diffusion Model을 위한 상세 프롬프트 생성
        Image-to-Image 개념을 텍스트로 구현
        """
        # 약점 영역 파악
        weak_areas = []
        detailed = scores.get('detailed_scores', {})
        
        if detailed.get('color_match', 100) < 70:
            weak_areas.append("color scheme")
        if detailed.get('layout_match', 100) < 70:
            weak_areas.append("layout structure")
        if detailed.get('style_match', 100) < 70:
            weak_areas.append("visual style")
        
        # 개선 방향 추출 (피드백에서)
        improvements = self._extract_improvements_from_feedback(feedback)
        
        # 프롬프트 구성
        prompt = f"""
        Create a professional UI/UX design that fulfills this intent: {intent}
        
        Design Requirements:
        - Style: Modern, clean, and professional
        - Elements to include: {', '.join(analysis.get('segments', ['header', 'main content', 'footer']))}
        - Layout: {analysis.get('overall_composition', 'Well-balanced and structured')}
        
        Key Improvements Needed:
        {improvements}
        
        Focus Areas (from analysis):
        {', '.join(weak_areas) if weak_areas else 'Overall refinement'}
        
        Technical Specifications:
        - High contrast for readability
        - Clear visual hierarchy
        - Proper whitespace usage
        - Consistent styling throughout
        - Professional color palette
        - Modern typography
        
        Output: A complete, polished UI design that directly addresses the user's intent.
        """
        
        # 프롬프트 길이 제한 (DALL-E 3: 4000자)
        if len(prompt) > 3900:
            prompt = prompt[:3900] + "..."
        
        return prompt.strip()
    
    def _extract_improvements_from_feedback(self, feedback: str) -> str:
        """피드백에서 핵심 개선사항 추출"""
        # 간단한 키워드 기반 추출
        lines = feedback.split('\n')
        improvements = []
        
        # "Recommendation", "Improve", "Change" 등의 키워드가 있는 줄 추출
        keywords = ['recommend', 'improve', 'change', 'increase', 'decrease', 
                   'add', 'remove', 'adjust', 'enhance', 'use']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                # 불필요한 기호 제거
                clean_line = line.strip('- •*#').strip()
                if len(clean_line) > 20 and len(clean_line) < 200:
                    improvements.append(clean_line)
        
        # 최대 5개까지만
        improvements = improvements[:5]
        
        return '\n'.join([f"- {imp}" for imp in improvements]) if improvements else "- Enhance overall design quality and alignment with intent"
    
    def _download_and_save_image(self, image_url: str, original_path: str) -> str:
        """생성된 이미지 다운로드 및 저장"""
        # 이미지 다운로드
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # 저장 경로 생성
        original_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(original_name)[0]
        output_dir = os.path.dirname(original_path) or 'test_samples'
        output_path = os.path.join(output_dir, f"{name_without_ext}_improved.png")
        
        # 저장
        img.save(output_path)
        return output_path
    
    def _create_fallback_image(self, original_path: str, message: str) -> str:
        """폴백: 원본 이미지 복사 또는 간단한 개선"""
        try:
            img = Image.open(original_path)
            
            # 간단한 처리: 약간 밝게
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img_enhanced = enhancer.enhance(1.1)
            
            # 저장
            original_name = os.path.basename(original_path)
            name_without_ext = os.path.splitext(original_name)[0]
            output_dir = os.path.dirname(original_path) or 'test_samples'
            output_path = os.path.join(output_dir, f"{name_without_ext}_improved.png")
            
            img_enhanced.save(output_path)
            print(f"[Generator] Fallback image created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[Generator Error] Fallback failed: {e}")
            return original_path
    
    def create_side_by_side_comparison(
        self,
        original_path: str,
        improved_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        원본과 개선안을 나란히 비교하는 이미지 생성
        
        Args:
            original_path: 원본 이미지 경로
            improved_path: 개선된 이미지 경로
            output_path: 출력 경로 (선택)
            
        Returns:
            str: 비교 이미지 경로
        """
        try:
            # 이미지 로드
            original = Image.open(original_path)
            improved = Image.open(improved_path)
            
            # 크기 조정 (같은 높이로)
            target_height = 600
            
            original_ratio = original.width / original.height
            improved_ratio = improved.width / improved.height
            
            original_resized = original.resize(
                (int(target_height * original_ratio), target_height),
                Image.Resampling.LANCZOS
            )
            improved_resized = improved.resize(
                (int(target_height * improved_ratio), target_height),
                Image.Resampling.LANCZOS
            )
            
            # 새 캔버스 생성
            total_width = original_resized.width + improved_resized.width + 60  # 여백 포함
            canvas = Image.new('RGB', (total_width, target_height + 100), 'white')
            
            # 이미지 붙이기
            canvas.paste(original_resized, (20, 50))
            canvas.paste(improved_resized, (original_resized.width + 40, 50))
            
            # 텍스트 추가 (PIL의 기본 폰트 사용)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas)
            
            draw.text((20, 20), "Original Design", fill='black')
            draw.text((original_resized.width + 40, 20), "Improved Design", fill='black')
            
            # 저장
            if not output_path:
                original_name = os.path.basename(original_path)
                name_without_ext = os.path.splitext(original_name)[0]
                output_dir = os.path.dirname(original_path) or 'test_samples'
                output_path = os.path.join(output_dir, f"{name_without_ext}_comparison.png")
            
            canvas.save(output_path)
            print(f"[Generator] Comparison image saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[Generator Error] Comparison creation failed: {e}")
            return improved_path


if __name__ == "__main__":
    # 테스트 코드
    generator = DesignGenerator()
    
    # 테스트 이미지 생성
    os.makedirs('test_samples', exist_ok=True)
    test_img = Image.new('RGB', (800, 600), color='lightgray')
    test_path = 'test_samples/test_original.png'
    test_img.save(test_path)
    
    # 테스트 데이터
    test_analysis = {
        'segments': ['Header', 'Hero Section', 'CTA Button', 'Footer'],
        'overall_composition': 'Centered layout with good balance'
    }
    
    test_scores = {
        'overall_score': 65.0,
        'visual_alignment': 70.0,
        'semantic_alignment': 60.0,
        'detailed_scores': {
            'color_match': 55.0,
            'layout_match': 75.0,
            'style_match': 65.0
        }
    }
    
    test_feedback = """
    Strengths:
    - Good layout structure
    
    Areas for Improvement:
    - Color scheme lacks vibrancy
    - Typography hierarchy needs improvement
    
    Recommendations:
    - Use a more vibrant blue (#2563eb)
    - Increase heading size to 48px
    - Add more whitespace between sections
    """
    
    # 개선안 생성
    improved_path = generator.generate_improved_design(
        test_path,
        "Modern SaaS landing page with blue theme",
        test_feedback,
        test_analysis,
        test_scores
    )
    
    print(f"\nImproved design generated: {improved_path}")
    
    # 비교 이미지 생성
    if os.path.exists(improved_path):
        comparison_path = generator.create_side_by_side_comparison(
            test_path,
            improved_path
        )
        print(f"Comparison image: {comparison_path}")