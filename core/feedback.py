import os
from typing import List, Dict
import openai
from pathlib import Path
import json


class FeedbackGenerator:
    """
    RAG 기반 디자인 피드백 생성기
    - Semantic & Recursive Chunking으로 지식 베이스 처리
    - Parent-Child Retriever로 문맥 검색
    - CoT/ReAct 프롬프팅으로 논리적 피드백 생성
    """
    
    def __init__(self, api_key: str = None, knowledge_base_dir: str = "data"):
        """
        Args:
            api_key: OpenAI API 키
            knowledge_base_dir: 디자인 가이드 지식 베이스 디렉토리
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.client = openai.OpenAI(api_key=self.api_key)
        
        self.knowledge_base = self._load_knowledge_base()
        print(f"[Feedback] Loaded {len(self.knowledge_base)} knowledge chunks")
    
    def _load_knowledge_base(self) -> List[Dict[str, str]]:
        """
        지식 베이스 로드 및 청킹
        Semantic Chunking: 의미 단위로 분할
        Recursive Chunking: 계층적 분할
        """
        chunks = []
        
        if not self.knowledge_base_dir.exists():
            print(f"[Feedback Warning] Knowledge base directory not found: {self.knowledge_base_dir}")
            return chunks
        
        for file_path in self.knowledge_base_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Semantic Chunking: 단락 단위로 분할
                    paragraphs = content.split('\n\n')
                    
                    for para in paragraphs:
                        para = para.strip()
                        if len(para) > 50:
                            chunks.append({
                                'source': file_path.name,
                                'content': para,
                                'type': 'semantic_chunk'
                            })
                    
                    # Recursive Chunking: 전체 문서도 포함 (Parent)
                    if len(content) > 100:
                        chunks.append({
                            'source': file_path.name,
                            'content': content,
                            'type': 'parent_chunk'
                        })
            except Exception as e:
                print(f"[Feedback Warning] Failed to load {file_path}: {e}")
        
        return chunks
    
    def _retrieve_relevant_knowledge(
        self, 
        query: str, 
        analysis: Dict,
        scores: Dict,
        top_k: int = 3
    ) -> List[str]:
        """
        Parent-Child Retriever 개념 적용
        쿼리에 가장 관련된 지식 검색 (키워드 기반)
        
        Note: 임베딩 API를 사용하지 않아 비용 절감
        """
        query_context = f"""
        Design Intent: {query}
        Current Issues: Low scores in {self._identify_weak_areas(scores)}
        Design Elements: {', '.join(analysis.get('segments', []))}
        """
        
        # 키워드 추출
        keywords = self._extract_keywords(query_context)
        
        # 관련성 점수 계산 (키워드 매칭)
        relevant_chunks = []
        
        for chunk in self.knowledge_base:
            content_lower = chunk['content'].lower()
            relevance_score = sum(1 for kw in keywords if kw in content_lower)
            
            if relevance_score > 0:
                relevant_chunks.append((relevance_score, chunk['content']))
        
        # 상위 k개 선택
        relevant_chunks.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [chunk[1] for chunk in relevant_chunks[:top_k]]
        
        # 폴백: 관련 청크가 없으면 첫 번째 청크 반환
        if not top_chunks and self.knowledge_base:
            top_chunks = [self.knowledge_base[0]['content']]
        
        return top_chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출"""
        keywords = [
            'color', 'layout', 'typography', 'contrast', 'hierarchy',
            'balance', 'whitespace', 'alignment', 'consistency', 'visual',
            'modern', 'minimal', 'bold', 'elegant', 'professional',
            'button', 'navigation', 'header', 'footer', 'text',
            'image', 'icon', 'form', 'card', 'grid'
        ]
        return [kw for kw in keywords if kw in text.lower()]
    
    def _identify_weak_areas(self, scores: Dict) -> str:
        """점수가 낮은 영역 식별"""
        detailed = scores.get('detailed_scores', {})
        weak_areas = []
        
        for area, score in detailed.items():
            if score < 70:
                weak_areas.append(area.replace('_', ' '))
        
        return ', '.join(weak_areas) if weak_areas else 'overall design alignment'
    
    def generate_feedback(
        self,
        text_intent: str,
        analysis_result: Dict,
        similarity_scores: Dict
    ) -> str:
        """
        CoT (Chain-of-Thought) + ReAct 방식으로 피드백 생성
        
        Args:
            text_intent: 사용자의 디자인 의도
            analysis_result: analyzer의 분석 결과
            similarity_scores: scorer의 점수 결과
            
        Returns:
            str: 전문적인 디자인 피드백
        """
        print("[Feedback] Generating professional feedback with CoT reasoning...")
        
        # RAG: 관련 지식 검색
        relevant_knowledge = self._retrieve_relevant_knowledge(
            text_intent, 
            analysis_result, 
            similarity_scores
        )
        
        # CoT + ReAct 프롬프트 구성
        prompt = self._build_cot_prompt(
            text_intent,
            analysis_result,
            similarity_scores,
            relevant_knowledge
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior UX/UI designer and design theorist. You provide expert feedback using design principles and theory."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            feedback = response.choices[0].message.content
            print("[Feedback] Feedback generated successfully")
            return feedback
            
        except openai.APIError as e:
            print(f"[Feedback Error] OpenAI API error: {e}")
            return self._get_fallback_feedback(text_intent, similarity_scores)
        except Exception as e:
            print(f"[Feedback Error] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_feedback(text_intent, similarity_scores)
    
    def _build_cot_prompt(
        self,
        intent: str,
        analysis: Dict,
        scores: Dict,
        knowledge: List[str]
    ) -> str:
        """CoT (Chain-of-Thought) 프롬프트 구성"""
        
        knowledge_text = "\n\n".join([f"Reference {i+1}:\n{k}" for i, k in enumerate(knowledge)])
        
        prompt = f"""
        Apply Chain-of-Thought reasoning to provide expert design feedback.
        
        **Step 1: Understand the Intent**
        User's Design Intent: "{intent}"
        
        **Step 2: Analyze Current State**
        Design Analysis:
        - Detected Elements: {', '.join(analysis.get('segments', []))}
        - Composition: {analysis.get('overall_composition', 'N/A')}
        - Visual Features: {len(analysis.get('visual_features', {}))} elements analyzed
        
        **Step 3: Evaluate Alignment**
        Similarity Scores:
        - Overall: {scores['overall_score']:.1f}/100
        - Visual Alignment: {scores['visual_alignment']:.1f}/100
        - Semantic Alignment: {scores['semantic_alignment']:.1f}/100
        Detailed:
        - Color Match: {scores['detailed_scores']['color_match']:.1f}/100
        - Layout Match: {scores['detailed_scores']['layout_match']:.1f}/100
        - Style Match: {scores['detailed_scores']['style_match']:.1f}/100
        
        **Step 4: Apply Design Theory (RAG Knowledge)**
        {knowledge_text}
        
        **Step 5: Generate Actionable Feedback**
        Using the above reasoning chain, provide:
        
        1. **Strengths**: What works well in the current design (2-3 points)
        
        2. **Areas for Improvement**: What doesn't align with the intent (3-4 points)
           - For each point, cite specific design principles
           - Explain WHY it's an issue based on theory
        
        3. **Concrete Recommendations**: Actionable steps to improve (4-5 specific actions)
           - Be specific (e.g., "Increase heading font size to 32px" not "make text bigger")
           - Prioritize by impact
        
        4. **Theory Application**: How these changes apply established design principles
        
        Use professional but accessible language. Be constructive and specific.
        """
        
        return prompt
    
    def _get_fallback_feedback(self, intent: str, scores: Dict) -> str:
        """폴백 피드백 생성"""
        overall_score = scores.get('overall_score', 50.0)
        
        if overall_score >= 70:
            assessment = "The design shows good alignment with your intent."
        elif overall_score >= 50:
            assessment = "The design partially aligns with your intent but needs refinement."
        else:
            assessment = "The design requires significant improvements to match your intent."
        
        weak_areas = self._identify_weak_areas(scores)
        
        return f"""
        Design Feedback for: "{intent}"
        
        Assessment: {assessment}
        
        Overall Score: {overall_score:.1f}/100
        
        Key Areas for Improvement:
        - {weak_areas}
        
        Recommendations:
        1. Review and refine the elements that scored below 70
        2. Ensure visual consistency across all design elements
        3. Consider user experience and accessibility principles
        4. Test the design with your target audience
        
        Note: Detailed feedback generation temporarily unavailable. Please try again.
        """


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    with open('data/design_principles.txt', 'w', encoding='utf-8') as f:
        f.write("""
        Color Theory in Design:
        Colors evoke emotions and guide user attention. Use a limited palette (3-5 colors) for consistency.
        
        Typography Hierarchy:
        Establish clear visual hierarchy through font size, weight, and spacing. Headings should be 2-3x body text size.
        
        Layout Principles:
        Follow the F-pattern for text-heavy content, Z-pattern for minimal designs. Use grid systems for alignment.
        """)
    
    generator = FeedbackGenerator()
    
    test_analysis = {
        'segments': ['Header', 'CTA Button', 'Footer'],
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
    
    feedback = generator.generate_feedback(
        "Modern SaaS landing page with blue theme",
        test_analysis,
        test_scores
    )
    
    print("\n=== FEEDBACK ===")
    print(feedback)