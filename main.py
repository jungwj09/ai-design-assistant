import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.analyzer import DesignAnalyzer
from core.scorer import SimilarityScorer
from core.feedback import FeedbackGenerator
from core.generator import DesignGenerator


class DesignIntentAI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required."
                "Set OPENAI_API_KEY environment variable or pass it as argument."
            )
        
        print("=" * 60)
        print("  🎨 DesignIntent AI - Initializing Pipeline")
        print("=" * 60)
        
        # 각 모듈 초기화
        self.analyzer = DesignAnalyzer(api_key=self.api_key)
        self.scorer = SimilarityScorer(api_key=self.api_key, use_local_clip=False)
        self.feedback_generator = FeedbackGenerator(api_key=self.api_key)
        self.design_generator = DesignGenerator(api_key=self.api_key)
        
        print("✓ All modules initialized successfully\n")
    
    def run_pipeline(
        self,
        image_path: str,
        text_intent: str,
        generate_improvement: bool = True,
        output_dir: str = "output"
    ) -> dict:
        """
        전체 파이프라인 실행
        
        Args:
            image_path: 분석할 이미지 경로
            text_intent: 사용자의 디자인 의도
            generate_improvement: 개선안 생성 여부
            output_dir: 출력 디렉토리
            
        Returns:
            dict: 전체 결과를 포함하는 딕셔너리
        """
        print("\n" + "=" * 60)
        print("  🚀 Starting DesignIntent AI Pipeline")
        print("=" * 60)
        print(f"📁 Image: {image_path}")
        print(f"💭 Intent: {text_intent}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'image_path': image_path,
            'text_intent': text_intent,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n[Step 1/5] 🔍 Analyzing Image...")
        print("-" * 60)
        analysis_result = self.analyzer.analyze_image(image_path)
        results['analysis'] = analysis_result
        
        print(f"  ✓ Detected {len(analysis_result['segments'])} design elements")
        print(f"  ✓ Extracted visual features")
        print(f"  ✓ Analyzed composition")
        
        print("\n[Step 2/5] 📊 Calculating Similarity Scores...")
        print("-" * 60)
        similarity_scores = self.scorer.calculate_similarity(
            image_path,
            text_intent,
            analysis_result
        )
        results['scores'] = similarity_scores
        
        score_report = self.scorer.generate_score_report(similarity_scores)
        print(score_report)
        
        print("\n[Step 3/5] 💡 Generating Expert Feedback...")
        print("-" * 60)
        feedback = self.feedback_generator.generate_feedback(
            text_intent,
            analysis_result,
            similarity_scores
        )
        results['feedback'] = feedback
        
        print("  ✓ Feedback generated successfully")
        print("\n" + "─" * 60)
        print(feedback)
        print("─" * 60)
        
        if generate_improvement:
            print("\n[Step 4/5] 🎨 Generating Improved Design...")
            print("-" * 60)
            
            improved_image_path = self.design_generator.generate_improved_design(
                image_path,
                text_intent,
                feedback,
                analysis_result,
                similarity_scores
            )
            results['improved_image_path'] = improved_image_path
            
            print(f"  ✓ Improved design saved to: {improved_image_path}")
            
            print("\n[Step 5/5] 📸 Creating Comparison Image...")
            print("-" * 60)
            
            comparison_path = self.design_generator.create_side_by_side_comparison(
                image_path,
                improved_image_path
            )
            results['comparison_image_path'] = comparison_path
            
            print(f"  ✓ Comparison saved to: {comparison_path}")
        else:
            print("\n[Step 4/5] ⏭️  Skipping image generation (as requested)")
            results['improved_image_path'] = None
            results['comparison_image_path'] = None
        
        self._save_results(results, output_dir)
        
        print("\n" + "=" * 60)
        print("  ✨ Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"\n📊 Overall Score: {similarity_scores['overall_score']:.1f}/100")
        print(f"📁 Results saved to: {output_dir}/")
        print("\nFiles generated:")
        print(f"  • Analysis report: {output_dir}/analysis_report.txt")
        if generate_improvement:
            print(f"  • Improved design: {results.get('improved_image_path', 'N/A')}")
            print(f"  • Comparison image: {results.get('comparison_image_path', 'N/A')}")
        print("\n" + "=" * 60 + "\n")
        
        return results
    
    def _save_results(self, results: dict, output_dir: str):
        """결과를 텍스트 파일로 저장"""
        report_path = os.path.join(output_dir, "analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  DesignIntent AI - Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Image: {results['image_path']}\n")
            f.write(f"Intent: {results['text_intent']}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("SIMILARITY SCORES\n")
            f.write("-" * 60 + "\n")
            scores = results['scores']
            f.write(f"Overall Score: {scores['overall_score']:.2f}/100\n")
            f.write(f"Visual Alignment: {scores['visual_alignment']:.2f}/100\n")
            f.write(f"Semantic Alignment: {scores['semantic_alignment']:.2f}/100\n\n")
            f.write("Detailed Breakdown:\n")
            for key, value in scores['detailed_scores'].items():
                f.write(f"  • {key.replace('_', ' ').title()}: {value:.2f}/100\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("DESIGN ANALYSIS\n")
            f.write("-" * 60 + "\n")
            analysis = results['analysis']
            f.write(f"Detected Elements: {', '.join(analysis['segments'])}\n\n")
            f.write(f"Composition: {analysis['overall_composition']}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("EXPERT FEEDBACK\n")
            f.write("-" * 60 + "\n")
            f.write(results['feedback'])
            f.write("\n\n")
            
            if results.get('improved_image_path'):
                f.write("-" * 60 + "\n")
                f.write("GENERATED OUTPUTS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Improved Design: {results['improved_image_path']}\n")
                f.write(f"Comparison Image: {results.get('comparison_image_path', 'N/A')}\n")
        
        print(f"  ✓ Report saved to: {report_path}")


def main():
    # CLI 진입점
    parser = argparse.ArgumentParser(
        description="DesignIntent AI - Design Analysis and Improvement System"
    )
    parser.add_argument(
        'image',
        type=str,
        help='Path to the design image to analyze'
    )
    parser.add_argument(
        'intent',
        type=str,
        help='Your design intent (text description)'
    )
    parser.add_argument(
        '--no-generation',
        action='store_true',
        help='Skip image generation (analysis and feedback only)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env variable)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        # 파이프라인 초기화 및 실행
        pipeline = DesignIntentAI(api_key=args.api_key)
        
        results = pipeline.run_pipeline(
            image_path=args.image,
            text_intent=args.intent,
            generate_improvement=not args.no_generation,
            output_dir=args.output_dir
        )
        
        print("🎉 Success! Check the output directory for results.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print("  DesignIntent AI - Example Usage")
        print("=" * 60 + "\n")
        
        from PIL import Image, ImageDraw
        
        os.makedirs('test_samples', exist_ok=True)
        
        img = Image.new('RGB', (1200, 800), color='#f0f9ff')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([0, 0, 1200, 100], fill='#1e40af')
        
        draw.rectangle([100, 150, 1100, 650], fill='white')
        draw.rectangle([150, 200, 1050, 300], fill='#3b82f6')
        
        test_image_path = 'test_samples/example_design.png'
        img.save(test_image_path)
        
        print(f"Created test image: {test_image_path}\n")
        
        pipeline = DesignIntentAI()
        
        results = pipeline.run_pipeline(
            image_path=test_image_path,
            text_intent="A modern, professional SaaS landing page with blue color scheme, clear call-to-action, and minimalist design",
            generate_improvement=True,
            output_dir='output'
        )
        
    else:
        main()