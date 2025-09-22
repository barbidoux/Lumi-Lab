#!/usr/bin/env python3
"""
Performance assessment tool that compares evaluation results against expected benchmarks.
Provides clear feedback on model quality and suggestions for improvement.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_evaluation_config(config_path: str = "evaluation/evaluation_config.json") -> Dict:
    """Load the evaluation configuration with expected benchmarks."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_model_size(num_parameters: int) -> str:
    """Detect model size based on parameter count."""
    if num_parameters < 20_000_000:  # < 20M
        return "tiny"
    elif num_parameters < 80_000_000:  # < 80M  
        return "small"
    else:
        return "base"


def assess_metric(value: float, thresholds: Dict, metric_name: str) -> Tuple[str, str, str]:
    """Assess a single metric against thresholds."""

    # Parse threshold strings (handle < > operators)
    def parse_threshold(threshold_str: str) -> float:
        return float(threshold_str.replace('<', '').replace('>', ''))

    # Handle nested structure in performance_indicators
    if isinstance(thresholds['good'], dict):
        # Find the matching metric key with better matching
        metric_key = None

        # Direct mapping for common metrics
        metric_mapping = {
            "perplexity": "perplexity",
            "boolq": "boolq_accuracy",
            "smoke": "smoke_quality",
            "smoke_test": "smoke_quality",
            "accuracy": "boolq_accuracy"
        }

        # Try direct mapping first
        for pattern, key in metric_mapping.items():
            if pattern in metric_name.lower() and key in thresholds['good']:
                metric_key = key
                break

        # Fallback to partial matching
        if not metric_key:
            for key in thresholds['good'].keys():
                if key.lower() in metric_name.lower() or metric_name.lower() in key.lower():
                    metric_key = key
                    break

        if not metric_key:
            # Error if no match found - don't use wrong thresholds!
            raise ValueError(f"No matching threshold found for metric '{metric_name}'. Available: {list(thresholds['good'].keys())}")

        good_threshold = thresholds['good'][metric_key]
        acceptable_threshold = thresholds['acceptable'][metric_key]
        poor_threshold = thresholds['poor'][metric_key]
    else:
        good_threshold = thresholds['good']
        acceptable_threshold = thresholds['acceptable']
        poor_threshold = thresholds['poor']

    good_val = parse_threshold(good_threshold)
    acceptable_val = parse_threshold(acceptable_threshold)
    poor_val = parse_threshold(poor_threshold)

    # Determine if lower or higher is better based on threshold pattern
    lower_is_better = '<' in good_threshold
    
    if lower_is_better:
        if value <= good_val:
            return "excellent", "✅", f"{metric_name} is excellent ({value:.3f} ≤ {good_val})"
        elif value <= acceptable_val:
            return "good", "✅", f"{metric_name} is good ({value:.3f} ≤ {acceptable_val})"  
        elif value <= poor_val:
            return "acceptable", "⚠️", f"{metric_name} is acceptable ({value:.3f} ≤ {poor_val})"
        else:
            return "poor", "❌", f"{metric_name} is poor ({value:.3f} > {poor_val})"
    else:
        if value >= good_val:
            return "excellent", "✅", f"{metric_name} is excellent ({value:.3f} ≥ {good_val})"
        elif value >= acceptable_val:
            return "good", "✅", f"{metric_name} is good ({value:.3f} ≥ {acceptable_val})"
        elif value >= poor_val:
            return "acceptable", "⚠️", f"{metric_name} is acceptable ({value:.3f} ≥ {poor_val})"
        else:
            return "poor", "❌", f"{metric_name} is poor ({value:.3f} < {poor_val})"


def generate_recommendations(assessments: Dict[str, Tuple], model_size: str, config: Dict) -> List[str]:
    """Generate specific recommendations based on assessment results."""
    recommendations = []
    
    # Get model-specific issues
    typical_issues = config["model_benchmarks"][model_size]["typical_issues"]
    
    for metric, (level, _, _) in assessments.items():
        if level in ["poor", "acceptable"]:
            if "perplexity" in metric:
                recommendations.extend([
                    "🔧 Consider longer pre-training (more epochs/steps)",
                    "📊 Check data quality and preprocessing",
                    "⚙️ Tune learning rate and batch size",
                    "💾 Verify training didn't collapse (loss plateaus)"
                ])
            
            elif "boolq" in metric:
                recommendations.extend([
                    "📚 Add instruction-following fine-tuning (SFT)",
                    "❓ Include question-answering datasets",
                    "🧠 Check if model understands Yes/No format",
                    "📝 Consider better prompt engineering"
                ])
                
            elif "smoke" in metric:
                recommendations.extend([
                    "🗣️ Improve conversational fine-tuning",
                    "✏️ Add more diverse training prompts",
                    "🔄 Check for repetition issues in generation",
                    "🎛️ Tune generation parameters (temperature, top_p)"
                ])
    
    # Add model-size specific suggestions
    if model_size == "tiny":
        recommendations.append("💡 Tiny models have natural limitations - consider upgrading to 'small'")
    elif model_size == "small":
        recommendations.append("⬆️ For better performance, consider training the 'base' model")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    return unique_recommendations[:8]  # Limit to top 8 recommendations


def assess_evaluation_results(results_file: str, config_file: str = None) -> None:
    """Main assessment function."""
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Load config
    config_path = config_file or "evaluation/evaluation_config.json"
    config = load_evaluation_config(config_path)
    
    print("=" * 60)
    print("🎯 PERFORMANCE ASSESSMENT REPORT")
    print("=" * 60)
    
    # Detect model size
    model_params = results.get("model_parameters", 0)
    model_size = detect_model_size(model_params)
    expected = config["model_benchmarks"][model_size]
    
    print(f"📊 Model: {model_params:,} parameters ({model_size.upper()} category)")
    print(f"🎯 Expected performance tier: {expected['expected_performance']}")
    print()
    
    # Assess each metric
    assessments = {}
    
    # Perplexity assessment
    if "perplexity_metrics" in results:
        ppl = results["perplexity_metrics"]["perplexity"]
        assessment = assess_metric(ppl, expected["performance_indicators"], "Perplexity")
        assessments["perplexity"] = assessment
        print(f"📈 {assessment[2]} {assessment[1]}")
        
    elif "perplexity" in results:
        ppl = results["perplexity"]
        assessment = assess_metric(ppl, expected["performance_indicators"], "Perplexity")
        assessments["perplexity"] = assessment
        print(f"📈 {assessment[2]} {assessment[1]}")
    
    # BoolQ assessment
    if "boolq_accuracy" in results:
        accuracy = results["boolq_accuracy"]
        assessment = assess_metric(accuracy, expected["performance_indicators"], "BoolQ accuracy")
        assessments["boolq"] = assessment
        print(f"🤔 {assessment[2]} {assessment[1]}")
    
    # Smoke test assessment
    if "smoke_tests" in results:
        if "summary_stats" in results["smoke_tests"]:
            quality = results["smoke_tests"]["summary_stats"]["avg_quality_score"]
            assessment = assess_metric(quality, expected["performance_indicators"], "Smoke test quality")
            assessments["smoke"] = assessment
            print(f"💬 {assessment[2]} {assessment[1]}")
    
    # Overall assessment
    print()
    print("🎯 OVERALL ASSESSMENT")
    print("-" * 30)
    
    levels = [assess[0] for assess in assessments.values()]
    if "excellent" in levels or levels.count("good") >= 2:
        overall = "🌟 EXCELLENT - Model performs very well!"
    elif "good" in levels and "poor" not in levels:
        overall = "✅ GOOD - Model meets expectations"
    elif "poor" not in levels:
        overall = "⚠️ ACCEPTABLE - Room for improvement"
    else:
        overall = "❌ NEEDS WORK - Significant issues detected"
    
    print(overall)
    
    # Recommendations
    recommendations = generate_recommendations(assessments, model_size, config)
    if recommendations:
        print()
        print("🛠️ RECOMMENDATIONS")
        print("-" * 20)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2}. {rec}")
    
    # Performance comparison
    print()
    print("📊 PERFORMANCE COMPARISON")
    print("-" * 25)
    
    target_ppl = expected["expected_performance"]["perplexity_target"]
    target_boolq = expected["expected_performance"]["boolq_target"]
    target_smoke = expected["expected_performance"]["smoke_test_target"]
    
    if "perplexity" in results or "perplexity_metrics" in results:
        actual_ppl = results.get("perplexity", results.get("perplexity_metrics", {}).get("perplexity"))
        if actual_ppl:
            ppl_ratio = target_ppl / actual_ppl if actual_ppl > 0 else 0
            status = "✅" if ppl_ratio >= 0.8 else "⚠️" if ppl_ratio >= 0.6 else "❌"
            print(f"Perplexity: {actual_ppl:.1f} vs target {target_ppl:.1f} {status}")
    
    if "boolq_accuracy" in results:
        actual_boolq = results["boolq_accuracy"]
        boolq_ratio = actual_boolq / target_boolq if target_boolq > 0 else 0
        status = "✅" if boolq_ratio >= 0.9 else "⚠️" if boolq_ratio >= 0.8 else "❌"
        print(f"BoolQ: {actual_boolq:.3f} vs target {target_boolq:.3f} {status}")
    
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Assess model evaluation results")
    parser.add_argument("results_file", help="Path to evaluation results JSON")
    parser.add_argument("--config", help="Path to evaluation config JSON", 
                       default="evaluation/evaluation_config.json")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"❌ Results file not found: {args.results_file}")
        return
    
    try:
        assess_evaluation_results(args.results_file, args.config)
    except Exception as e:
        print(f"❌ Error during assessment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()