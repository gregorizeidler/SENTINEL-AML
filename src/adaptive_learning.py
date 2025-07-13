"""
Adaptive Learning System for AML-FT Adversarial Simulation

This module implements adaptive learning capabilities that allow both Red Team
and Blue Team agents to improve their performance based on feedback from
previous simulation rounds.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import yaml

# For machine learning
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# LLM imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None


@dataclass
class SimulationRound:
    """Data class for storing simulation round results."""
    round_id: str
    timestamp: datetime
    config: Dict[str, Any]
    red_team_performance: Dict[str, Any]
    blue_team_performance: Dict[str, Any]
    overall_metrics: Dict[str, Any]
    lessons_learned: List[str]
    adaptations_made: Dict[str, Any]


@dataclass
class LearningInsight:
    """Data class for learning insights."""
    insight_id: str
    category: str  # 'red_team', 'blue_team', 'general'
    description: str
    confidence: float
    supporting_evidence: List[str]
    recommended_action: str
    impact_score: float
    timestamp: datetime


class AdaptiveLearningSystem:
    """
    Implements adaptive learning for the AML-FT adversarial simulation.
    
    This system analyzes performance across multiple simulation rounds
    and provides recommendations for improving both Red Team and Blue Team
    strategies and configurations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the adaptive learning system."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.simulation_history = []
        self.learning_insights = []
        self.adaptation_strategies = {}
        self.performance_trends = {}
        self.model_registry = {}
        
        # Load existing history if available
        self._load_history()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.3,
                'max_tokens': 3000
            },
            'adaptive_learning': {
                'min_rounds_for_learning': 3,
                'confidence_threshold': 0.7,
                'adaptation_aggressiveness': 0.5,
                'learning_rate': 0.1,
                'history_retention_days': 90
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for learning analysis."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def record_simulation_round(self, 
                              round_config: Dict,
                              red_team_results: Dict,
                              blue_team_results: Dict,
                              performance_metrics: Dict) -> str:
        """
        Record results from a simulation round for learning.
        
        Args:
            round_config: Configuration used for the round
            red_team_results: Red team execution results
            blue_team_results: Blue team analysis results
            performance_metrics: Overall performance metrics
            
        Returns:
            Round ID for the recorded simulation
        """
        round_id = f"ROUND_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract key performance indicators
        red_team_performance = self._extract_red_team_performance(red_team_results)
        blue_team_performance = self._extract_blue_team_performance(blue_team_results, performance_metrics)
        
        # Create simulation round record
        simulation_round = SimulationRound(
            round_id=round_id,
            timestamp=datetime.now(),
            config=round_config,
            red_team_performance=red_team_performance,
            blue_team_performance=blue_team_performance,
            overall_metrics=performance_metrics,
            lessons_learned=[],
            adaptations_made={}
        )
        
        self.simulation_history.append(simulation_round)
        
        # Trigger learning if we have enough data
        if len(self.simulation_history) >= self.config['adaptive_learning']['min_rounds_for_learning']:
            self._analyze_and_learn()
        
        # Save history
        self._save_history()
        
        print(f"📚 Recorded simulation round: {round_id}")
        return round_id
    
    def _extract_red_team_performance(self, red_team_results: Dict) -> Dict:
        """Extract Red Team performance metrics."""
        execution_result = red_team_results.get('execution_result', {})
        criminal_plan = red_team_results.get('criminal_plan', {})
        
        return {
            'execution_success': execution_result.get('success', False),
            'plan_complexity': criminal_plan.get('complexity_level', 'unknown'),
            'techniques_used': criminal_plan.get('techniques_used', []),
            'entities_created': len(execution_result.get('entities_created', {})),
            'transactions_generated': len(execution_result.get('criminal_transactions', [])),
            'plan_confidence': criminal_plan.get('risk_assessment', {}).get('detection_probability', 0.5),
            'execution_duration': execution_result.get('execution_context', {}).get('duration', 0),
            'target_amount': criminal_plan.get('target_amount', 0)
        }
    
    def _extract_blue_team_performance(self, blue_team_results: Dict, metrics: Dict) -> Dict:
        """Extract Blue Team performance metrics."""
        analysis_results = blue_team_results.get('analysis_results', {})
        
        return {
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'entities_detected': metrics.get('detected_entities', 0),
            'anomalies_found': analysis_results.get('anomalies_detected', {}).get('count', 0),
            'analysis_methods_used': len(analysis_results.get('analysis_methods', [])),
            'network_communities_found': len(analysis_results.get('network_analysis', {}).get('suspicious_subgraphs', [])),
            'structuring_cases_detected': analysis_results.get('structuring_analysis', {}).get('cases_detected', 0),
            'overall_risk_assessment': analysis_results.get('risk_assessment', {}).get('overall_risk_level', 'low')
        }
    
    def _analyze_and_learn(self):
        """Analyze simulation history and generate learning insights."""
        print("🧠 Analyzing simulation history for learning insights...")
        
        # Analyze performance trends
        self._analyze_performance_trends()
        
        # Generate insights using multiple approaches
        insights = []
        
        # Statistical analysis
        statistical_insights = self._generate_statistical_insights()
        insights.extend(statistical_insights)
        
        # Pattern recognition
        pattern_insights = self._generate_pattern_insights()
        insights.extend(pattern_insights)
        
        # LLM-based analysis
        if self.llm_client:
            llm_insights = self._generate_llm_insights()
            insights.extend(llm_insights)
        
        # Machine learning insights
        if HAS_SKLEARN:
            ml_insights = self._generate_ml_insights()
            insights.extend(ml_insights)
        
        # Store insights
        self.learning_insights.extend(insights)
        
        # Generate adaptation strategies
        self._generate_adaptation_strategies(insights)
        
        print(f"✅ Generated {len(insights)} learning insights")
    
    def _analyze_performance_trends(self):
        """Analyze performance trends across simulation rounds."""
        if len(self.simulation_history) < 2:
            return
        
        # Extract metrics over time
        rounds = sorted(self.simulation_history, key=lambda x: x.timestamp)
        
        trends = {
            'precision_trend': [r.blue_team_performance['precision'] for r in rounds],
            'recall_trend': [r.blue_team_performance['recall'] for r in rounds],
            'f1_score_trend': [r.blue_team_performance['f1_score'] for r in rounds],
            'red_team_success_trend': [r.red_team_performance['execution_success'] for r in rounds],
            'complexity_trend': [self._complexity_to_numeric(r.red_team_performance['plan_complexity']) for r in rounds]
        }
        
        # Calculate trend directions
        for metric, values in trends.items():
            if len(values) >= 3:
                recent_avg = np.mean(values[-3:])
                earlier_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]
                trend_direction = 'improving' if recent_avg > earlier_avg else 'declining'
                trends[f'{metric}_direction'] = trend_direction
        
        self.performance_trends = trends
    
    def _complexity_to_numeric(self, complexity: str) -> float:
        """Convert complexity level to numeric value."""
        mapping = {'simple': 1.0, 'medium': 2.0, 'complex': 3.0}
        return mapping.get(complexity, 1.0)
    
    def _generate_statistical_insights(self) -> List[LearningInsight]:
        """Generate insights using statistical analysis."""
        insights = []
        
        if len(self.simulation_history) < 3:
            return insights
        
        # Analyze correlation between Red Team complexity and Blue Team performance
        complexities = [self._complexity_to_numeric(r.red_team_performance['plan_complexity']) 
                       for r in self.simulation_history]
        f1_scores = [r.blue_team_performance['f1_score'] for r in self.simulation_history]
        
        if len(complexities) >= 3:
            correlation = np.corrcoef(complexities, f1_scores)[0, 1]
            
            if abs(correlation) > 0.5:
                insight = LearningInsight(
                    insight_id=f"STAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_001",
                    category='general',
                    description=f"Strong {'negative' if correlation < 0 else 'positive'} correlation ({correlation:.2f}) between Red Team complexity and Blue Team F1-score",
                    confidence=min(0.9, abs(correlation)),
                    supporting_evidence=[f"Correlation coefficient: {correlation:.3f}"],
                    recommended_action="Adjust complexity levels to optimize learning" if correlation < 0 else "Maintain current complexity progression",
                    impact_score=abs(correlation),
                    timestamp=datetime.now()
                )
                insights.append(insight)
        
        # Analyze technique effectiveness
        technique_performance = {}
        for round_data in self.simulation_history:
            techniques = round_data.red_team_performance.get('techniques_used', [])
            f1_score = round_data.blue_team_performance['f1_score']
            
            for technique in techniques:
                if technique not in technique_performance:
                    technique_performance[technique] = []
                technique_performance[technique].append(f1_score)
        
        # Find most/least effective techniques
        for technique, scores in technique_performance.items():
            if len(scores) >= 2:
                avg_detection = np.mean(scores)
                
                if avg_detection > 0.8:
                    insight = LearningInsight(
                        insight_id=f"STAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_002",
                        category='red_team',
                        description=f"Technique '{technique}' consistently detected (avg F1: {avg_detection:.2f})",
                        confidence=0.8,
                        supporting_evidence=[f"Average detection rate: {avg_detection:.2f} across {len(scores)} rounds"],
                        recommended_action=f"Red Team should improve or avoid '{technique}' technique",
                        impact_score=avg_detection,
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
                elif avg_detection < 0.4:
                    insight = LearningInsight(
                        insight_id=f"STAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_003",
                        category='blue_team',
                        description=f"Technique '{technique}' poorly detected (avg F1: {avg_detection:.2f})",
                        confidence=0.8,
                        supporting_evidence=[f"Average detection rate: {avg_detection:.2f} across {len(scores)} rounds"],
                        recommended_action=f"Blue Team should improve detection methods for '{technique}'",
                        impact_score=1.0 - avg_detection,
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_pattern_insights(self) -> List[LearningInsight]:
        """Generate insights using pattern recognition."""
        insights = []
        
        # Analyze configuration patterns that lead to better performance
        high_performance_rounds = [r for r in self.simulation_history 
                                 if r.blue_team_performance['f1_score'] > 0.7]
        low_performance_rounds = [r for r in self.simulation_history 
                                if r.blue_team_performance['f1_score'] < 0.4]
        
        if high_performance_rounds and low_performance_rounds:
            # Analyze configuration differences
            high_configs = [r.config for r in high_performance_rounds]
            low_configs = [r.config for r in low_performance_rounds]
            
            # Find common patterns in high-performance configurations
            high_detection_thresholds = [c.get('simulation', {}).get('blue_team', {}).get('detection_threshold', 0.7) 
                                       for c in high_configs]
            low_detection_thresholds = [c.get('simulation', {}).get('blue_team', {}).get('detection_threshold', 0.7) 
                                      for c in low_configs]
            
            if high_detection_thresholds and low_detection_thresholds:
                avg_high = np.mean(high_detection_thresholds)
                avg_low = np.mean(low_detection_thresholds)
                
                if abs(avg_high - avg_low) > 0.1:
                    insight = LearningInsight(
                        insight_id=f"PATT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_001",
                        category='blue_team',
                        description=f"Detection threshold of {avg_high:.2f} associated with higher performance vs {avg_low:.2f}",
                        confidence=0.7,
                        supporting_evidence=[f"High performance avg: {avg_high:.2f}, Low performance avg: {avg_low:.2f}"],
                        recommended_action=f"Consider using detection threshold around {avg_high:.2f}",
                        impact_score=abs(avg_high - avg_low),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_llm_insights(self) -> List[LearningInsight]:
        """Generate insights using LLM analysis."""
        insights = []
        
        if not self.llm_client or len(self.simulation_history) < 3:
            return insights
        
        # Prepare context for LLM
        context = self._prepare_llm_context()
        
        prompt = f"""
Analyze the following AML-FT adversarial simulation history and provide insights for improving performance:

{context}

Please identify:
1. Patterns in Red Team vs Blue Team performance
2. Configuration settings that correlate with better detection
3. Techniques that are consistently detected or missed
4. Recommendations for improving both teams

Provide your analysis in JSON format:
{{
    "insights": [
        {{
            "category": "red_team|blue_team|general",
            "description": "Clear description of the insight",
            "confidence": 0.8,
            "supporting_evidence": ["evidence1", "evidence2"],
            "recommended_action": "Specific actionable recommendation",
            "impact_score": 0.7
        }}
    ]
}}
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are an expert in financial crime detection and adversarial machine learning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                analysis = json.loads(json_str)
                
                # Convert to LearningInsight objects
                for i, insight_data in enumerate(analysis.get('insights', [])):
                    insight = LearningInsight(
                        insight_id=f"LLM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}",
                        category=insight_data.get('category', 'general'),
                        description=insight_data.get('description', ''),
                        confidence=insight_data.get('confidence', 0.5),
                        supporting_evidence=insight_data.get('supporting_evidence', []),
                        recommended_action=insight_data.get('recommended_action', ''),
                        impact_score=insight_data.get('impact_score', 0.5),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
                    
        except Exception as e:
            print(f"   ⚠️ LLM insight generation failed: {str(e)}")
        
        return insights
    
    def _prepare_llm_context(self) -> str:
        """Prepare context for LLM analysis."""
        context = "Simulation History Summary:\n\n"
        
        for i, round_data in enumerate(self.simulation_history[-5:], 1):  # Last 5 rounds
            context += f"Round {i}:\n"
            context += f"  Config: {round_data.config.get('simulation', {}).get('red_team', {}).get('complexity_level', 'unknown')} complexity, "
            context += f"{round_data.config.get('simulation', {}).get('blue_team', {}).get('detection_threshold', 0.7)} threshold\n"
            context += f"  Red Team: {', '.join(round_data.red_team_performance.get('techniques_used', []))}\n"
            context += f"  Blue Team: P={round_data.blue_team_performance['precision']:.2f}, "
            context += f"R={round_data.blue_team_performance['recall']:.2f}, "
            context += f"F1={round_data.blue_team_performance['f1_score']:.2f}\n\n"
        
        # Add trend information
        if self.performance_trends:
            context += "Performance Trends:\n"
            for metric, values in self.performance_trends.items():
                if not metric.endswith('_direction') and len(values) > 1:
                    trend = self.performance_trends.get(f"{metric}_direction", "stable")
                    context += f"  {metric}: {trend} (latest: {values[-1]:.2f})\n"
        
        return context
    
    def _generate_ml_insights(self) -> List[LearningInsight]:
        """Generate insights using machine learning analysis."""
        insights = []
        
        if not HAS_SKLEARN or len(self.simulation_history) < 5:
            return insights
        
        try:
            # Prepare feature matrix
            features = []
            targets = []
            
            for round_data in self.simulation_history:
                # Extract features from configuration and Red Team performance
                feature_vector = [
                    self._complexity_to_numeric(round_data.red_team_performance['plan_complexity']),
                    round_data.config.get('simulation', {}).get('blue_team', {}).get('detection_threshold', 0.7),
                    len(round_data.red_team_performance.get('techniques_used', [])),
                    round_data.red_team_performance.get('entities_created', 0),
                    round_data.red_team_performance.get('transactions_generated', 0),
                    round_data.blue_team_performance.get('analysis_methods_used', 0)
                ]
                
                # Target: High performance (F1 > 0.7)
                target = 1 if round_data.blue_team_performance['f1_score'] > 0.7 else 0
                
                features.append(feature_vector)
                targets.append(target)
            
            if len(set(targets)) > 1:  # Need both classes
                X = np.array(features)
                y = np.array(targets)
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Feature importance
                feature_names = [
                    'complexity', 'detection_threshold', 'techniques_count',
                    'entities_created', 'transactions_generated', 'analysis_methods'
                ]
                
                importances = model.feature_importances_
                
                # Generate insights from feature importance
                for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                    if importance > 0.2:  # Significant feature
                        insight = LearningInsight(
                            insight_id=f"ML_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}",
                            category='general',
                            description=f"Feature '{feature}' has high importance ({importance:.2f}) for detection performance",
                            confidence=0.8,
                            supporting_evidence=[f"Random Forest feature importance: {importance:.3f}"],
                            recommended_action=f"Focus on optimizing '{feature}' parameter",
                            impact_score=importance,
                            timestamp=datetime.now()
                        )
                        insights.append(insight)
                
                # Store model for future use
                self.model_registry['performance_predictor'] = {
                    'model': model,
                    'feature_names': feature_names,
                    'trained_at': datetime.now()
                }
                
        except Exception as e:
            print(f"   ⚠️ ML insight generation failed: {str(e)}")
        
        return insights
    
    def _generate_adaptation_strategies(self, insights: List[LearningInsight]):
        """Generate adaptation strategies based on insights."""
        strategies = {
            'red_team_adaptations': [],
            'blue_team_adaptations': [],
            'configuration_adaptations': []
        }
        
        # Group insights by category
        red_team_insights = [i for i in insights if i.category == 'red_team']
        blue_team_insights = [i for i in insights if i.category == 'blue_team']
        general_insights = [i for i in insights if i.category == 'general']
        
        # Generate Red Team adaptations
        for insight in red_team_insights:
            if insight.confidence > self.config['adaptive_learning']['confidence_threshold']:
                strategies['red_team_adaptations'].append({
                    'strategy': insight.recommended_action,
                    'reasoning': insight.description,
                    'confidence': insight.confidence,
                    'priority': insight.impact_score
                })
        
        # Generate Blue Team adaptations
        for insight in blue_team_insights:
            if insight.confidence > self.config['adaptive_learning']['confidence_threshold']:
                strategies['blue_team_adaptations'].append({
                    'strategy': insight.recommended_action,
                    'reasoning': insight.description,
                    'confidence': insight.confidence,
                    'priority': insight.impact_score
                })
        
        # Generate configuration adaptations
        for insight in general_insights:
            if insight.confidence > self.config['adaptive_learning']['confidence_threshold']:
                strategies['configuration_adaptations'].append({
                    'strategy': insight.recommended_action,
                    'reasoning': insight.description,
                    'confidence': insight.confidence,
                    'priority': insight.impact_score
                })
        
        self.adaptation_strategies = strategies
    
    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """Get adaptive recommendations for the next simulation round."""
        if not self.adaptation_strategies:
            return {'message': 'No adaptations available. Run more simulations to generate insights.'}
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'total_rounds_analyzed': len(self.simulation_history),
            'insights_generated': len(self.learning_insights),
            'adaptations': self.adaptation_strategies,
            'performance_trends': self.performance_trends
        }
        
        # Add specific parameter recommendations
        if self.performance_trends:
            recommendations['parameter_recommendations'] = self._generate_parameter_recommendations()
        
        return recommendations
    
    def _generate_parameter_recommendations(self) -> Dict[str, Any]:
        """Generate specific parameter recommendations."""
        recommendations = {}
        
        # Analyze F1-score trend
        f1_trend = self.performance_trends.get('f1_score_trend', [])
        if len(f1_trend) >= 3:
            recent_performance = np.mean(f1_trend[-3:])
            
            if recent_performance < 0.5:
                recommendations['detection_threshold'] = {
                    'current_avg': recent_performance,
                    'recommended_change': 'decrease',
                    'suggested_value': 0.6,
                    'reasoning': 'Low F1-score suggests threshold may be too high'
                }
            elif recent_performance > 0.8:
                recommendations['complexity_level'] = {
                    'current_avg': recent_performance,
                    'recommended_change': 'increase',
                    'suggested_value': 'complex',
                    'reasoning': 'High F1-score suggests Red Team should use more complex techniques'
                }
        
        return recommendations
    
    def apply_adaptations(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned adaptations to a base configuration."""
        adapted_config = base_config.copy()
        
        if not self.adaptation_strategies:
            return adapted_config
        
        # Apply configuration adaptations
        config_adaptations = self.adaptation_strategies.get('configuration_adaptations', [])
        
        for adaptation in config_adaptations:
            if adaptation['confidence'] > 0.8:  # High confidence adaptations
                strategy = adaptation['strategy']
                
                # Parse and apply specific adaptations
                if 'detection threshold' in strategy.lower():
                    # Extract suggested threshold value
                    import re
                    threshold_match = re.search(r'(\d+\.?\d*)', strategy)
                    if threshold_match:
                        new_threshold = float(threshold_match.group(1))
                        adapted_config.setdefault('simulation', {}).setdefault('blue_team', {})['detection_threshold'] = new_threshold
                
                elif 'complexity' in strategy.lower():
                    if 'increase' in strategy.lower():
                        adapted_config.setdefault('simulation', {}).setdefault('red_team', {})['complexity_level'] = 'complex'
                    elif 'decrease' in strategy.lower():
                        adapted_config.setdefault('simulation', {}).setdefault('red_team', {})['complexity_level'] = 'simple'
        
        return adapted_config
    
    def _save_history(self):
        """Save simulation history to file."""
        history_dir = Path("adaptive_learning")
        history_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        history_data = {
            'simulation_history': [asdict(round_data) for round_data in self.simulation_history],
            'learning_insights': [asdict(insight) for insight in self.learning_insights],
            'adaptation_strategies': self.adaptation_strategies,
            'performance_trends': self.performance_trends
        }
        
        with open(history_dir / "learning_history.json", 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        # Save models if available
        if self.model_registry:
            with open(history_dir / "model_registry.pkl", 'wb') as f:
                pickle.dump(self.model_registry, f)
    
    def _load_history(self):
        """Load existing simulation history."""
        history_file = Path("adaptive_learning/learning_history.json")
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct simulation history
                self.simulation_history = [
                    SimulationRound(**round_data) for round_data in data.get('simulation_history', [])
                ]
                
                # Reconstruct learning insights
                self.learning_insights = [
                    LearningInsight(**insight_data) for insight_data in data.get('learning_insights', [])
                ]
                
                self.adaptation_strategies = data.get('adaptation_strategies', {})
                self.performance_trends = data.get('performance_trends', {})
                
                print(f"📚 Loaded {len(self.simulation_history)} simulation rounds from history")
                
            except Exception as e:
                print(f"⚠️ Failed to load history: {str(e)}")
        
        # Load models
        model_file = Path("adaptive_learning/model_registry.pkl")
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    self.model_registry = pickle.load(f)
                print(f"📚 Loaded {len(self.model_registry)} trained models")
            except Exception as e:
                print(f"⚠️ Failed to load models: {str(e)}")
    
    def get_learning_report(self) -> str:
        """Generate a comprehensive learning report."""
        if not self.simulation_history:
            return "No learning data available. Run simulations to generate insights."
        
        report = f"""
ADAPTIVE LEARNING REPORT
========================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Simulation Rounds: {len(self.simulation_history)}
Learning Insights Generated: {len(self.learning_insights)}

PERFORMANCE TRENDS
==================
"""
        
        if self.performance_trends:
            for metric, values in self.performance_trends.items():
                if not metric.endswith('_direction') and len(values) > 1:
                    trend = self.performance_trends.get(f"{metric}_direction", "stable")
                    current = values[-1] if values else 0
                    report += f"{metric}: {trend} (current: {current:.3f})\n"
        
        report += f"\nKEY INSIGHTS\n============\n"
        
        # Group insights by category
        categories = ['red_team', 'blue_team', 'general']
        for category in categories:
            category_insights = [i for i in self.learning_insights if i.category == category]
            if category_insights:
                report += f"\n{category.replace('_', ' ').title()} Insights:\n"
                for insight in category_insights[-3:]:  # Last 3 insights per category
                    report += f"  • {insight.description} (Confidence: {insight.confidence:.2f})\n"
                    report += f"    Action: {insight.recommended_action}\n"
        
        report += f"\nADAPTATION STRATEGIES\n====================\n"
        
        if self.adaptation_strategies:
            for category, adaptations in self.adaptation_strategies.items():
                if adaptations:
                    report += f"\n{category.replace('_', ' ').title()}:\n"
                    for adaptation in adaptations[:3]:  # Top 3 adaptations
                        report += f"  • {adaptation['strategy']} (Priority: {adaptation['priority']:.2f})\n"
        
        return report


def main():
    """Main function for testing the Adaptive Learning System."""
    print("Testing Adaptive Learning System...")
    
    # Initialize system
    learning_system = AdaptiveLearningSystem()
    
    # Simulate some rounds
    for i in range(5):
        # Create mock simulation data
        round_config = {
            'simulation': {
                'red_team': {
                    'complexity_level': random.choice(['simple', 'medium', 'complex']),
                    'target_amount': random.randint(100000, 1000000)
                },
                'blue_team': {
                    'detection_threshold': random.uniform(0.5, 0.9)
                }
            }
        }
        
        red_team_results = {
            'criminal_plan': {
                'complexity_level': round_config['simulation']['red_team']['complexity_level'],
                'techniques_used': random.sample(['smurfing', 'shell_companies', 'money_mules'], 2),
                'target_amount': round_config['simulation']['red_team']['target_amount']
            },
            'execution_result': {
                'success': random.choice([True, False]),
                'entities_created': {'money_mule': [{'entity_id': f'MULE_{j}'} for j in range(random.randint(2, 5))]},
                'criminal_transactions': [{'id': f'TX_{k}'} for k in range(random.randint(10, 50))]
            }
        }
        
        blue_team_results = {
            'analysis_results': {
                'analysis_methods': ['statistical_analysis', 'anomaly_detection', 'network_analysis'],
                'anomalies_detected': {'count': random.randint(5, 20)},
                'network_analysis': {'suspicious_subgraphs': []},
                'structuring_analysis': {'cases_detected': random.randint(0, 3)}
            }
        }
        
        performance_metrics = {
            'precision': random.uniform(0.3, 0.9),
            'recall': random.uniform(0.3, 0.9),
            'f1_score': random.uniform(0.3, 0.9),
            'detected_entities': random.randint(3, 15),
            'actual_criminal_entities': random.randint(5, 20)
        }
        
        # Record round
        round_id = learning_system.record_simulation_round(
            round_config, red_team_results, blue_team_results, performance_metrics
        )
        
        print(f"Recorded round: {round_id}")
    
    # Get recommendations
    print("\nAdaptive Recommendations:")
    recommendations = learning_system.get_adaptive_recommendations()
    print(json.dumps(recommendations, indent=2, default=str))
    
    # Generate learning report
    print("\nLearning Report:")
    print("=" * 60)
    print(learning_system.get_learning_report())


if __name__ == "__main__":
    main() 