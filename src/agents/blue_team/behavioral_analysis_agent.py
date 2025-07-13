"""
Blue Team Behavioral Analysis Agent

This agent performs advanced behavioral analysis of entities in the AML-FT system.
It creates behavioral profiles, detects anomalies, analyzes temporal patterns,
and identifies deviations from normal behavior.
"""

import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import pickle

# Statistical analysis imports
try:
    from scipy import stats
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# LLM imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None


@dataclass
class BehavioralProfile:
    """Data class for behavioral profiles."""
    entity_id: str
    entity_type: str
    profile_type: str  # 'baseline', 'current', 'anomalous'
    features: Dict[str, float]
    temporal_patterns: Dict[str, Any]
    transaction_patterns: Dict[str, Any]
    network_behavior: Dict[str, Any]
    risk_indicators: List[str]
    confidence_score: float
    created_at: datetime
    last_updated: datetime
    sample_size: int


@dataclass
class BehavioralAnomaly:
    """Data class for behavioral anomalies."""
    anomaly_id: str
    entity_id: str
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    deviation_score: float
    affected_features: List[str]
    description: str
    evidence: List[str]
    baseline_values: Dict[str, float]
    current_values: Dict[str, float]
    temporal_context: Dict[str, Any]
    recommended_actions: List[str]
    timestamp: datetime


@dataclass
class BehavioralInsight:
    """Data class for behavioral insights."""
    insight_id: str
    entity_id: str
    insight_type: str
    description: str
    statistical_significance: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    time_period: str
    supporting_data: Dict[str, Any]
    implications: List[str]
    timestamp: datetime


class BehavioralAnalysisAgent:
    """
    Performs advanced behavioral analysis for AML-FT operations.
    
    This agent creates behavioral profiles, detects anomalies,
    analyzes temporal patterns, and provides behavioral insights
    for risk assessment and investigation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Behavioral Analysis Agent."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.behavioral_profiles = {}
        self.behavioral_anomalies = []
        self.behavioral_insights = []
        self.baseline_models = {}
        self.scaler = StandardScaler() if HAS_SCIPY else None
        self.feature_extractors = self._initialize_feature_extractors()
        
        # Load existing profiles
        self._load_profiles()
        
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
                'max_tokens': 2500
            },
            'behavioral_analysis': {
                'baseline_period_days': 30,
                'anomaly_threshold': 2.0,  # Standard deviations
                'min_transactions_for_profile': 10,
                'profile_update_frequency': 'daily',
                'temporal_window_hours': 24,
                'confidence_threshold': 0.7
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for behavioral analysis."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def _initialize_feature_extractors(self) -> Dict[str, callable]:
        """Initialize feature extraction functions."""
        return {
            'transaction_frequency': self._extract_transaction_frequency,
            'amount_patterns': self._extract_amount_patterns,
            'temporal_patterns': self._extract_temporal_patterns,
            'counterparty_patterns': self._extract_counterparty_patterns,
            'geographic_patterns': self._extract_geographic_patterns,
            'velocity_patterns': self._extract_velocity_patterns,
            'seasonality_patterns': self._extract_seasonality_patterns,
            'network_behavior': self._extract_network_behavior
        }
    
    def create_behavioral_profile(self, entity_id: str, transactions: pd.DataFrame,
                                entity_data: Dict = None) -> BehavioralProfile:
        """
        Create comprehensive behavioral profile for an entity.
        
        Args:
            entity_id: ID of the entity to profile
            transactions: DataFrame of transactions for the entity
            entity_data: Additional entity information
            
        Returns:
            BehavioralProfile object
        """
        print(f"🧠 Creating behavioral profile for entity: {entity_id}")
        
        # Filter transactions for this entity
        entity_transactions = transactions[
            (transactions['sender_id'] == entity_id) | 
            (transactions['receiver_id'] == entity_id)
        ].copy()
        
        if len(entity_transactions) < self.config['behavioral_analysis']['min_transactions_for_profile']:
            return self._create_minimal_profile(entity_id, entity_data)
        
        # Extract behavioral features
        features = {}
        for feature_name, extractor in self.feature_extractors.items():
            try:
                feature_value = extractor(entity_id, entity_transactions)
                features[feature_name] = feature_value
            except Exception as e:
                print(f"⚠️ Failed to extract {feature_name}: {str(e)}")
                features[feature_name] = 0.0
        
        # Extract temporal patterns
        temporal_patterns = self._analyze_temporal_behavior(entity_id, entity_transactions)
        
        # Extract transaction patterns
        transaction_patterns = self._analyze_transaction_patterns(entity_id, entity_transactions)
        
        # Extract network behavior
        network_behavior = self._analyze_network_behavior(entity_id, entity_transactions)
        
        # Identify risk indicators
        risk_indicators = self._identify_risk_indicators(features, temporal_patterns, transaction_patterns)
        
        # Calculate confidence score
        confidence_score = self._calculate_profile_confidence(entity_transactions, features)
        
        # Create profile
        profile = BehavioralProfile(
            entity_id=entity_id,
            entity_type=entity_data.get('entity_type', 'unknown') if entity_data else 'unknown',
            profile_type='baseline',
            features=features,
            temporal_patterns=temporal_patterns,
            transaction_patterns=transaction_patterns,
            network_behavior=network_behavior,
            risk_indicators=risk_indicators,
            confidence_score=confidence_score,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            sample_size=len(entity_transactions)
        )
        
        # Store profile
        self.behavioral_profiles[entity_id] = profile
        
        print(f"✅ Behavioral profile created (confidence: {confidence_score:.3f})")
        return profile
    
    def _extract_transaction_frequency(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract transaction frequency feature."""
        if transactions.empty:
            return 0.0
        
        # Calculate transactions per day
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        date_range = (transactions['timestamp'].max() - transactions['timestamp'].min()).days
        
        if date_range == 0:
            return len(transactions)
        
        return len(transactions) / max(date_range, 1)
    
    def _extract_amount_patterns(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract amount pattern features."""
        if transactions.empty:
            return 0.0
        
        amounts = transactions['amount'].values
        
        # Calculate coefficient of variation
        if len(amounts) > 1 and np.mean(amounts) > 0:
            cv = np.std(amounts) / np.mean(amounts)
        else:
            cv = 0.0
        
        return cv
    
    def _extract_temporal_patterns(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract temporal pattern features."""
        if transactions.empty:
            return 0.0
        
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions['hour'] = transactions['timestamp'].dt.hour
        
        # Calculate temporal dispersion
        hour_counts = transactions['hour'].value_counts()
        if len(hour_counts) > 1:
            # Entropy as measure of temporal dispersion
            probabilities = hour_counts / len(transactions)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy / np.log2(24)  # Normalize to [0,1]
        
        return 0.0
    
    def _extract_counterparty_patterns(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract counterparty pattern features."""
        if transactions.empty:
            return 0.0
        
        # Get counterparties
        counterparties = set()
        for _, txn in transactions.iterrows():
            if txn['sender_id'] == entity_id:
                counterparties.add(txn['receiver_id'])
            else:
                counterparties.add(txn['sender_id'])
        
        # Calculate counterparty diversity
        if len(transactions) > 0:
            return len(counterparties) / len(transactions)
        
        return 0.0
    
    def _extract_geographic_patterns(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract geographic pattern features."""
        if transactions.empty or 'location' not in transactions.columns:
            return 0.0
        
        locations = transactions['location'].value_counts()
        
        if len(locations) > 1:
            # Geographic concentration (inverse of diversity)
            max_concentration = locations.iloc[0] / len(transactions)
            return max_concentration
        
        return 1.0  # Complete concentration
    
    def _extract_velocity_patterns(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract velocity pattern features."""
        if transactions.empty:
            return 0.0
        
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions_sorted = transactions.sort_values('timestamp')
        
        # Calculate time differences between consecutive transactions
        time_diffs = transactions_sorted['timestamp'].diff().dt.total_seconds()
        time_diffs = time_diffs.dropna()
        
        if len(time_diffs) > 0:
            # Average time between transactions (in hours)
            avg_time_diff = np.mean(time_diffs) / 3600
            return 1.0 / (avg_time_diff + 1)  # Velocity score
        
        return 0.0
    
    def _extract_seasonality_patterns(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract seasonality pattern features."""
        if transactions.empty:
            return 0.0
        
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
        
        # Calculate weekday vs weekend ratio
        weekday_txns = len(transactions[transactions['day_of_week'] < 5])
        weekend_txns = len(transactions[transactions['day_of_week'] >= 5])
        
        if weekend_txns > 0:
            return weekday_txns / weekend_txns
        
        return float('inf') if weekday_txns > 0 else 0.0
    
    def _extract_network_behavior(self, entity_id: str, transactions: pd.DataFrame) -> float:
        """Extract network behavior features."""
        if transactions.empty:
            return 0.0
        
        # Calculate in-degree and out-degree
        in_degree = len(transactions[transactions['receiver_id'] == entity_id])
        out_degree = len(transactions[transactions['sender_id'] == entity_id])
        
        total_degree = in_degree + out_degree
        
        if total_degree > 0:
            # Balance between incoming and outgoing transactions
            return abs(in_degree - out_degree) / total_degree
        
        return 0.0
    
    def _analyze_temporal_behavior(self, entity_id: str, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal behavior patterns."""
        if transactions.empty:
            return {}
        
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        
        # Hour of day analysis
        hour_distribution = transactions['timestamp'].dt.hour.value_counts().sort_index()
        
        # Day of week analysis
        dow_distribution = transactions['timestamp'].dt.dayofweek.value_counts().sort_index()
        
        # Monthly analysis
        month_distribution = transactions['timestamp'].dt.month.value_counts().sort_index()
        
        # Peak activity times
        peak_hour = hour_distribution.idxmax() if not hour_distribution.empty else 0
        peak_day = dow_distribution.idxmax() if not dow_distribution.empty else 0
        
        return {
            'peak_hour': int(peak_hour),
            'peak_day': int(peak_day),
            'hour_distribution': hour_distribution.to_dict(),
            'dow_distribution': dow_distribution.to_dict(),
            'month_distribution': month_distribution.to_dict(),
            'activity_spread': len(hour_distribution[hour_distribution > 0]),
            'weekend_activity_ratio': len(transactions[transactions['timestamp'].dt.dayofweek >= 5]) / len(transactions)
        }
    
    def _analyze_transaction_patterns(self, entity_id: str, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction patterns."""
        if transactions.empty:
            return {}
        
        amounts = transactions['amount'].values
        
        # Amount statistics
        amount_stats = {
            'mean': float(np.mean(amounts)),
            'median': float(np.median(amounts)),
            'std': float(np.std(amounts)),
            'min': float(np.min(amounts)),
            'max': float(np.max(amounts)),
            'q25': float(np.percentile(amounts, 25)),
            'q75': float(np.percentile(amounts, 75))
        }
        
        # Round amount analysis
        round_amounts = np.sum(amounts % 1000 == 0) / len(amounts)
        
        # Threshold analysis (amounts near common thresholds)
        threshold_amounts = np.sum((amounts >= 9000) & (amounts < 10000)) / len(amounts)
        
        # Transaction type analysis
        sent_txns = transactions[transactions['sender_id'] == entity_id]
        received_txns = transactions[transactions['receiver_id'] == entity_id]
        
        return {
            'amount_statistics': amount_stats,
            'round_amount_ratio': float(round_amounts),
            'threshold_amount_ratio': float(threshold_amounts),
            'sent_count': len(sent_txns),
            'received_count': len(received_txns),
            'sent_volume': float(sent_txns['amount'].sum()) if not sent_txns.empty else 0.0,
            'received_volume': float(received_txns['amount'].sum()) if not received_txns.empty else 0.0,
            'transaction_balance': len(sent_txns) - len(received_txns),
            'volume_balance': float(sent_txns['amount'].sum() - received_txns['amount'].sum()) if not sent_txns.empty and not received_txns.empty else 0.0
        }
    
    def _analyze_network_behavior(self, entity_id: str, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze network behavior patterns."""
        if transactions.empty:
            return {}
        
        # Counterparty analysis
        counterparties = set()
        counterparty_volumes = defaultdict(float)
        
        for _, txn in transactions.iterrows():
            if txn['sender_id'] == entity_id:
                counterparty = txn['receiver_id']
                counterparties.add(counterparty)
                counterparty_volumes[counterparty] += txn['amount']
            else:
                counterparty = txn['sender_id']
                counterparties.add(counterparty)
                counterparty_volumes[counterparty] += txn['amount']
        
        # Concentration analysis
        if counterparty_volumes:
            total_volume = sum(counterparty_volumes.values())
            max_counterparty_volume = max(counterparty_volumes.values())
            concentration_ratio = max_counterparty_volume / total_volume
        else:
            concentration_ratio = 0.0
        
        return {
            'unique_counterparties': len(counterparties),
            'counterparty_concentration': float(concentration_ratio),
            'avg_counterparty_volume': float(np.mean(list(counterparty_volumes.values()))) if counterparty_volumes else 0.0,
            'counterparty_diversity': len(counterparties) / len(transactions) if len(transactions) > 0 else 0.0
        }
    
    def _identify_risk_indicators(self, features: Dict[str, float], 
                                temporal_patterns: Dict[str, Any],
                                transaction_patterns: Dict[str, Any]) -> List[str]:
        """Identify behavioral risk indicators."""
        risk_indicators = []
        
        # High transaction frequency
        if features.get('transaction_frequency', 0) > 10:
            risk_indicators.append("High transaction frequency")
        
        # High amount variability
        if features.get('amount_patterns', 0) > 2.0:
            risk_indicators.append("High amount variability")
        
        # Unusual temporal patterns
        if temporal_patterns.get('weekend_activity_ratio', 0) > 0.5:
            risk_indicators.append("High weekend activity")
        
        # Round amount preference
        if transaction_patterns.get('round_amount_ratio', 0) > 0.3:
            risk_indicators.append("Preference for round amounts")
        
        # Threshold structuring
        if transaction_patterns.get('threshold_amount_ratio', 0) > 0.1:
            risk_indicators.append("Potential structuring behavior")
        
        # High counterparty concentration
        if features.get('counterparty_patterns', 0) < 0.1:
            risk_indicators.append("Low counterparty diversity")
        
        # High velocity
        if features.get('velocity_patterns', 0) > 0.8:
            risk_indicators.append("High transaction velocity")
        
        return risk_indicators
    
    def _calculate_profile_confidence(self, transactions: pd.DataFrame, features: Dict[str, float]) -> float:
        """Calculate confidence score for behavioral profile."""
        base_confidence = 0.5
        
        # Increase confidence with more data
        data_confidence = min(len(transactions) / 100, 0.3)
        
        # Increase confidence with feature completeness
        feature_completeness = len([f for f in features.values() if f > 0]) / len(features)
        feature_confidence = feature_completeness * 0.2
        
        return min(base_confidence + data_confidence + feature_confidence, 1.0)
    
    def _create_minimal_profile(self, entity_id: str, entity_data: Dict = None) -> BehavioralProfile:
        """Create minimal profile for entities with insufficient data."""
        return BehavioralProfile(
            entity_id=entity_id,
            entity_type=entity_data.get('entity_type', 'unknown') if entity_data else 'unknown',
            profile_type='minimal',
            features={},
            temporal_patterns={},
            transaction_patterns={},
            network_behavior={},
            risk_indicators=['Insufficient data for behavioral analysis'],
            confidence_score=0.1,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            sample_size=0
        )
    
    def detect_behavioral_anomalies(self, entity_id: str, transactions: pd.DataFrame,
                                  lookback_days: int = 7) -> List[BehavioralAnomaly]:
        """
        Detect behavioral anomalies by comparing recent behavior to baseline.
        
        Args:
            entity_id: ID of the entity to analyze
            transactions: DataFrame of recent transactions
            lookback_days: Number of days to look back for anomaly detection
            
        Returns:
            List of behavioral anomalies detected
        """
        print(f"🔍 Detecting behavioral anomalies for entity: {entity_id}")
        
        anomalies = []
        
        # Get baseline profile
        if entity_id not in self.behavioral_profiles:
            print(f"⚠️ No baseline profile found for {entity_id}")
            return anomalies
        
        baseline_profile = self.behavioral_profiles[entity_id]
        
        # Filter recent transactions
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_transactions = transactions[
            pd.to_datetime(transactions['timestamp']) >= cutoff_date
        ]
        
        entity_recent_txns = recent_transactions[
            (recent_transactions['sender_id'] == entity_id) | 
            (recent_transactions['receiver_id'] == entity_id)
        ]
        
        if entity_recent_txns.empty:
            return anomalies
        
        # Create current profile
        current_profile = self.create_behavioral_profile(entity_id, entity_recent_txns)
        current_profile.profile_type = 'current'
        
        # Compare profiles
        for feature_name in baseline_profile.features:
            baseline_value = baseline_profile.features[feature_name]
            current_value = current_profile.features.get(feature_name, 0.0)
            
            # Calculate deviation
            if baseline_value != 0:
                deviation = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                deviation = abs(current_value)
            
            # Check if deviation exceeds threshold
            if deviation > self.config['behavioral_analysis']['anomaly_threshold']:
                anomaly = self._create_behavioral_anomaly(
                    entity_id, feature_name, baseline_value, current_value, deviation
                )
                anomalies.append(anomaly)
        
        # Store anomalies
        self.behavioral_anomalies.extend(anomalies)
        
        print(f"✅ Found {len(anomalies)} behavioral anomalies")
        return anomalies
    
    def _create_behavioral_anomaly(self, entity_id: str, feature_name: str,
                                 baseline_value: float, current_value: float,
                                 deviation: float) -> BehavioralAnomaly:
        """Create behavioral anomaly object."""
        # Determine severity
        if deviation > 5.0:
            severity = 'critical'
        elif deviation > 3.0:
            severity = 'high'
        elif deviation > 2.0:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate description
        change_direction = "increased" if current_value > baseline_value else "decreased"
        description = f"{feature_name.replace('_', ' ').title()} has {change_direction} significantly"
        
        # Generate evidence
        evidence = [
            f"Baseline {feature_name}: {baseline_value:.3f}",
            f"Current {feature_name}: {current_value:.3f}",
            f"Deviation: {deviation:.3f}x from baseline"
        ]
        
        # Generate recommendations
        recommendations = self._generate_anomaly_recommendations(feature_name, severity)
        
        return BehavioralAnomaly(
            anomaly_id=f"BEHAV_ANOM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}",
            entity_id=entity_id,
            anomaly_type=f"{feature_name}_anomaly",
            severity=severity,
            deviation_score=deviation,
            affected_features=[feature_name],
            description=description,
            evidence=evidence,
            baseline_values={feature_name: baseline_value},
            current_values={feature_name: current_value},
            temporal_context={'analysis_period': 'recent_7_days'},
            recommended_actions=recommendations,
            timestamp=datetime.now()
        )
    
    def _generate_anomaly_recommendations(self, feature_name: str, severity: str) -> List[str]:
        """Generate recommendations based on anomaly type and severity."""
        recommendations = []
        
        if severity in ['critical', 'high']:
            recommendations.append("Immediate investigation required")
            recommendations.append("Enhanced monitoring")
        
        if feature_name == 'transaction_frequency':
            recommendations.extend([
                "Review transaction patterns",
                "Check for automated systems",
                "Verify business justification"
            ])
        elif feature_name == 'amount_patterns':
            recommendations.extend([
                "Analyze amount distributions",
                "Check for structuring patterns",
                "Review transaction purposes"
            ])
        elif feature_name == 'temporal_patterns':
            recommendations.extend([
                "Review activity timing",
                "Check for unusual hours",
                "Verify business operations"
            ])
        elif feature_name == 'counterparty_patterns':
            recommendations.extend([
                "Review counterparty relationships",
                "Check for new business partners",
                "Verify counterparty legitimacy"
            ])
        
        return recommendations
    
    def generate_behavioral_insights(self, entity_id: str, time_period: str = '30d') -> List[BehavioralInsight]:
        """
        Generate behavioral insights for an entity.
        
        Args:
            entity_id: ID of the entity to analyze
            time_period: Time period for analysis (e.g., '30d', '7d')
            
        Returns:
            List of behavioral insights
        """
        print(f"💡 Generating behavioral insights for entity: {entity_id}")
        
        insights = []
        
        if entity_id not in self.behavioral_profiles:
            return insights
        
        profile = self.behavioral_profiles[entity_id]
        
        # Analyze trends in behavioral features
        insights.extend(self._analyze_feature_trends(entity_id, profile))
        
        # Analyze temporal behavior insights
        insights.extend(self._analyze_temporal_insights(entity_id, profile))
        
        # Analyze network behavior insights
        insights.extend(self._analyze_network_insights(entity_id, profile))
        
        # Store insights
        self.behavioral_insights.extend(insights)
        
        print(f"✅ Generated {len(insights)} behavioral insights")
        return insights
    
    def _analyze_feature_trends(self, entity_id: str, profile: BehavioralProfile) -> List[BehavioralInsight]:
        """Analyze trends in behavioral features."""
        insights = []
        
        # High-risk feature analysis
        high_risk_features = ['transaction_frequency', 'velocity_patterns', 'amount_patterns']
        
        for feature in high_risk_features:
            if feature in profile.features:
                feature_value = profile.features[feature]
                
                if feature_value > 0.8:  # High threshold
                    insight = BehavioralInsight(
                        insight_id=f"INSIGHT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}",
                        entity_id=entity_id,
                        insight_type=f"high_{feature}",
                        description=f"Entity shows high {feature.replace('_', ' ')}: {feature_value:.3f}",
                        statistical_significance=0.95,
                        trend_direction='increasing',
                        time_period='current',
                        supporting_data={feature: feature_value},
                        implications=[
                            f"Potential risk indicator: {feature.replace('_', ' ')}",
                            "Requires enhanced monitoring",
                            "May indicate suspicious activity"
                        ],
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
        
        return insights
    
    def _analyze_temporal_insights(self, entity_id: str, profile: BehavioralProfile) -> List[BehavioralInsight]:
        """Analyze temporal behavior insights."""
        insights = []
        
        temporal_patterns = profile.temporal_patterns
        
        # Weekend activity analysis
        if temporal_patterns.get('weekend_activity_ratio', 0) > 0.3:
            insight = BehavioralInsight(
                insight_id=f"TEMP_INSIGHT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}",
                entity_id=entity_id,
                insight_type='high_weekend_activity',
                description=f"Entity shows high weekend activity: {temporal_patterns['weekend_activity_ratio']:.1%}",
                statistical_significance=0.8,
                trend_direction='stable',
                time_period='current',
                supporting_data={'weekend_ratio': temporal_patterns['weekend_activity_ratio']},
                implications=[
                    "Unusual business hours activity",
                    "May indicate personal rather than business transactions",
                    "Requires verification of business operations"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # Peak hour analysis
        peak_hour = temporal_patterns.get('peak_hour', 12)
        if peak_hour < 6 or peak_hour > 22:
            insight = BehavioralInsight(
                insight_id=f"PEAK_INSIGHT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}",
                entity_id=entity_id,
                insight_type='unusual_peak_hours',
                description=f"Entity's peak activity hour is unusual: {peak_hour}:00",
                statistical_significance=0.7,
                trend_direction='stable',
                time_period='current',
                supporting_data={'peak_hour': peak_hour},
                implications=[
                    "Activity during unusual hours",
                    "May indicate automated systems",
                    "Requires business justification"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_network_insights(self, entity_id: str, profile: BehavioralProfile) -> List[BehavioralInsight]:
        """Analyze network behavior insights."""
        insights = []
        
        network_behavior = profile.network_behavior
        
        # Counterparty concentration analysis
        concentration = network_behavior.get('counterparty_concentration', 0)
        if concentration > 0.8:
            insight = BehavioralInsight(
                insight_id=f"NET_INSIGHT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}",
                entity_id=entity_id,
                insight_type='high_counterparty_concentration',
                description=f"Entity shows high counterparty concentration: {concentration:.1%}",
                statistical_significance=0.9,
                trend_direction='stable',
                time_period='current',
                supporting_data={'concentration': concentration},
                implications=[
                    "Limited counterparty diversity",
                    "May indicate shell company behavior",
                    "Requires counterparty verification"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def get_behavioral_dashboard(self) -> Dict[str, Any]:
        """Get behavioral analysis dashboard data."""
        current_time = datetime.now()
        
        # Profile statistics
        total_profiles = len(self.behavioral_profiles)
        high_risk_profiles = len([p for p in self.behavioral_profiles.values() 
                                if len(p.risk_indicators) > 3])
        
        # Anomaly statistics
        recent_anomalies = [a for a in self.behavioral_anomalies 
                          if (current_time - a.timestamp).days < 7]
        
        # Insight statistics
        recent_insights = [i for i in self.behavioral_insights 
                         if (current_time - i.timestamp).days < 7]
        
        return {
            'timestamp': current_time.isoformat(),
            'total_profiles': total_profiles,
            'high_risk_profiles': high_risk_profiles,
            'recent_anomalies': len(recent_anomalies),
            'recent_insights': len(recent_insights),
            'anomaly_types': self._get_anomaly_type_distribution(),
            'insight_types': self._get_insight_type_distribution(),
            'top_risk_entities': self._get_top_risk_entities(),
            'behavioral_trends': self._get_behavioral_trends()
        }
    
    def _get_anomaly_type_distribution(self) -> Dict[str, int]:
        """Get distribution of anomaly types."""
        distribution = {}
        for anomaly in self.behavioral_anomalies:
            anomaly_type = anomaly.anomaly_type
            distribution[anomaly_type] = distribution.get(anomaly_type, 0) + 1
        return distribution
    
    def _get_insight_type_distribution(self) -> Dict[str, int]:
        """Get distribution of insight types."""
        distribution = {}
        for insight in self.behavioral_insights:
            insight_type = insight.insight_type
            distribution[insight_type] = distribution.get(insight_type, 0) + 1
        return distribution
    
    def _get_top_risk_entities(self) -> List[Dict[str, Any]]:
        """Get top risk entities based on behavioral analysis."""
        entities = []
        
        for entity_id, profile in self.behavioral_profiles.items():
            risk_score = len(profile.risk_indicators) / 10.0  # Normalize to 0-1
            entities.append({
                'entity_id': entity_id,
                'risk_score': risk_score,
                'risk_indicators': len(profile.risk_indicators),
                'confidence': profile.confidence_score,
                'last_updated': profile.last_updated.isoformat()
            })
        
        return sorted(entities, key=lambda x: x['risk_score'], reverse=True)[:10]
    
    def _get_behavioral_trends(self) -> Dict[str, Any]:
        """Get behavioral trends across all entities."""
        if not self.behavioral_profiles:
            return {}
        
        # Calculate average features
        feature_averages = {}
        for profile in self.behavioral_profiles.values():
            for feature, value in profile.features.items():
                if feature not in feature_averages:
                    feature_averages[feature] = []
                feature_averages[feature].append(value)
        
        # Calculate statistics
        trends = {}
        for feature, values in feature_averages.items():
            trends[feature] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return trends
    
    def export_behavioral_data(self, output_dir: str = "behavioral_analysis/"):
        """Export behavioral analysis data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export profiles
        profiles_data = []
        for profile in self.behavioral_profiles.values():
            profiles_data.append({
                'entity_id': profile.entity_id,
                'entity_type': profile.entity_type,
                'profile_type': profile.profile_type,
                'features': profile.features,
                'temporal_patterns': profile.temporal_patterns,
                'transaction_patterns': profile.transaction_patterns,
                'network_behavior': profile.network_behavior,
                'risk_indicators': profile.risk_indicators,
                'confidence_score': profile.confidence_score,
                'created_at': profile.created_at.isoformat(),
                'last_updated': profile.last_updated.isoformat(),
                'sample_size': profile.sample_size
            })
        
        with open(output_path / f"behavioral_profiles_{timestamp}.json", 'w') as f:
            json.dump(profiles_data, f, indent=2)
        
        # Export anomalies
        anomalies_data = []
        for anomaly in self.behavioral_anomalies:
            anomalies_data.append({
                'anomaly_id': anomaly.anomaly_id,
                'entity_id': anomaly.entity_id,
                'anomaly_type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'deviation_score': anomaly.deviation_score,
                'affected_features': anomaly.affected_features,
                'description': anomaly.description,
                'evidence': anomaly.evidence,
                'baseline_values': anomaly.baseline_values,
                'current_values': anomaly.current_values,
                'temporal_context': anomaly.temporal_context,
                'recommended_actions': anomaly.recommended_actions,
                'timestamp': anomaly.timestamp.isoformat()
            })
        
        with open(output_path / f"behavioral_anomalies_{timestamp}.json", 'w') as f:
            json.dump(anomalies_data, f, indent=2)
        
        # Export insights
        insights_data = []
        for insight in self.behavioral_insights:
            insights_data.append({
                'insight_id': insight.insight_id,
                'entity_id': insight.entity_id,
                'insight_type': insight.insight_type,
                'description': insight.description,
                'statistical_significance': insight.statistical_significance,
                'trend_direction': insight.trend_direction,
                'time_period': insight.time_period,
                'supporting_data': insight.supporting_data,
                'implications': insight.implications,
                'timestamp': insight.timestamp.isoformat()
            })
        
        with open(output_path / f"behavioral_insights_{timestamp}.json", 'w') as f:
            json.dump(insights_data, f, indent=2)
        
        print(f"📁 Behavioral analysis data exported to {output_path}")
    
    def _load_profiles(self):
        """Load existing behavioral profiles."""
        profiles_dir = Path("behavioral_analysis")
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("behavioral_profiles_*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        profiles_data = json.load(f)
                    
                    for profile_data in profiles_data:
                        profile = BehavioralProfile(
                            entity_id=profile_data['entity_id'],
                            entity_type=profile_data['entity_type'],
                            profile_type=profile_data['profile_type'],
                            features=profile_data['features'],
                            temporal_patterns=profile_data['temporal_patterns'],
                            transaction_patterns=profile_data['transaction_patterns'],
                            network_behavior=profile_data['network_behavior'],
                            risk_indicators=profile_data['risk_indicators'],
                            confidence_score=profile_data['confidence_score'],
                            created_at=datetime.fromisoformat(profile_data['created_at']),
                            last_updated=datetime.fromisoformat(profile_data['last_updated']),
                            sample_size=profile_data['sample_size']
                        )
                        self.behavioral_profiles[profile.entity_id] = profile
                    
                    print(f"📚 Loaded {len(profiles_data)} behavioral profiles")
                    break  # Load only the most recent file
                    
                except Exception as e:
                    print(f"⚠️ Failed to load profiles from {profile_file}: {str(e)}")
    
    def get_behavioral_summary(self) -> str:
        """Generate human-readable behavioral analysis summary."""
        total_profiles = len(self.behavioral_profiles)
        total_anomalies = len(self.behavioral_anomalies)
        total_insights = len(self.behavioral_insights)
        
        high_risk_count = len([p for p in self.behavioral_profiles.values() 
                             if len(p.risk_indicators) > 3])
        
        summary = f"""
BEHAVIORAL ANALYSIS SUMMARY
===========================

Total Profiles Created: {total_profiles}
High-Risk Profiles: {high_risk_count}
Anomalies Detected: {total_anomalies}
Insights Generated: {total_insights}

Risk Distribution:
"""
        
        # Risk level distribution
        risk_levels = {'low': 0, 'medium': 0, 'high': 0}
        for profile in self.behavioral_profiles.values():
            risk_count = len(profile.risk_indicators)
            if risk_count > 5:
                risk_levels['high'] += 1
            elif risk_count > 2:
                risk_levels['medium'] += 1
            else:
                risk_levels['low'] += 1
        
        for level, count in risk_levels.items():
            percentage = (count / total_profiles * 100) if total_profiles > 0 else 0
            summary += f"  {level.title()}: {count} ({percentage:.1f}%)\n"
        
        # Recent anomaly types
        if self.behavioral_anomalies:
            summary += f"\nRecent Anomaly Types:\n"
            anomaly_types = {}
            for anomaly in self.behavioral_anomalies[-10:]:  # Last 10 anomalies
                anomaly_types[anomaly.anomaly_type] = anomaly_types.get(anomaly.anomaly_type, 0) + 1
            
            for anomaly_type, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                summary += f"  {anomaly_type.replace('_', ' ').title()}: {count}\n"
        
        summary += f"\nOverall Status: {'🚨 ANOMALIES DETECTED' if total_anomalies > 0 else '✅ NORMAL BEHAVIOR'}"
        
        return summary


def main():
    """Main function for testing the Behavioral Analysis Agent."""
    print("Testing Blue Team Behavioral Analysis Agent...")
    
    # Initialize agent
    behavioral_agent = BehavioralAnalysisAgent()
    
    # Create sample transaction data
    np.random.seed(42)
    sample_transactions = pd.DataFrame({
        'transaction_id': [f'TX_{i:06d}' for i in range(300)],
        'timestamp': pd.date_range('2024-01-01', periods=300, freq='2H'),
        'sender_id': [f'ENTITY_{i%15:03d}' for i in range(300)],
        'receiver_id': [f'ENTITY_{(i+3)%15:03d}' for i in range(300)],
        'amount': np.random.lognormal(8, 1, 300),
        'location': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney'], 300)
    })
    
    # Add some behavioral anomalies
    # High-frequency entity
    for i in range(50):
        sample_transactions.loc[len(sample_transactions)] = {
            'transaction_id': f'HIGH_FREQ_{i:03d}',
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i*0.5),
            'sender_id': 'BEHAVIORAL_001',
            'receiver_id': f'ENTITY_{i%5:03d}',
            'amount': 1000.0,
            'location': 'New York'
        }
    
    # Test profile creation
    print("\nCreating behavioral profiles...")
    entities = ['ENTITY_001', 'ENTITY_002', 'BEHAVIORAL_001']
    
    for entity_id in entities:
        profile = behavioral_agent.create_behavioral_profile(entity_id, sample_transactions)
        print(f"  Profile for {entity_id}: {len(profile.risk_indicators)} risk indicators")
    
    # Test anomaly detection
    print("\nDetecting behavioral anomalies...")
    anomalies = behavioral_agent.detect_behavioral_anomalies('BEHAVIORAL_001', sample_transactions)
    
    # Test insight generation
    print("\nGenerating behavioral insights...")
    insights = behavioral_agent.generate_behavioral_insights('BEHAVIORAL_001')
    
    # Display results
    print("\nBehavioral Analysis Summary:")
    print("=" * 50)
    print(behavioral_agent.get_behavioral_summary())
    
    # Export data
    behavioral_agent.export_behavioral_data()
    
    print("\n✅ Behavioral Analysis Agent test completed!")


if __name__ == "__main__":
    main() 