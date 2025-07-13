"""
Blue Team Risk Assessment Agent

This agent performs dynamic risk analysis and advanced scoring for entities
and transactions in the AML-FT system. It provides real-time risk scoring,
adaptive risk models, and risk-based alerting.
"""

import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
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
class RiskScore:
    """Data class for risk scores."""
    entity_id: str
    entity_type: str
    risk_score: float  # 0.0 to 1.0
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    risk_factors: List[str]
    confidence: float
    timestamp: datetime
    model_version: str
    expires_at: Optional[datetime] = None


@dataclass
class RiskAlert:
    """Data class for risk alerts."""
    alert_id: str
    entity_id: str
    alert_type: str
    risk_score: float
    previous_score: float
    risk_change: float
    description: str
    recommended_actions: List[str]
    timestamp: datetime
    severity: str


@dataclass
class RiskModel:
    """Data class for risk models."""
    model_id: str
    model_type: str
    model_version: str
    features: List[str]
    performance_metrics: Dict[str, float]
    created_at: datetime
    last_updated: datetime
    is_active: bool = True


class RiskAssessmentAgent:
    """
    Performs dynamic risk analysis and scoring for AML-FT operations.
    
    This agent provides real-time risk scoring, adaptive risk models,
    risk-based alerting, and comprehensive risk analytics.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Risk Assessment Agent."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.risk_models = {}
        self.risk_scores = {}
        self.risk_alerts = []
        self.risk_history = {}
        self.feature_importance = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Load existing models
        self._load_models()
        
        # Initialize risk parameters
        self.risk_thresholds = self._get_risk_thresholds()
        self.risk_factors = self._get_risk_factors()
        
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
                'temperature': 0.2,
                'max_tokens': 2000
            },
            'risk_assessment': {
                'update_frequency': 'hourly',
                'score_decay_rate': 0.95,
                'alert_threshold': 0.7,
                'model_retrain_threshold': 0.1,
                'feature_selection_method': 'importance'
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for risk analysis."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def _get_risk_thresholds(self) -> Dict[str, float]:
        """Get risk level thresholds."""
        return {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
    
    def _get_risk_factors(self) -> Dict[str, Dict]:
        """Get risk factors and their weights."""
        return {
            'transaction_velocity': {
                'weight': 0.15,
                'description': 'Frequency of transactions',
                'calculation': 'transactions_per_hour'
            },
            'amount_patterns': {
                'weight': 0.20,
                'description': 'Unusual amount patterns',
                'calculation': 'amount_variance_analysis'
            },
            'geographic_risk': {
                'weight': 0.10,
                'description': 'Geographic risk factors',
                'calculation': 'location_risk_score'
            },
            'network_centrality': {
                'weight': 0.15,
                'description': 'Position in transaction network',
                'calculation': 'centrality_measures'
            },
            'temporal_patterns': {
                'weight': 0.10,
                'description': 'Time-based transaction patterns',
                'calculation': 'temporal_analysis'
            },
            'counterparty_risk': {
                'weight': 0.15,
                'description': 'Risk of transaction counterparties',
                'calculation': 'counterparty_analysis'
            },
            'historical_behavior': {
                'weight': 0.10,
                'description': 'Historical behavior patterns',
                'calculation': 'behavior_analysis'
            },
            'external_intelligence': {
                'weight': 0.05,
                'description': 'External risk intelligence',
                'calculation': 'osint_risk_score'
            }
        }
    
    def calculate_entity_risk(self, entity_id: str, transactions: pd.DataFrame,
                            entity_data: Dict = None, osint_data: Dict = None) -> RiskScore:
        """
        Calculate comprehensive risk score for an entity.
        
        Args:
            entity_id: ID of the entity to assess
            transactions: DataFrame of transactions involving the entity
            entity_data: Additional entity information
            osint_data: OSINT intelligence data
            
        Returns:
            RiskScore object containing the assessment
        """
        print(f"📊 Calculating risk score for entity: {entity_id}")
        
        # Filter transactions for this entity
        entity_transactions = transactions[
            (transactions['sender_id'] == entity_id) | 
            (transactions['receiver_id'] == entity_id)
        ].copy()
        
        if entity_transactions.empty:
            return self._create_default_risk_score(entity_id)
        
        # Calculate individual risk factors
        risk_factors = {}
        risk_explanations = []
        
        # Transaction velocity risk
        velocity_score, velocity_explanation = self._calculate_velocity_risk(entity_transactions)
        risk_factors['transaction_velocity'] = velocity_score
        risk_explanations.extend(velocity_explanation)
        
        # Amount pattern risk
        amount_score, amount_explanation = self._calculate_amount_risk(entity_transactions)
        risk_factors['amount_patterns'] = amount_score
        risk_explanations.extend(amount_explanation)
        
        # Geographic risk
        geo_score, geo_explanation = self._calculate_geographic_risk(entity_transactions)
        risk_factors['geographic_risk'] = geo_score
        risk_explanations.extend(geo_explanation)
        
        # Network centrality risk
        network_score, network_explanation = self._calculate_network_risk(entity_id, transactions)
        risk_factors['network_centrality'] = network_score
        risk_explanations.extend(network_explanation)
        
        # Temporal pattern risk
        temporal_score, temporal_explanation = self._calculate_temporal_risk(entity_transactions)
        risk_factors['temporal_patterns'] = temporal_score
        risk_explanations.extend(temporal_explanation)
        
        # Counterparty risk
        counterparty_score, counterparty_explanation = self._calculate_counterparty_risk(entity_id, transactions)
        risk_factors['counterparty_risk'] = counterparty_score
        risk_explanations.extend(counterparty_explanation)
        
        # Historical behavior risk
        historical_score, historical_explanation = self._calculate_historical_risk(entity_id)
        risk_factors['historical_behavior'] = historical_score
        risk_explanations.extend(historical_explanation)
        
        # External intelligence risk
        external_score, external_explanation = self._calculate_external_risk(entity_id, osint_data)
        risk_factors['external_intelligence'] = external_score
        risk_explanations.extend(external_explanation)
        
        # Calculate weighted risk score
        total_score = 0.0
        for factor, score in risk_factors.items():
            weight = self.risk_factors[factor]['weight']
            total_score += score * weight
        
        # Apply ML model if available
        if 'entity_risk_model' in self.risk_models and HAS_SKLEARN:
            ml_score = self._apply_ml_model(entity_id, entity_transactions, risk_factors)
            total_score = 0.7 * total_score + 0.3 * ml_score  # Blend scores
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(risk_factors, len(entity_transactions))
        
        # Create risk score object
        risk_score = RiskScore(
            entity_id=entity_id,
            entity_type=entity_data.get('entity_type', 'unknown') if entity_data else 'unknown',
            risk_score=total_score,
            risk_level=risk_level,
            risk_factors=risk_explanations,
            confidence=confidence,
            timestamp=datetime.now(),
            model_version='v1.0',
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        # Store risk score
        self.risk_scores[entity_id] = risk_score
        
        # Update risk history
        if entity_id not in self.risk_history:
            self.risk_history[entity_id] = []
        self.risk_history[entity_id].append(risk_score)
        
        # Check for risk alerts
        self._check_risk_alerts(entity_id, risk_score)
        
        print(f"✅ Risk score calculated: {total_score:.3f} ({risk_level})")
        return risk_score
    
    def _calculate_velocity_risk(self, transactions: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate transaction velocity risk."""
        explanations = []
        
        if transactions.empty:
            return 0.0, explanations
        
        # Calculate transactions per hour
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        time_span = (transactions['timestamp'].max() - transactions['timestamp'].min()).total_seconds() / 3600
        
        if time_span == 0:
            txn_per_hour = len(transactions)
        else:
            txn_per_hour = len(transactions) / max(time_span, 1)
        
        # Risk scoring based on velocity
        if txn_per_hour > 10:
            score = 0.9
            explanations.append(f"Very high transaction velocity: {txn_per_hour:.1f} txn/hour")
        elif txn_per_hour > 5:
            score = 0.7
            explanations.append(f"High transaction velocity: {txn_per_hour:.1f} txn/hour")
        elif txn_per_hour > 2:
            score = 0.4
            explanations.append(f"Moderate transaction velocity: {txn_per_hour:.1f} txn/hour")
        else:
            score = 0.1
        
        return score, explanations
    
    def _calculate_amount_risk(self, transactions: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate amount pattern risk."""
        explanations = []
        
        if transactions.empty:
            return 0.0, explanations
        
        amounts = transactions['amount'].values
        score = 0.0
        
        # Check for round amounts
        round_amounts = np.sum(amounts % 1000 == 0) / len(amounts)
        if round_amounts > 0.5:
            score += 0.3
            explanations.append(f"High proportion of round amounts: {round_amounts:.1%}")
        
        # Check for amounts just below thresholds
        threshold_amounts = np.sum((amounts >= 9000) & (amounts < 10000)) / len(amounts)
        if threshold_amounts > 0.2:
            score += 0.4
            explanations.append(f"Transactions near reporting thresholds: {threshold_amounts:.1%}")
        
        # Check for unusual amount variance
        cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
        if cv > 2.0:
            score += 0.2
            explanations.append(f"High amount variance (CV: {cv:.2f})")
        elif cv < 0.1:
            score += 0.3
            explanations.append(f"Suspiciously low amount variance (CV: {cv:.2f})")
        
        return min(score, 1.0), explanations
    
    def _calculate_geographic_risk(self, transactions: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate geographic risk."""
        explanations = []
        
        if 'location' not in transactions.columns:
            return 0.0, explanations
        
        # High-risk locations (simulated)
        high_risk_locations = ['Unknown', 'Offshore', 'High-Risk Country']
        
        locations = transactions['location'].value_counts()
        total_transactions = len(transactions)
        
        score = 0.0
        
        for location in locations.index:
            if location in high_risk_locations:
                proportion = locations[location] / total_transactions
                score += proportion * 0.8
                explanations.append(f"Transactions from high-risk location: {location} ({proportion:.1%})")
        
        # Check for geographic concentration
        if len(locations) == 1 and locations.iloc[0] > 10:
            score += 0.2
            explanations.append(f"High geographic concentration: {locations.index[0]}")
        
        return min(score, 1.0), explanations
    
    def _calculate_network_risk(self, entity_id: str, all_transactions: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate network centrality risk."""
        explanations = []
        
        try:
            import networkx as nx
            
            # Build transaction network
            G = nx.DiGraph()
            
            for _, txn in all_transactions.iterrows():
                G.add_edge(txn['sender_id'], txn['receiver_id'], weight=txn['amount'])
            
            if entity_id not in G:
                return 0.0, explanations
            
            # Calculate centrality measures
            try:
                betweenness = nx.betweenness_centrality(G)
                closeness = nx.closeness_centrality(G)
                degree = dict(G.degree())
                
                entity_betweenness = betweenness.get(entity_id, 0)
                entity_closeness = closeness.get(entity_id, 0)
                entity_degree = degree.get(entity_id, 0)
                
                # Normalize scores
                max_degree = max(degree.values()) if degree else 1
                normalized_degree = entity_degree / max_degree
                
                # Calculate network risk score
                score = (entity_betweenness * 0.4 + entity_closeness * 0.3 + normalized_degree * 0.3)
                
                if score > 0.7:
                    explanations.append(f"High network centrality (score: {score:.3f})")
                elif score > 0.4:
                    explanations.append(f"Moderate network centrality (score: {score:.3f})")
                
                return score, explanations
                
            except:
                return 0.0, explanations
                
        except ImportError:
            return 0.0, explanations
    
    def _calculate_temporal_risk(self, transactions: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate temporal pattern risk."""
        explanations = []
        
        if transactions.empty:
            return 0.0, explanations
        
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions['hour'] = transactions['timestamp'].dt.hour
        transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
        
        score = 0.0
        
        # Check for unusual hours
        unusual_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        unusual_hour_txns = transactions[transactions['hour'].isin(unusual_hours)]
        
        if len(unusual_hour_txns) > 0:
            proportion = len(unusual_hour_txns) / len(transactions)
            if proportion > 0.3:
                score += 0.4
                explanations.append(f"High proportion of unusual hour transactions: {proportion:.1%}")
        
        # Check for weekend activity
        weekend_txns = transactions[transactions['day_of_week'].isin([5, 6])]
        if len(weekend_txns) > 0:
            proportion = len(weekend_txns) / len(transactions)
            if proportion > 0.5:
                score += 0.3
                explanations.append(f"High weekend activity: {proportion:.1%}")
        
        # Check for time clustering
        time_diffs = transactions['timestamp'].diff().dt.total_seconds()
        rapid_transactions = np.sum(time_diffs < 300) / len(time_diffs)  # < 5 minutes
        
        if rapid_transactions > 0.3:
            score += 0.3
            explanations.append(f"High proportion of rapid transactions: {rapid_transactions:.1%}")
        
        return min(score, 1.0), explanations
    
    def _calculate_counterparty_risk(self, entity_id: str, all_transactions: pd.DataFrame) -> Tuple[float, List[str]]:
        """Calculate counterparty risk."""
        explanations = []
        
        # Get counterparties
        sent_to = all_transactions[all_transactions['sender_id'] == entity_id]['receiver_id'].unique()
        received_from = all_transactions[all_transactions['receiver_id'] == entity_id]['sender_id'].unique()
        
        all_counterparties = set(sent_to) | set(received_from)
        
        if not all_counterparties:
            return 0.0, explanations
        
        # Check for high-risk counterparties (simulated)
        high_risk_entities = ['SHELL_', 'MULE_', 'CRIMINAL_', 'SUSPICIOUS_']
        
        high_risk_counterparties = [
            cp for cp in all_counterparties 
            if any(cp.startswith(prefix) for prefix in high_risk_entities)
        ]
        
        score = 0.0
        
        if high_risk_counterparties:
            risk_proportion = len(high_risk_counterparties) / len(all_counterparties)
            score = risk_proportion * 0.8
            explanations.append(f"High-risk counterparties: {len(high_risk_counterparties)}/{len(all_counterparties)}")
        
        # Check for counterparty concentration
        counterparty_counts = {}
        for _, txn in all_transactions.iterrows():
            if txn['sender_id'] == entity_id:
                counterparty_counts[txn['receiver_id']] = counterparty_counts.get(txn['receiver_id'], 0) + 1
            elif txn['receiver_id'] == entity_id:
                counterparty_counts[txn['sender_id']] = counterparty_counts.get(txn['sender_id'], 0) + 1
        
        if counterparty_counts:
            max_concentration = max(counterparty_counts.values()) / sum(counterparty_counts.values())
            if max_concentration > 0.7:
                score += 0.2
                explanations.append(f"High counterparty concentration: {max_concentration:.1%}")
        
        return min(score, 1.0), explanations
    
    def _calculate_historical_risk(self, entity_id: str) -> Tuple[float, List[str]]:
        """Calculate historical behavior risk."""
        explanations = []
        
        if entity_id not in self.risk_history:
            return 0.0, explanations
        
        history = self.risk_history[entity_id]
        
        if len(history) < 2:
            return 0.0, explanations
        
        # Calculate risk trend
        recent_scores = [score.risk_score for score in history[-5:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        score = 0.0
        
        if trend > 0.1:
            score += 0.4
            explanations.append(f"Increasing risk trend: {trend:.3f}")
        
        # Check for risk volatility
        volatility = np.std(recent_scores)
        if volatility > 0.2:
            score += 0.2
            explanations.append(f"High risk volatility: {volatility:.3f}")
        
        # Check for consistently high risk
        avg_risk = np.mean(recent_scores)
        if avg_risk > 0.7:
            score += 0.3
            explanations.append(f"Consistently high risk: {avg_risk:.3f}")
        
        return min(score, 1.0), explanations
    
    def _calculate_external_risk(self, entity_id: str, osint_data: Dict = None) -> Tuple[float, List[str]]:
        """Calculate external intelligence risk."""
        explanations = []
        
        if not osint_data:
            return 0.0, explanations
        
        score = 0.0
        
        # Check OSINT findings
        entity_osint = osint_data.get(entity_id, [])
        
        for finding in entity_osint:
            if hasattr(finding, 'relevance_score'):
                if finding.relevance_score > 0.8:
                    score += 0.3
                    explanations.append(f"High-relevance OSINT finding: {finding.information_type}")
                elif finding.relevance_score > 0.6:
                    score += 0.2
                    explanations.append(f"Medium-relevance OSINT finding: {finding.information_type}")
        
        return min(score, 1.0), explanations
    
    def _apply_ml_model(self, entity_id: str, transactions: pd.DataFrame, 
                       risk_factors: Dict[str, float]) -> float:
        """Apply machine learning model for risk scoring."""
        if not HAS_SKLEARN or 'entity_risk_model' not in self.risk_models:
            return 0.0
        
        try:
            model = self.risk_models['entity_risk_model']
            
            # Prepare features
            features = []
            for factor in model.features:
                features.append(risk_factors.get(factor, 0.0))
            
            # Add transaction-based features
            if not transactions.empty:
                features.extend([
                    len(transactions),  # Transaction count
                    transactions['amount'].sum(),  # Total amount
                    transactions['amount'].mean(),  # Average amount
                    transactions['amount'].std(),  # Amount standard deviation
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Predict risk score
            features_array = np.array(features).reshape(1, -1)
            
            if hasattr(model, 'predict_proba'):
                risk_score = model.predict_proba(features_array)[0][1]  # Probability of high risk
            else:
                risk_score = model.predict(features_array)[0]
            
            return min(max(risk_score, 0.0), 1.0)
            
        except Exception as e:
            print(f"⚠️ ML model application failed: {str(e)}")
            return 0.0
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        if score >= self.risk_thresholds['critical']:
            return 'critical'
        elif score >= self.risk_thresholds['high']:
            return 'high'
        elif score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, risk_factors: Dict[str, float], 
                            transaction_count: int) -> float:
        """Calculate confidence in risk assessment."""
        base_confidence = 0.5
        
        # Increase confidence with more data
        data_confidence = min(transaction_count / 50, 0.3)
        
        # Increase confidence with consistent risk factors
        factor_values = list(risk_factors.values())
        if factor_values:
            factor_consistency = 1.0 - np.std(factor_values)
            consistency_confidence = factor_consistency * 0.2
        else:
            consistency_confidence = 0.0
        
        return min(base_confidence + data_confidence + consistency_confidence, 1.0)
    
    def _create_default_risk_score(self, entity_id: str) -> RiskScore:
        """Create default risk score for entities with no data."""
        return RiskScore(
            entity_id=entity_id,
            entity_type='unknown',
            risk_score=0.0,
            risk_level='low',
            risk_factors=['No transaction data available'],
            confidence=0.1,
            timestamp=datetime.now(),
            model_version='v1.0'
        )
    
    def _check_risk_alerts(self, entity_id: str, current_score: RiskScore):
        """Check for risk alerts based on score changes."""
        if entity_id not in self.risk_history or len(self.risk_history[entity_id]) < 2:
            return
        
        previous_score = self.risk_history[entity_id][-2]
        risk_change = current_score.risk_score - previous_score.risk_score
        
        # Generate alert if significant risk change
        if abs(risk_change) > 0.2:
            alert_type = "risk_increase" if risk_change > 0 else "risk_decrease"
            severity = "high" if abs(risk_change) > 0.4 else "medium"
            
            alert = RiskAlert(
                alert_id=f"RISK_ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}",
                entity_id=entity_id,
                alert_type=alert_type,
                risk_score=current_score.risk_score,
                previous_score=previous_score.risk_score,
                risk_change=risk_change,
                description=f"Significant risk change detected: {risk_change:+.3f}",
                recommended_actions=self._generate_risk_recommendations(current_score),
                timestamp=datetime.now(),
                severity=severity
            )
            
            self.risk_alerts.append(alert)
    
    def _generate_risk_recommendations(self, risk_score: RiskScore) -> List[str]:
        """Generate recommendations based on risk score."""
        recommendations = []
        
        if risk_score.risk_level == 'critical':
            recommendations.extend([
                "Immediate investigation required",
                "Freeze account if legally permissible",
                "File suspicious activity report",
                "Enhanced monitoring"
            ])
        elif risk_score.risk_level == 'high':
            recommendations.extend([
                "Conduct enhanced due diligence",
                "Increase monitoring frequency",
                "Review transaction patterns",
                "Consider filing SAR"
            ])
        elif risk_score.risk_level == 'medium':
            recommendations.extend([
                "Increased monitoring",
                "Review counterparty relationships",
                "Document risk assessment"
            ])
        else:
            recommendations.append("Continue standard monitoring")
        
        return recommendations
    
    def batch_risk_assessment(self, entities: List[str], transactions: pd.DataFrame,
                            entity_data: Dict = None, osint_data: Dict = None) -> Dict[str, RiskScore]:
        """Perform batch risk assessment for multiple entities."""
        print(f"📊 Performing batch risk assessment for {len(entities)} entities...")
        
        risk_scores = {}
        
        for entity_id in entities:
            try:
                risk_score = self.calculate_entity_risk(
                    entity_id, transactions, 
                    entity_data.get(entity_id) if entity_data else None,
                    osint_data
                )
                risk_scores[entity_id] = risk_score
            except Exception as e:
                print(f"⚠️ Error assessing {entity_id}: {str(e)}")
                risk_scores[entity_id] = self._create_default_risk_score(entity_id)
        
        print(f"✅ Batch assessment completed")
        return risk_scores
    
    def train_risk_model(self, training_data: pd.DataFrame, labels: pd.Series):
        """Train machine learning model for risk assessment."""
        if not HAS_SKLEARN:
            print("⚠️ scikit-learn not available for model training")
            return
        
        print("🤖 Training risk assessment model...")
        
        # Prepare features
        features = []
        feature_names = []
        
        for _, row in training_data.iterrows():
            # Extract risk factors (simplified)
            row_features = [
                row.get('transaction_count', 0),
                row.get('total_amount', 0),
                row.get('avg_amount', 0),
                row.get('amount_std', 0),
                row.get('velocity_score', 0),
                row.get('amount_risk_score', 0),
                row.get('network_score', 0),
                row.get('temporal_score', 0)
            ]
            features.append(row_features)
        
        feature_names = [
            'transaction_count', 'total_amount', 'avg_amount', 'amount_std',
            'velocity_score', 'amount_risk_score', 'network_score', 'temporal_score'
        ]
        
        # Split data
        X = np.array(features)
        y = labels.values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Store model
        risk_model = RiskModel(
            model_id="entity_risk_model",
            model_type="gradient_boosting",
            model_version="v1.0",
            features=feature_names,
            performance_metrics={'auc': auc_score},
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.risk_models['entity_risk_model'] = model
        
        # Store feature importance
        self.feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        print(f"✅ Model trained successfully (AUC: {auc_score:.3f})")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get risk assessment dashboard data."""
        current_time = datetime.now()
        
        # Risk distribution
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for risk_score in self.risk_scores.values():
            risk_distribution[risk_score.risk_level] += 1
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.risk_alerts
            if (current_time - alert.timestamp).total_seconds() < 86400  # Last 24 hours
        ]
        
        # High-risk entities
        high_risk_entities = [
            {'entity_id': entity_id, 'risk_score': score.risk_score, 'risk_level': score.risk_level}
            for entity_id, score in self.risk_scores.items()
            if score.risk_level in ['high', 'critical']
        ]
        
        return {
            'timestamp': current_time.isoformat(),
            'total_entities_assessed': len(self.risk_scores),
            'risk_distribution': risk_distribution,
            'recent_alerts': len(recent_alerts),
            'high_risk_entities': len(high_risk_entities),
            'models_active': len([m for m in self.risk_models.values() if hasattr(m, 'is_active') and m.is_active]),
            'average_risk_score': np.mean([score.risk_score for score in self.risk_scores.values()]) if self.risk_scores else 0.0,
            'feature_importance': self.feature_importance,
            'recent_risk_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'entity_id': alert.entity_id,
                    'alert_type': alert.alert_type,
                    'risk_score': alert.risk_score,
                    'risk_change': alert.risk_change,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def _load_models(self):
        """Load existing risk models."""
        models_dir = Path("risk_models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pkl"):
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                        self.risk_models[model_file.stem] = model
                    print(f"📚 Loaded model: {model_file.stem}")
                except Exception as e:
                    print(f"⚠️ Failed to load model {model_file}: {str(e)}")
    
    def save_models(self):
        """Save risk models to disk."""
        models_dir = Path("risk_models")
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.risk_models.items():
            model_file = models_dir / f"{model_name}.pkl"
            try:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"💾 Saved model: {model_name}")
            except Exception as e:
                print(f"⚠️ Failed to save model {model_name}: {str(e)}")
    
    def export_risk_data(self, output_dir: str = "risk_assessments/"):
        """Export risk assessment data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export risk scores
        risk_scores_data = [
            {
                'entity_id': score.entity_id,
                'entity_type': score.entity_type,
                'risk_score': score.risk_score,
                'risk_level': score.risk_level,
                'risk_factors': score.risk_factors,
                'confidence': score.confidence,
                'timestamp': score.timestamp.isoformat(),
                'model_version': score.model_version
            }
            for score in self.risk_scores.values()
        ]
        
        with open(output_path / f"risk_scores_{timestamp}.json", 'w') as f:
            json.dump(risk_scores_data, f, indent=2)
        
        # Export risk alerts
        alerts_data = [
            {
                'alert_id': alert.alert_id,
                'entity_id': alert.entity_id,
                'alert_type': alert.alert_type,
                'risk_score': alert.risk_score,
                'previous_score': alert.previous_score,
                'risk_change': alert.risk_change,
                'description': alert.description,
                'recommended_actions': alert.recommended_actions,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity
            }
            for alert in self.risk_alerts
        ]
        
        with open(output_path / f"risk_alerts_{timestamp}.json", 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        print(f"📁 Risk assessment data exported to {output_path}")
    
    def get_risk_summary(self) -> str:
        """Generate human-readable risk assessment summary."""
        total_entities = len(self.risk_scores)
        high_risk_count = len([s for s in self.risk_scores.values() if s.risk_level in ['high', 'critical']])
        
        summary = f"""
RISK ASSESSMENT SUMMARY
======================

Total Entities Assessed: {total_entities}
High/Critical Risk Entities: {high_risk_count}
Risk Alerts Generated: {len(self.risk_alerts)}

Risk Distribution:
"""
        
        # Risk level distribution
        risk_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for score in self.risk_scores.values():
            risk_levels[score.risk_level] += 1
        
        for level, count in risk_levels.items():
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            summary += f"  {level.title()}: {count} ({percentage:.1f}%)\n"
        
        if self.feature_importance:
            summary += f"\nTop Risk Factors:\n"
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                summary += f"  {feature}: {importance:.3f}\n"
        
        summary += f"\nOverall Risk Status: {'🚨 HIGH RISK' if high_risk_count > 0 else '✅ NORMAL'}"
        
        return summary


def main():
    """Main function for testing the Risk Assessment Agent."""
    print("Testing Blue Team Risk Assessment Agent...")
    
    # Initialize agent
    risk_agent = RiskAssessmentAgent()
    
    # Create sample transaction data
    np.random.seed(42)
    sample_transactions = pd.DataFrame({
        'transaction_id': [f'TX_{i:06d}' for i in range(200)],
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
        'sender_id': [f'ENTITY_{i%20:03d}' for i in range(200)],
        'receiver_id': [f'ENTITY_{(i+5)%20:03d}' for i in range(200)],
        'amount': np.random.lognormal(8, 1, 200),
        'location': np.random.choice(['New York', 'London', 'Unknown', 'Tokyo'], 200)
    })
    
    # Add some high-risk patterns
    # High velocity entity
    for i in range(10):
        sample_transactions.loc[len(sample_transactions)] = {
            'transaction_id': f'HIGH_VEL_{i:03d}',
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i*10),
            'sender_id': 'SUSPICIOUS_001',
            'receiver_id': f'ENTITY_{i:03d}',
            'amount': 9500.0,
            'location': 'Unknown'
        }
    
    # Test individual risk assessment
    print("\nTesting individual risk assessment...")
    risk_score = risk_agent.calculate_entity_risk('SUSPICIOUS_001', sample_transactions)
    
    print(f"Risk Score: {risk_score.risk_score:.3f} ({risk_score.risk_level})")
    print(f"Risk Factors: {risk_score.risk_factors}")
    
    # Test batch assessment
    print("\nTesting batch risk assessment...")
    entities = ['ENTITY_001', 'ENTITY_002', 'SUSPICIOUS_001']
    batch_scores = risk_agent.batch_risk_assessment(entities, sample_transactions)
    
    # Display results
    print("\nRisk Assessment Summary:")
    print("=" * 50)
    print(risk_agent.get_risk_summary())
    
    # Export data
    risk_agent.export_risk_data()
    risk_agent.save_models()
    
    print("\n✅ Risk Assessment Agent test completed!")


if __name__ == "__main__":
    main() 