"""
Blue Team Transaction Analyst Agent

This agent performs the initial analysis of transaction data to identify
suspicious patterns and anomalies that may indicate money laundering activities.
It uses statistical analysis, machine learning, and graph theory to detect
potential criminal networks.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class TransactionAnalyst:
    """
    Analyzes transaction data to identify suspicious patterns and entities.
    
    This agent serves as the first line of defense in the Blue Team,
    using advanced analytics to flag potentially criminal activities.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Transaction Analyst.
        
        Args:
            config: Configuration dictionary for analysis parameters
        """
        self.config = config or self._get_default_config()
        self.analysis_results = {}
        self.suspicious_entities = []
        self.network_graph = None
        self.anomaly_scores = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for analysis."""
        return {
            'anomaly_detection': {
                'contamination': 0.1,
                'algorithm': 'isolation_forest'
            },
            'clustering': {
                'algorithm': 'dbscan',
                'eps': 0.5,
                'min_samples': 5
            },
            'thresholds': {
                'high_amount': 10000,
                'frequent_transactions': 10,
                'suspicious_score': 0.7,
                'structuring_threshold': 9999
            },
            'time_windows': {
                'short_term': 7,  # days
                'medium_term': 30,  # days
                'long_term': 90   # days
            }
        }
    
    def analyze_transactions(self, transactions: pd.DataFrame) -> Dict:
        """
        Perform comprehensive analysis of transaction data.
        
        Args:
            transactions: DataFrame containing transaction data
            
        Returns:
            Dictionary containing analysis results and suspicious entities
        """
        print("🔍 Starting comprehensive transaction analysis...")
        
        # Initialize analysis results
        self.analysis_results = {
            'timestamp': datetime.now(),
            'total_transactions': len(transactions),
            'analysis_methods': [],
            'suspicious_entities': [],
            'anomalies_detected': [],
            'network_analysis': {},
            'statistical_summary': {},
            'risk_assessment': {}
        }
        
        # Perform different types of analysis
        try:
            # 1. Basic statistical analysis
            print("  📊 Performing statistical analysis...")
            self._perform_statistical_analysis(transactions)
            
            # 2. Anomaly detection
            print("  🚨 Running anomaly detection...")
            self._detect_anomalies(transactions)
            
            # 3. Pattern analysis
            print("  🔄 Analyzing transaction patterns...")
            self._analyze_patterns(transactions)
            
            # 4. Network analysis
            print("  🕸️ Building and analyzing transaction network...")
            self._build_transaction_network(transactions)
            self._analyze_network()
            
            # 5. Temporal analysis
            print("  ⏰ Analyzing temporal patterns...")
            self._analyze_temporal_patterns(transactions)
            
            # 6. Entity behavior analysis
            print("  👤 Analyzing entity behaviors...")
            self._analyze_entity_behavior(transactions)
            
            # 7. Structuring detection
            print("  🏗️ Detecting structuring patterns...")
            self._detect_structuring(transactions)
            
            # 8. Generate final assessment
            print("  📋 Generating risk assessment...")
            self._generate_risk_assessment()
            
            print("✅ Transaction analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            self.analysis_results['error'] = str(e)
        
        return self.analysis_results
    
    def _perform_statistical_analysis(self, transactions: pd.DataFrame):
        """Perform basic statistical analysis of transactions."""
        stats = {
            'total_volume': transactions['amount'].sum(),
            'mean_amount': transactions['amount'].mean(),
            'median_amount': transactions['amount'].median(),
            'std_amount': transactions['amount'].std(),
            'max_amount': transactions['amount'].max(),
            'min_amount': transactions['amount'].min(),
            'unique_senders': transactions['sender_id'].nunique(),
            'unique_receivers': transactions['receiver_id'].nunique(),
            'transaction_types': transactions['transaction_type'].value_counts().to_dict(),
            'time_range': {
                'start': transactions['timestamp'].min(),
                'end': transactions['timestamp'].max(),
                'duration_days': (transactions['timestamp'].max() - transactions['timestamp'].min()).days
            }
        }
        
        self.analysis_results['statistical_summary'] = stats
        self.analysis_results['analysis_methods'].append('statistical_analysis')
    
    def _detect_anomalies(self, transactions: pd.DataFrame):
        """Detect anomalous transactions using machine learning."""
        # Prepare features for anomaly detection
        features = self._prepare_anomaly_features(transactions)
        
        if features.empty:
            return
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.config['anomaly_detection']['contamination'],
            random_state=42
        )
        
        anomaly_scores = iso_forest.fit_predict(features_scaled)
        anomaly_scores_proba = iso_forest.score_samples(features_scaled)
        
        # Identify anomalous transactions
        anomalous_indices = np.where(anomaly_scores == -1)[0]
        anomalous_transactions = transactions.iloc[anomalous_indices].copy()
        anomalous_transactions['anomaly_score'] = anomaly_scores_proba[anomalous_indices]
        
        # Store results
        self.analysis_results['anomalies_detected'] = {
            'count': len(anomalous_transactions),
            'percentage': len(anomalous_transactions) / len(transactions) * 100,
            'transactions': anomalous_transactions.to_dict('records')
        }
        
        # Add to suspicious entities
        suspicious_senders = anomalous_transactions['sender_id'].unique()
        suspicious_receivers = anomalous_transactions['receiver_id'].unique()
        
        for entity_id in np.concatenate([suspicious_senders, suspicious_receivers]):
            if entity_id not in [se['entity_id'] for se in self.suspicious_entities]:
                self.suspicious_entities.append({
                    'entity_id': entity_id,
                    'reason': 'anomalous_transaction_pattern',
                    'risk_score': 0.8,
                    'detection_method': 'isolation_forest'
                })
        
        self.analysis_results['analysis_methods'].append('anomaly_detection')
    
    def _prepare_anomaly_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        features = pd.DataFrame()
        
        # Amount-based features
        features['amount'] = transactions['amount']
        features['amount_log'] = np.log1p(transactions['amount'])
        features['is_round_amount'] = (transactions['amount'] % 100 == 0).astype(int)
        
        # Time-based features
        features['hour'] = transactions['timestamp'].dt.hour
        features['day_of_week'] = transactions['timestamp'].dt.dayofweek
        features['is_weekend'] = transactions['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        features['is_business_hours'] = transactions['timestamp'].dt.hour.between(9, 17).astype(int)
        
        # Transaction type features
        transaction_type_dummies = pd.get_dummies(transactions['transaction_type'], prefix='type')
        features = pd.concat([features, transaction_type_dummies], axis=1)
        
        # Channel features
        if 'channel' in transactions.columns:
            channel_dummies = pd.get_dummies(transactions['channel'], prefix='channel')
            features = pd.concat([features, channel_dummies], axis=1)
        
        return features.fillna(0)
    
    def _analyze_patterns(self, transactions: pd.DataFrame):
        """Analyze transaction patterns for suspicious behavior."""
        patterns = {
            'structuring_patterns': self._find_structuring_patterns(transactions),
            'rapid_movement': self._find_rapid_movement(transactions),
            'circular_transactions': self._find_circular_patterns(transactions),
            'unusual_timing': self._find_unusual_timing(transactions),
            'amount_clustering': self._find_amount_clustering(transactions)
        }
        
        self.analysis_results['patterns'] = patterns
        self.analysis_results['analysis_methods'].append('pattern_analysis')
    
    def _find_structuring_patterns(self, transactions: pd.DataFrame) -> List[Dict]:
        """Find potential structuring (smurfing) patterns."""
        structuring_patterns = []
        threshold = self.config['thresholds']['structuring_threshold']
        
        # Group by sender and look for multiple transactions just below threshold
        for sender_id in transactions['sender_id'].unique():
            sender_txs = transactions[transactions['sender_id'] == sender_id]
            
            # Check for multiple transactions just below threshold
            near_threshold = sender_txs[
                (sender_txs['amount'] > threshold * 0.8) & 
                (sender_txs['amount'] < threshold)
            ]
            
            if len(near_threshold) >= 3:  # 3 or more transactions near threshold
                structuring_patterns.append({
                    'entity_id': sender_id,
                    'pattern_type': 'structuring',
                    'transaction_count': len(near_threshold),
                    'total_amount': near_threshold['amount'].sum(),
                    'avg_amount': near_threshold['amount'].mean(),
                    'time_span': (near_threshold['timestamp'].max() - near_threshold['timestamp'].min()).days,
                    'risk_score': min(0.9, 0.5 + (len(near_threshold) * 0.1))
                })
        
        return structuring_patterns
    
    def _find_rapid_movement(self, transactions: pd.DataFrame) -> List[Dict]:
        """Find rapid movement of funds between accounts."""
        rapid_patterns = []
        
        # Look for chains of transactions where money moves quickly
        for entity_id in transactions['receiver_id'].unique():
            received = transactions[transactions['receiver_id'] == entity_id]
            sent = transactions[transactions['sender_id'] == entity_id]
            
            if len(received) > 0 and len(sent) > 0:
                # Check if money is quickly moved out after being received
                for _, rx_tx in received.iterrows():
                    quick_sends = sent[
                        (sent['timestamp'] > rx_tx['timestamp']) &
                        (sent['timestamp'] < rx_tx['timestamp'] + timedelta(days=1)) &
                        (sent['amount'] >= rx_tx['amount'] * 0.8)  # Similar amount
                    ]
                    
                    if len(quick_sends) > 0:
                        rapid_patterns.append({
                            'entity_id': entity_id,
                            'pattern_type': 'rapid_movement',
                            'received_amount': rx_tx['amount'],
                            'sent_amount': quick_sends['amount'].sum(),
                            'time_diff_hours': (quick_sends['timestamp'].min() - rx_tx['timestamp']).total_seconds() / 3600,
                            'risk_score': 0.7
                        })
        
        return rapid_patterns
    
    def _find_circular_patterns(self, transactions: pd.DataFrame) -> List[Dict]:
        """Find circular transaction patterns."""
        circular_patterns = []
        
        # Build a simple transaction graph
        edges = transactions[['sender_id', 'receiver_id', 'amount']].values
        
        # Look for simple cycles (A -> B -> A)
        for sender, receiver, amount in edges:
            reverse_txs = transactions[
                (transactions['sender_id'] == receiver) & 
                (transactions['receiver_id'] == sender)
            ]
            
            if len(reverse_txs) > 0:
                circular_patterns.append({
                    'entities': [sender, receiver],
                    'pattern_type': 'circular_transaction',
                    'forward_amount': amount,
                    'reverse_amount': reverse_txs['amount'].sum(),
                    'risk_score': 0.6
                })
        
        return circular_patterns
    
    def _find_unusual_timing(self, transactions: pd.DataFrame) -> List[Dict]:
        """Find transactions at unusual times."""
        unusual_patterns = []
        
        # Transactions outside business hours
        after_hours = transactions[
            (transactions['timestamp'].dt.hour < 6) | 
            (transactions['timestamp'].dt.hour > 22)
        ]
        
        if len(after_hours) > 0:
            # Group by entity to find those with many after-hours transactions
            entity_after_hours = after_hours.groupby('sender_id').size()
            
            for entity_id, count in entity_after_hours.items():
                if count >= 5:  # 5 or more after-hours transactions
                    unusual_patterns.append({
                        'entity_id': entity_id,
                        'pattern_type': 'unusual_timing',
                        'after_hours_count': count,
                        'risk_score': min(0.8, 0.4 + (count * 0.05))
                    })
        
        return unusual_patterns
    
    def _find_amount_clustering(self, transactions: pd.DataFrame) -> List[Dict]:
        """Find suspicious clustering of transaction amounts."""
        clustering_patterns = []
        
        # Use DBSCAN to find clusters of similar amounts
        amounts = transactions['amount'].values.reshape(-1, 1)
        
        if len(amounts) > 10:  # Need enough data for clustering
            dbscan = DBSCAN(eps=100, min_samples=5)  # Cluster amounts within $100
            clusters = dbscan.fit_predict(amounts)
            
            # Analyze each cluster
            for cluster_id in np.unique(clusters):
                if cluster_id != -1:  # Ignore noise points
                    cluster_txs = transactions[clusters == cluster_id]
                    
                    if len(cluster_txs) >= 10:  # Significant cluster
                        clustering_patterns.append({
                            'cluster_id': cluster_id,
                            'pattern_type': 'amount_clustering',
                            'transaction_count': len(cluster_txs),
                            'amount_range': [cluster_txs['amount'].min(), cluster_txs['amount'].max()],
                            'entities_involved': len(cluster_txs['sender_id'].unique()),
                            'risk_score': min(0.9, 0.3 + (len(cluster_txs) * 0.02))
                        })
        
        return clustering_patterns
    
    def _build_transaction_network(self, transactions: pd.DataFrame):
        """Build a network graph of transactions."""
        self.network_graph = nx.DiGraph()
        
        # Add edges for each transaction
        for _, row in transactions.iterrows():
            sender = row['sender_id']
            receiver = row['receiver_id']
            amount = row['amount']
            
            if self.network_graph.has_edge(sender, receiver):
                # Update existing edge
                self.network_graph[sender][receiver]['weight'] += amount
                self.network_graph[sender][receiver]['count'] += 1
            else:
                # Add new edge
                self.network_graph.add_edge(sender, receiver, weight=amount, count=1)
        
        # Add node attributes
        for node in self.network_graph.nodes():
            node_txs = transactions[
                (transactions['sender_id'] == node) | 
                (transactions['receiver_id'] == node)
            ]
            
            self.network_graph.nodes[node]['transaction_count'] = len(node_txs)
            self.network_graph.nodes[node]['total_volume'] = node_txs['amount'].sum()
    
    def _analyze_network(self):
        """Analyze the transaction network for suspicious patterns."""
        if self.network_graph is None or len(self.network_graph) == 0:
            return
        
        network_analysis = {
            'nodes': len(self.network_graph.nodes()),
            'edges': len(self.network_graph.edges()),
            'density': nx.density(self.network_graph),
            'centrality_analysis': {},
            'community_detection': {},
            'suspicious_subgraphs': []
        }
        
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.network_graph)
            closeness = nx.closeness_centrality(self.network_graph)
            pagerank = nx.pagerank(self.network_graph)
            
            # Find top central nodes
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            
            network_analysis['centrality_analysis'] = {
                'top_betweenness': top_betweenness,
                'top_closeness': top_closeness,
                'top_pagerank': top_pagerank
            }
            
            # Flag highly central nodes as suspicious
            for node, score in top_betweenness[:5]:  # Top 5 most central
                if score > 0.1:  # Threshold for suspicion
                    if node not in [se['entity_id'] for se in self.suspicious_entities]:
                        self.suspicious_entities.append({
                            'entity_id': node,
                            'reason': 'high_network_centrality',
                            'risk_score': min(0.9, score * 5),
                            'detection_method': 'network_analysis',
                            'centrality_score': score
                        })
        
        except Exception as e:
            network_analysis['centrality_error'] = str(e)
        
        # Community detection
        try:
            # Convert to undirected for community detection
            undirected = self.network_graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            
            network_analysis['community_detection'] = {
                'num_communities': len(communities),
                'modularity': nx.community.modularity(undirected, communities),
                'communities': [list(community) for community in communities]
            }
            
            # Analyze communities for suspicious patterns
            for i, community in enumerate(communities):
                if len(community) >= 5:  # Significant community
                    subgraph = self.network_graph.subgraph(community)
                    
                    # Calculate community metrics
                    internal_edges = len(subgraph.edges())
                    total_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
                    
                    if internal_edges > len(community) * 2:  # Highly connected
                        network_analysis['suspicious_subgraphs'].append({
                            'community_id': i,
                            'nodes': list(community),
                            'internal_connections': internal_edges,
                            'total_volume': total_weight,
                            'suspicion_reason': 'highly_connected_community'
                        })
        
        except Exception as e:
            network_analysis['community_error'] = str(e)
        
        self.analysis_results['network_analysis'] = network_analysis
        self.analysis_results['analysis_methods'].append('network_analysis')
    
    def _analyze_temporal_patterns(self, transactions: pd.DataFrame):
        """Analyze temporal patterns in transactions."""
        temporal_analysis = {
            'hourly_distribution': transactions.groupby(transactions['timestamp'].dt.hour)['amount'].sum().to_dict(),
            'daily_distribution': transactions.groupby(transactions['timestamp'].dt.dayofweek)['amount'].sum().to_dict(),
            'monthly_trends': transactions.groupby(transactions['timestamp'].dt.month)['amount'].sum().to_dict(),
            'burst_periods': self._find_burst_periods(transactions)
        }
        
        self.analysis_results['temporal_analysis'] = temporal_analysis
        self.analysis_results['analysis_methods'].append('temporal_analysis')
    
    def _find_burst_periods(self, transactions: pd.DataFrame) -> List[Dict]:
        """Find periods of unusually high transaction activity."""
        burst_periods = []
        
        # Group transactions by day
        daily_counts = transactions.groupby(transactions['timestamp'].dt.date).size()
        daily_volumes = transactions.groupby(transactions['timestamp'].dt.date)['amount'].sum()
        
        # Find days with unusually high activity
        count_threshold = daily_counts.mean() + 2 * daily_counts.std()
        volume_threshold = daily_volumes.mean() + 2 * daily_volumes.std()
        
        for date in daily_counts.index:
            if daily_counts[date] > count_threshold or daily_volumes[date] > volume_threshold:
                burst_periods.append({
                    'date': date.isoformat(),
                    'transaction_count': daily_counts[date],
                    'total_volume': daily_volumes[date],
                    'count_zscore': (daily_counts[date] - daily_counts.mean()) / daily_counts.std(),
                    'volume_zscore': (daily_volumes[date] - daily_volumes.mean()) / daily_volumes.std()
                })
        
        return burst_periods
    
    def _analyze_entity_behavior(self, transactions: pd.DataFrame):
        """Analyze individual entity behaviors."""
        entity_analysis = {}
        
        # Analyze each unique entity
        all_entities = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        
        for entity_id in all_entities:
            entity_txs = transactions[
                (transactions['sender_id'] == entity_id) | 
                (transactions['receiver_id'] == entity_id)
            ]
            
            sent_txs = transactions[transactions['sender_id'] == entity_id]
            received_txs = transactions[transactions['receiver_id'] == entity_id]
            
            behavior_profile = {
                'total_transactions': len(entity_txs),
                'sent_count': len(sent_txs),
                'received_count': len(received_txs),
                'sent_volume': sent_txs['amount'].sum() if len(sent_txs) > 0 else 0,
                'received_volume': received_txs['amount'].sum() if len(received_txs) > 0 else 0,
                'avg_sent_amount': sent_txs['amount'].mean() if len(sent_txs) > 0 else 0,
                'avg_received_amount': received_txs['amount'].mean() if len(received_txs) > 0 else 0,
                'activity_span_days': (entity_txs['timestamp'].max() - entity_txs['timestamp'].min()).days,
                'unique_counterparties': len(set(entity_txs['sender_id'].unique()) | set(entity_txs['receiver_id'].unique())) - 1
            }
            
            # Calculate risk indicators
            risk_indicators = []
            risk_score = 0.0
            
            # High transaction frequency
            if behavior_profile['total_transactions'] > 100:
                risk_indicators.append('high_frequency')
                risk_score += 0.2
            
            # Imbalanced send/receive ratio
            if behavior_profile['sent_count'] > 0 and behavior_profile['received_count'] > 0:
                ratio = max(behavior_profile['sent_count'], behavior_profile['received_count']) / min(behavior_profile['sent_count'], behavior_profile['received_count'])
                if ratio > 10:
                    risk_indicators.append('imbalanced_flow')
                    risk_score += 0.3
            
            # High volume relative to frequency
            if behavior_profile['total_transactions'] > 0:
                avg_tx_amount = (behavior_profile['sent_volume'] + behavior_profile['received_volume']) / behavior_profile['total_transactions']
                if avg_tx_amount > 50000:
                    risk_indicators.append('high_value_transactions')
                    risk_score += 0.2
            
            # Many counterparties
            if behavior_profile['unique_counterparties'] > 50:
                risk_indicators.append('many_counterparties')
                risk_score += 0.2
            
            behavior_profile['risk_indicators'] = risk_indicators
            behavior_profile['risk_score'] = min(risk_score, 1.0)
            
            entity_analysis[entity_id] = behavior_profile
            
            # Add to suspicious entities if high risk
            if risk_score > self.config['thresholds']['suspicious_score']:
                if entity_id not in [se['entity_id'] for se in self.suspicious_entities]:
                    self.suspicious_entities.append({
                        'entity_id': entity_id,
                        'reason': 'suspicious_behavior_profile',
                        'risk_score': risk_score,
                        'detection_method': 'entity_behavior_analysis',
                        'risk_indicators': risk_indicators
                    })
        
        self.analysis_results['entity_analysis'] = entity_analysis
        self.analysis_results['analysis_methods'].append('entity_behavior_analysis')
    
    def _detect_structuring(self, transactions: pd.DataFrame):
        """Detect potential structuring (smurfing) activities."""
        structuring_cases = []
        threshold = self.config['thresholds']['structuring_threshold']
        
        # Group by sender and analyze transaction patterns
        for sender_id in transactions['sender_id'].unique():
            sender_txs = transactions[transactions['sender_id'] == sender_id]
            
            # Look for multiple transactions just below reporting threshold
            near_threshold_txs = sender_txs[
                (sender_txs['amount'] >= threshold * 0.8) & 
                (sender_txs['amount'] <= threshold)
            ]
            
            if len(near_threshold_txs) >= 3:
                # Check if transactions are spread over short time period
                time_span = (near_threshold_txs['timestamp'].max() - near_threshold_txs['timestamp'].min()).days
                
                if time_span <= 30:  # Within 30 days
                    structuring_cases.append({
                        'entity_id': sender_id,
                        'transaction_count': len(near_threshold_txs),
                        'total_amount': near_threshold_txs['amount'].sum(),
                        'time_span_days': time_span,
                        'avg_amount': near_threshold_txs['amount'].mean(),
                        'amount_variance': near_threshold_txs['amount'].var(),
                        'risk_score': min(0.95, 0.6 + (len(near_threshold_txs) * 0.05))
                    })
                    
                    # Add to suspicious entities
                    if sender_id not in [se['entity_id'] for se in self.suspicious_entities]:
                        self.suspicious_entities.append({
                            'entity_id': sender_id,
                            'reason': 'potential_structuring',
                            'risk_score': min(0.95, 0.6 + (len(near_threshold_txs) * 0.05)),
                            'detection_method': 'structuring_detection',
                            'transaction_count': len(near_threshold_txs)
                        })
        
        self.analysis_results['structuring_analysis'] = {
            'cases_detected': len(structuring_cases),
            'cases': structuring_cases
        }
        self.analysis_results['analysis_methods'].append('structuring_detection')
    
    def _generate_risk_assessment(self):
        """Generate overall risk assessment."""
        risk_assessment = {
            'overall_risk_level': 'low',
            'suspicious_entities_count': len(self.suspicious_entities),
            'high_risk_entities': [se for se in self.suspicious_entities if se['risk_score'] > 0.8],
            'anomalies_percentage': 0,
            'network_risk_indicators': [],
            'recommendations': []
        }
        
        # Calculate overall risk level
        if len(self.suspicious_entities) > 0:
            avg_risk = sum(se['risk_score'] for se in self.suspicious_entities) / len(self.suspicious_entities)
            
            if avg_risk > 0.8:
                risk_assessment['overall_risk_level'] = 'high'
            elif avg_risk > 0.5:
                risk_assessment['overall_risk_level'] = 'medium'
        
        # Add anomalies percentage
        if 'anomalies_detected' in self.analysis_results:
            risk_assessment['anomalies_percentage'] = self.analysis_results['anomalies_detected']['percentage']
        
        # Network risk indicators
        if 'network_analysis' in self.analysis_results:
            network_analysis = self.analysis_results['network_analysis']
            if network_analysis.get('density', 0) > 0.1:
                risk_assessment['network_risk_indicators'].append('high_network_density')
            if len(network_analysis.get('suspicious_subgraphs', [])) > 0:
                risk_assessment['network_risk_indicators'].append('suspicious_communities_detected')
        
        # Generate recommendations
        if risk_assessment['overall_risk_level'] == 'high':
            risk_assessment['recommendations'].extend([
                'Immediate investigation required',
                'Review high-risk entities for potential SAR filing',
                'Enhanced monitoring recommended'
            ])
        elif risk_assessment['overall_risk_level'] == 'medium':
            risk_assessment['recommendations'].extend([
                'Further investigation recommended',
                'Monitor flagged entities closely',
                'Consider additional data sources'
            ])
        
        # Store suspicious entities in results
        self.analysis_results['suspicious_entities'] = self.suspicious_entities
        self.analysis_results['risk_assessment'] = risk_assessment
    
    def get_analysis_summary(self) -> str:
        """Generate a human-readable summary of the analysis."""
        if not self.analysis_results:
            return "No analysis has been performed yet."
        
        summary = f"""
Transaction Analysis Summary
===========================

Analysis Timestamp: {self.analysis_results['timestamp']}
Total Transactions Analyzed: {self.analysis_results['total_transactions']:,}

Analysis Methods Used:
{chr(10).join(f"  ✓ {method}" for method in self.analysis_results['analysis_methods'])}

Risk Assessment:
  Overall Risk Level: {self.analysis_results.get('risk_assessment', {}).get('overall_risk_level', 'Unknown').upper()}
  Suspicious Entities: {len(self.suspicious_entities)}
  High-Risk Entities: {len(self.analysis_results.get('risk_assessment', {}).get('high_risk_entities', []))}

Key Findings:
"""
        
        # Add anomaly detection results
        if 'anomalies_detected' in self.analysis_results:
            anomalies = self.analysis_results['anomalies_detected']
            summary += f"  • {anomalies['count']} anomalous transactions detected ({anomalies['percentage']:.2f}%)\n"
        
        # Add network analysis results
        if 'network_analysis' in self.analysis_results:
            network = self.analysis_results['network_analysis']
            summary += f"  • Network: {network['nodes']} entities, {network['edges']} connections\n"
            if 'suspicious_subgraphs' in network:
                summary += f"  • {len(network['suspicious_subgraphs'])} suspicious communities identified\n"
        
        # Add structuring detection results
        if 'structuring_analysis' in self.analysis_results:
            structuring = self.analysis_results['structuring_analysis']
            summary += f"  • {structuring['cases_detected']} potential structuring cases detected\n"
        
        # Add top suspicious entities
        if len(self.suspicious_entities) > 0:
            summary += "\nTop Suspicious Entities:\n"
            sorted_entities = sorted(self.suspicious_entities, key=lambda x: x['risk_score'], reverse=True)
            for entity in sorted_entities[:5]:
                summary += f"  • {entity['entity_id']} (Risk: {entity['risk_score']:.2f}, Reason: {entity['reason']})\n"
        
        # Add recommendations
        recommendations = self.analysis_results.get('risk_assessment', {}).get('recommendations', [])
        if recommendations:
            summary += f"\nRecommendations:\n"
            for rec in recommendations:
                summary += f"  • {rec}\n"
        
        return summary


def main():
    """Main function for testing the Transaction Analyst."""
    print("Testing Blue Team Transaction Analyst...")
    
    # This would normally receive data from the Red Team test
    # For testing, we'll create some sample data
    sample_data = pd.DataFrame({
        'transaction_id': [f'TX_{i:06d}' for i in range(100)],
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'sender_id': [f'SENDER_{i%10:03d}' for i in range(100)],
        'receiver_id': [f'RECEIVER_{i%8:03d}' for i in range(100)],
        'amount': np.random.lognormal(8, 1, 100),
        'transaction_type': np.random.choice(['transfer', 'payment', 'deposit'], 100)
    })
    
    # Add some suspicious patterns
    # Structuring pattern
    for i in range(5):
        sample_data.loc[len(sample_data)] = {
            'transaction_id': f'STRUCT_{i:03d}',
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
            'sender_id': 'SUSPICIOUS_001',
            'receiver_id': f'RECEIVER_{i:03d}',
            'amount': 9500 + np.random.randint(-100, 100),
            'transaction_type': 'transfer'
        }
    
    # Initialize analyst
    analyst = TransactionAnalyst()
    
    # Perform analysis
    print("\nPerforming comprehensive analysis...")
    results = analyst.analyze_transactions(sample_data)
    
    # Display results
    print("\nAnalysis Results:")
    print("=" * 50)
    print(analyst.get_analysis_summary())


if __name__ == "__main__":
    main() 