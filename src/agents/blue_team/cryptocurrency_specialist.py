"""
Blue Team Cryptocurrency Specialist Agent

This agent specializes in analyzing cryptocurrency transactions for money laundering
detection. It tracks blockchain transactions, identifies mixing services, analyzes
privacy coins, and provides crypto-specific risk assessments.
"""

import json
import yaml
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

# LLM imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

# Crypto analysis imports (simulated)
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class CryptoTransaction:
    """Data class for cryptocurrency transactions."""
    tx_hash: str
    blockchain: str
    from_address: str
    to_address: str
    amount: float
    currency: str
    timestamp: datetime
    block_height: int
    confirmations: int
    fee: float
    risk_score: float = 0.0
    risk_factors: List[str] = None


@dataclass
class CryptoAddress:
    """Data class for cryptocurrency addresses."""
    address: str
    blockchain: str
    address_type: str  # 'wallet', 'exchange', 'mixer', 'darknet'
    risk_level: str
    total_received: float
    total_sent: float
    transaction_count: int
    first_seen: datetime
    last_seen: datetime
    labels: List[str] = None
    cluster_id: Optional[str] = None


@dataclass
class CryptoAlert:
    """Data class for cryptocurrency alerts."""
    alert_id: str
    alert_type: str
    blockchain: str
    addresses_involved: List[str]
    transactions_involved: List[str]
    description: str
    risk_score: float
    evidence: List[str]
    recommended_actions: List[str]
    timestamp: datetime
    severity: str


@dataclass
class BlockchainAnalysis:
    """Data class for blockchain analysis results."""
    blockchain: str
    analysis_type: str
    entities_analyzed: int
    transactions_analyzed: int
    clusters_identified: int
    mixing_services_detected: int
    high_risk_addresses: int
    analysis_timestamp: datetime
    summary: str


class CryptocurrencySpecialist:
    """
    Specializes in cryptocurrency transaction analysis for AML-FT.
    
    This agent provides advanced blockchain analysis, crypto-specific
    risk assessment, mixing service detection, and privacy coin analysis.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Cryptocurrency Specialist."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.crypto_transactions = []
        self.crypto_addresses = {}
        self.crypto_alerts = []
        self.blockchain_graphs = {}
        self.mixing_services = self._load_mixing_services()
        self.exchange_addresses = self._load_exchange_addresses()
        self.privacy_coins = self._load_privacy_coins()
        self.risk_patterns = self._load_risk_patterns()
        
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
                'max_tokens': 3000
            },
            'crypto_analysis': {
                'supported_blockchains': ['bitcoin', 'ethereum', 'monero', 'zcash'],
                'clustering_enabled': True,
                'mixing_detection_enabled': True,
                'privacy_coin_analysis_enabled': True,
                'risk_threshold': 0.7,
                'analysis_depth': 3  # hops in transaction graph
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for crypto analysis."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def _load_mixing_services(self) -> Dict[str, Dict]:
        """Load known mixing services and their characteristics."""
        return {
            'bitcoin': {
                'tornado_cash': {
                    'addresses': ['1TornadoCash...', '1Mixer123...'],
                    'risk_level': 'high',
                    'description': 'Decentralized mixing service'
                },
                'wasabi_wallet': {
                    'addresses': ['1Wasabi...', '1CoinJoin...'],
                    'risk_level': 'medium',
                    'description': 'CoinJoin implementation'
                },
                'samourai_whirlpool': {
                    'addresses': ['1Samourai...', '1Whirlpool...'],
                    'risk_level': 'medium',
                    'description': 'Mobile wallet mixing'
                }
            },
            'ethereum': {
                'tornado_cash': {
                    'addresses': ['0x12345...', '0x67890...'],
                    'risk_level': 'high',
                    'description': 'Ethereum mixing protocol'
                },
                'aztec_protocol': {
                    'addresses': ['0xAztec...', '0xPrivacy...'],
                    'risk_level': 'high',
                    'description': 'Privacy protocol'
                }
            },
            'monero': {
                'built_in_mixing': {
                    'addresses': ['all'],
                    'risk_level': 'high',
                    'description': 'Built-in privacy features'
                }
            }
        }
    
    def _load_exchange_addresses(self) -> Dict[str, Dict]:
        """Load known exchange addresses."""
        return {
            'bitcoin': {
                'binance': {
                    'addresses': ['1BinanceHot...', '1BinanceCold...'],
                    'risk_level': 'low',
                    'kyc_required': True
                },
                'coinbase': {
                    'addresses': ['1CoinbaseHot...', '1CoinbaseCold...'],
                    'risk_level': 'low',
                    'kyc_required': True
                },
                'localbitcoins': {
                    'addresses': ['1LocalBTC...'],
                    'risk_level': 'medium',
                    'kyc_required': False
                }
            },
            'ethereum': {
                'binance': {
                    'addresses': ['0xBinance...'],
                    'risk_level': 'low',
                    'kyc_required': True
                },
                'uniswap': {
                    'addresses': ['0xUniswap...'],
                    'risk_level': 'medium',
                    'kyc_required': False
                }
            }
        }
    
    def _load_privacy_coins(self) -> Dict[str, Dict]:
        """Load privacy coin characteristics."""
        return {
            'monero': {
                'privacy_level': 'high',
                'default_mixing': True,
                'traceability': 'very_low',
                'risk_multiplier': 1.5
            },
            'zcash': {
                'privacy_level': 'high',
                'default_mixing': False,
                'traceability': 'low',
                'risk_multiplier': 1.3
            },
            'dash': {
                'privacy_level': 'medium',
                'default_mixing': False,
                'traceability': 'medium',
                'risk_multiplier': 1.2
            }
        }
    
    def _load_risk_patterns(self) -> Dict[str, Dict]:
        """Load cryptocurrency risk patterns."""
        return {
            'rapid_movement': {
                'description': 'Funds moved quickly through multiple addresses',
                'risk_score': 0.8,
                'time_threshold': 3600  # 1 hour
            },
            'mixing_service_usage': {
                'description': 'Interaction with known mixing services',
                'risk_score': 0.9,
                'confirmation_required': True
            },
            'privacy_coin_conversion': {
                'description': 'Conversion to privacy coins',
                'risk_score': 0.7,
                'currencies': ['monero', 'zcash', 'dash']
            },
            'exchange_hopping': {
                'description': 'Rapid movement between exchanges',
                'risk_score': 0.6,
                'min_exchanges': 3
            },
            'small_amount_splitting': {
                'description': 'Large amounts split into many small transactions',
                'risk_score': 0.7,
                'threshold_ratio': 0.1
            },
            'dormant_address_activation': {
                'description': 'Previously inactive addresses suddenly active',
                'risk_score': 0.5,
                'dormancy_threshold': 180  # days
            }
        }
    
    def analyze_crypto_transactions(self, transactions: List[Dict]) -> BlockchainAnalysis:
        """
        Analyze cryptocurrency transactions for suspicious patterns.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            BlockchainAnalysis containing analysis results
        """
        print(f"🔍 Analyzing {len(transactions)} cryptocurrency transactions...")
        
        # Convert to CryptoTransaction objects
        crypto_txns = []
        for txn in transactions:
            crypto_txn = self._convert_to_crypto_transaction(txn)
            crypto_txns.append(crypto_txn)
        
        self.crypto_transactions.extend(crypto_txns)
        
        # Group by blockchain
        blockchain_groups = {}
        for txn in crypto_txns:
            if txn.blockchain not in blockchain_groups:
                blockchain_groups[txn.blockchain] = []
            blockchain_groups[txn.blockchain].append(txn)
        
        # Analyze each blockchain
        total_clusters = 0
        total_mixing_services = 0
        total_high_risk = 0
        
        for blockchain, txns in blockchain_groups.items():
            print(f"  📊 Analyzing {blockchain} blockchain ({len(txns)} transactions)...")
            
            # Build transaction graph
            self._build_transaction_graph(blockchain, txns)
            
            # Detect mixing services
            mixing_detected = self._detect_mixing_services(blockchain, txns)
            total_mixing_services += mixing_detected
            
            # Perform clustering
            clusters = self._perform_address_clustering(blockchain, txns)
            total_clusters += len(clusters)
            
            # Identify high-risk addresses
            high_risk = self._identify_high_risk_addresses(blockchain, txns)
            total_high_risk += len(high_risk)
            
            # Analyze privacy coin usage
            if blockchain in self.privacy_coins:
                self._analyze_privacy_coin_patterns(blockchain, txns)
        
        # Generate analysis summary
        analysis = BlockchainAnalysis(
            blockchain='multi_chain',
            analysis_type='comprehensive',
            entities_analyzed=len(set(txn.from_address for txn in crypto_txns) | 
                               set(txn.to_address for txn in crypto_txns)),
            transactions_analyzed=len(crypto_txns),
            clusters_identified=total_clusters,
            mixing_services_detected=total_mixing_services,
            high_risk_addresses=total_high_risk,
            analysis_timestamp=datetime.now(),
            summary=f"Analyzed {len(crypto_txns)} transactions across {len(blockchain_groups)} blockchains"
        )
        
        print(f"✅ Crypto analysis completed: {total_high_risk} high-risk addresses, {total_mixing_services} mixing services detected")
        return analysis
    
    def _convert_to_crypto_transaction(self, txn_dict: Dict) -> CryptoTransaction:
        """Convert transaction dictionary to CryptoTransaction object."""
        return CryptoTransaction(
            tx_hash=txn_dict.get('tx_hash', self._generate_tx_hash()),
            blockchain=txn_dict.get('blockchain', 'bitcoin'),
            from_address=txn_dict.get('from_address', txn_dict.get('sender_id', 'unknown')),
            to_address=txn_dict.get('to_address', txn_dict.get('receiver_id', 'unknown')),
            amount=float(txn_dict.get('amount', 0)),
            currency=txn_dict.get('currency', 'BTC'),
            timestamp=pd.to_datetime(txn_dict.get('timestamp', datetime.now())),
            block_height=txn_dict.get('block_height', random.randint(700000, 800000)),
            confirmations=txn_dict.get('confirmations', 6),
            fee=float(txn_dict.get('fee', 0.0001)),
            risk_factors=[]
        )
    
    def _generate_tx_hash(self) -> str:
        """Generate a realistic transaction hash."""
        return hashlib.sha256(f"{datetime.now().isoformat()}{random.random()}".encode()).hexdigest()
    
    def _build_transaction_graph(self, blockchain: str, transactions: List[CryptoTransaction]):
        """Build transaction graph for blockchain analysis."""
        if not HAS_NETWORKX:
            return
        
        G = nx.DiGraph()
        
        for txn in transactions:
            G.add_edge(
                txn.from_address, 
                txn.to_address,
                weight=txn.amount,
                timestamp=txn.timestamp,
                tx_hash=txn.tx_hash
            )
        
        self.blockchain_graphs[blockchain] = G
    
    def _detect_mixing_services(self, blockchain: str, transactions: List[CryptoTransaction]) -> int:
        """Detect mixing service usage."""
        mixing_services = self.mixing_services.get(blockchain, {})
        detected_count = 0
        
        for txn in transactions:
            # Check if addresses are known mixing services
            for service_name, service_data in mixing_services.items():
                service_addresses = service_data['addresses']
                
                if (txn.from_address in service_addresses or 
                    txn.to_address in service_addresses or
                    any(addr in txn.from_address for addr in service_addresses if addr != 'all')):
                    
                    detected_count += 1
                    
                    # Create alert
                    alert = CryptoAlert(
                        alert_id=f"MIXING_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{txn.tx_hash[:8]}",
                        alert_type="mixing_service_detected",
                        blockchain=blockchain,
                        addresses_involved=[txn.from_address, txn.to_address],
                        transactions_involved=[txn.tx_hash],
                        description=f"Interaction with {service_name} mixing service",
                        risk_score=0.9,
                        evidence=[f"Address matches known {service_name} service"],
                        recommended_actions=[
                            "Enhanced monitoring required",
                            "Investigate source and destination of funds",
                            "Consider filing SAR"
                        ],
                        timestamp=datetime.now(),
                        severity="high"
                    )
                    
                    self.crypto_alerts.append(alert)
                    
                    # Update transaction risk
                    txn.risk_score = max(txn.risk_score, 0.9)
                    txn.risk_factors.append(f"Mixing service usage: {service_name}")
        
        return detected_count
    
    def _perform_address_clustering(self, blockchain: str, transactions: List[CryptoTransaction]) -> List[Dict]:
        """Perform address clustering analysis."""
        if not HAS_NETWORKX or blockchain not in self.blockchain_graphs:
            return []
        
        G = self.blockchain_graphs[blockchain]
        
        # Find strongly connected components
        try:
            components = list(nx.strongly_connected_components(G))
            clusters = []
            
            for i, component in enumerate(components):
                if len(component) > 1:  # Only consider clusters with multiple addresses
                    cluster = {
                        'cluster_id': f"{blockchain}_cluster_{i}",
                        'addresses': list(component),
                        'size': len(component),
                        'total_volume': 0.0,
                        'transaction_count': 0
                    }
                    
                    # Calculate cluster statistics
                    for addr in component:
                        addr_txns = [txn for txn in transactions 
                                   if txn.from_address == addr or txn.to_address == addr]
                        cluster['total_volume'] += sum(txn.amount for txn in addr_txns)
                        cluster['transaction_count'] += len(addr_txns)
                    
                    clusters.append(cluster)
                    
                    # Update address cluster IDs
                    for addr in component:
                        if addr not in self.crypto_addresses:
                            self.crypto_addresses[addr] = CryptoAddress(
                                address=addr,
                                blockchain=blockchain,
                                address_type='wallet',
                                risk_level='unknown',
                                total_received=0.0,
                                total_sent=0.0,
                                transaction_count=0,
                                first_seen=datetime.now(),
                                last_seen=datetime.now(),
                                labels=[]
                            )
                        
                        self.crypto_addresses[addr].cluster_id = cluster['cluster_id']
            
            return clusters
            
        except Exception as e:
            print(f"⚠️ Clustering failed for {blockchain}: {str(e)}")
            return []
    
    def _identify_high_risk_addresses(self, blockchain: str, transactions: List[CryptoTransaction]) -> List[str]:
        """Identify high-risk cryptocurrency addresses."""
        high_risk_addresses = []
        address_stats = {}
        
        # Calculate address statistics
        for txn in transactions:
            # From address stats
            if txn.from_address not in address_stats:
                address_stats[txn.from_address] = {
                    'sent_amount': 0.0,
                    'received_amount': 0.0,
                    'sent_count': 0,
                    'received_count': 0,
                    'first_seen': txn.timestamp,
                    'last_seen': txn.timestamp,
                    'counterparties': set()
                }
            
            address_stats[txn.from_address]['sent_amount'] += txn.amount
            address_stats[txn.from_address]['sent_count'] += 1
            address_stats[txn.from_address]['last_seen'] = max(
                address_stats[txn.from_address]['last_seen'], txn.timestamp
            )
            address_stats[txn.from_address]['counterparties'].add(txn.to_address)
            
            # To address stats
            if txn.to_address not in address_stats:
                address_stats[txn.to_address] = {
                    'sent_amount': 0.0,
                    'received_amount': 0.0,
                    'sent_count': 0,
                    'received_count': 0,
                    'first_seen': txn.timestamp,
                    'last_seen': txn.timestamp,
                    'counterparties': set()
                }
            
            address_stats[txn.to_address]['received_amount'] += txn.amount
            address_stats[txn.to_address]['received_count'] += 1
            address_stats[txn.to_address]['last_seen'] = max(
                address_stats[txn.to_address]['last_seen'], txn.timestamp
            )
            address_stats[txn.to_address]['counterparties'].add(txn.from_address)
        
        # Analyze each address for risk factors
        for address, stats in address_stats.items():
            risk_score = 0.0
            risk_factors = []
            
            # High transaction volume
            total_volume = stats['sent_amount'] + stats['received_amount']
            if total_volume > 100.0:  # Threshold for high volume
                risk_score += 0.2
                risk_factors.append(f"High transaction volume: {total_volume:.2f}")
            
            # High transaction frequency
            total_count = stats['sent_count'] + stats['received_count']
            if total_count > 50:
                risk_score += 0.2
                risk_factors.append(f"High transaction frequency: {total_count}")
            
            # Many counterparties (potential hub)
            if len(stats['counterparties']) > 20:
                risk_score += 0.3
                risk_factors.append(f"Many counterparties: {len(stats['counterparties'])}")
            
            # Rapid turnover
            time_active = (stats['last_seen'] - stats['first_seen']).total_seconds()
            if time_active > 0:
                turnover_rate = total_volume / (time_active / 3600)  # per hour
                if turnover_rate > 10.0:
                    risk_score += 0.2
                    risk_factors.append(f"Rapid turnover: {turnover_rate:.2f}/hour")
            
            # Check against known high-risk patterns
            risk_score += self._check_address_patterns(address, stats)
            
            # Classify as high-risk if score exceeds threshold
            if risk_score >= self.config['crypto_analysis']['risk_threshold']:
                high_risk_addresses.append(address)
                
                # Create/update address record
                self.crypto_addresses[address] = CryptoAddress(
                    address=address,
                    blockchain=blockchain,
                    address_type='wallet',
                    risk_level='high',
                    total_received=stats['received_amount'],
                    total_sent=stats['sent_amount'],
                    transaction_count=total_count,
                    first_seen=stats['first_seen'],
                    last_seen=stats['last_seen'],
                    labels=risk_factors
                )
        
        return high_risk_addresses
    
    def _check_address_patterns(self, address: str, stats: Dict) -> float:
        """Check address against known risk patterns."""
        risk_score = 0.0
        
        # Check for exchange hopping pattern
        exchange_interactions = 0
        for blockchain in self.exchange_addresses:
            for exchange, data in self.exchange_addresses[blockchain].items():
                if any(addr in address for addr in data['addresses']):
                    exchange_interactions += 1
        
        if exchange_interactions >= 3:
            risk_score += 0.3
        
        # Check for small amount splitting
        if stats['sent_count'] > 10:
            avg_sent = stats['sent_amount'] / stats['sent_count']
            if avg_sent < 0.1:  # Small average amounts
                risk_score += 0.2
        
        return risk_score
    
    def _analyze_privacy_coin_patterns(self, blockchain: str, transactions: List[CryptoTransaction]):
        """Analyze privacy coin usage patterns."""
        if blockchain not in self.privacy_coins:
            return
        
        privacy_coin_data = self.privacy_coins[blockchain]
        
        for txn in transactions:
            # Apply privacy coin risk multiplier
            base_risk = txn.risk_score
            txn.risk_score = min(base_risk * privacy_coin_data['risk_multiplier'], 1.0)
            
            if privacy_coin_data['privacy_level'] == 'high':
                txn.risk_factors.append(f"Privacy coin usage: {blockchain}")
                
                # Create alert for high-value privacy coin transactions
                if txn.amount > 10.0:
                    alert = CryptoAlert(
                        alert_id=f"PRIVACY_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{txn.tx_hash[:8]}",
                        alert_type="privacy_coin_usage",
                        blockchain=blockchain,
                        addresses_involved=[txn.from_address, txn.to_address],
                        transactions_involved=[txn.tx_hash],
                        description=f"High-value {blockchain} transaction with enhanced privacy",
                        risk_score=txn.risk_score,
                        evidence=[f"Privacy coin: {blockchain}", f"Amount: {txn.amount}"],
                        recommended_actions=[
                            "Enhanced due diligence required",
                            "Monitor for additional privacy coin usage",
                            "Consider source of funds investigation"
                        ],
                        timestamp=datetime.now(),
                        severity="medium"
                    )
                    
                    self.crypto_alerts.append(alert)
    
    def trace_crypto_funds(self, start_address: str, blockchain: str, 
                          max_hops: int = 3) -> Dict[str, Any]:
        """
        Trace cryptocurrency funds through the blockchain.
        
        Args:
            start_address: Starting address for tracing
            blockchain: Blockchain to trace on
            max_hops: Maximum number of hops to trace
            
        Returns:
            Dictionary containing trace results
        """
        print(f"🔍 Tracing funds from {start_address} on {blockchain}...")
        
        if blockchain not in self.blockchain_graphs:
            return {'error': 'No transaction graph available for blockchain'}
        
        G = self.blockchain_graphs[blockchain]
        
        if start_address not in G:
            return {'error': 'Address not found in transaction graph'}
        
        # Perform forward trace
        forward_trace = self._trace_forward(G, start_address, max_hops)
        
        # Perform backward trace
        backward_trace = self._trace_backward(G, start_address, max_hops)
        
        # Analyze trace results
        trace_analysis = self._analyze_trace_results(forward_trace, backward_trace)
        
        return {
            'start_address': start_address,
            'blockchain': blockchain,
            'forward_trace': forward_trace,
            'backward_trace': backward_trace,
            'analysis': trace_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _trace_forward(self, graph: nx.DiGraph, start_address: str, max_hops: int) -> List[Dict]:
        """Trace funds forward through the blockchain."""
        trace_path = []
        current_addresses = {start_address}
        
        for hop in range(max_hops):
            next_addresses = set()
            
            for addr in current_addresses:
                if addr in graph:
                    for successor in graph.successors(addr):
                        edge_data = graph[addr][successor]
                        trace_path.append({
                            'hop': hop + 1,
                            'from': addr,
                            'to': successor,
                            'amount': edge_data.get('weight', 0),
                            'timestamp': edge_data.get('timestamp', datetime.now()).isoformat(),
                            'tx_hash': edge_data.get('tx_hash', 'unknown')
                        })
                        next_addresses.add(successor)
            
            current_addresses = next_addresses
            
            if not current_addresses:
                break
        
        return trace_path
    
    def _trace_backward(self, graph: nx.DiGraph, start_address: str, max_hops: int) -> List[Dict]:
        """Trace funds backward through the blockchain."""
        trace_path = []
        current_addresses = {start_address}
        
        for hop in range(max_hops):
            prev_addresses = set()
            
            for addr in current_addresses:
                if addr in graph:
                    for predecessor in graph.predecessors(addr):
                        edge_data = graph[predecessor][addr]
                        trace_path.append({
                            'hop': hop + 1,
                            'from': predecessor,
                            'to': addr,
                            'amount': edge_data.get('weight', 0),
                            'timestamp': edge_data.get('timestamp', datetime.now()).isoformat(),
                            'tx_hash': edge_data.get('tx_hash', 'unknown')
                        })
                        prev_addresses.add(predecessor)
            
            current_addresses = prev_addresses
            
            if not current_addresses:
                break
        
        return trace_path
    
    def _analyze_trace_results(self, forward_trace: List[Dict], backward_trace: List[Dict]) -> Dict:
        """Analyze trace results for suspicious patterns."""
        analysis = {
            'total_forward_hops': len(forward_trace),
            'total_backward_hops': len(backward_trace),
            'suspicious_patterns': [],
            'risk_score': 0.0,
            'mixing_services_encountered': 0,
            'exchange_interactions': 0
        }
        
        # Analyze all trace transactions
        all_traces = forward_trace + backward_trace
        
        for trace in all_traces:
            # Check for mixing services
            for blockchain in self.mixing_services:
                for service_name, service_data in self.mixing_services[blockchain].items():
                    if (trace['from'] in service_data['addresses'] or 
                        trace['to'] in service_data['addresses']):
                        analysis['mixing_services_encountered'] += 1
                        analysis['suspicious_patterns'].append(f"Mixing service: {service_name}")
            
            # Check for exchange interactions
            for blockchain in self.exchange_addresses:
                for exchange, data in self.exchange_addresses[blockchain].items():
                    if (trace['from'] in data['addresses'] or 
                        trace['to'] in data['addresses']):
                        analysis['exchange_interactions'] += 1
        
        # Calculate risk score
        if analysis['mixing_services_encountered'] > 0:
            analysis['risk_score'] += 0.4
        
        if analysis['exchange_interactions'] > 3:
            analysis['risk_score'] += 0.3
        
        if len(all_traces) > 10:
            analysis['risk_score'] += 0.2
            analysis['suspicious_patterns'].append("Complex transaction path")
        
        return analysis
    
    def generate_crypto_report(self, blockchain: str = None) -> Dict[str, Any]:
        """Generate comprehensive cryptocurrency analysis report."""
        print("📊 Generating cryptocurrency analysis report...")
        
        # Filter transactions by blockchain if specified
        if blockchain:
            transactions = [txn for txn in self.crypto_transactions if txn.blockchain == blockchain]
            alerts = [alert for alert in self.crypto_alerts if alert.blockchain == blockchain]
        else:
            transactions = self.crypto_transactions
            alerts = self.crypto_alerts
        
        # Calculate statistics
        total_volume = sum(txn.amount for txn in transactions)
        high_risk_txns = [txn for txn in transactions if txn.risk_score > 0.7]
        
        # Group by blockchain
        blockchain_stats = {}
        for txn in transactions:
            if txn.blockchain not in blockchain_stats:
                blockchain_stats[txn.blockchain] = {
                    'transaction_count': 0,
                    'total_volume': 0.0,
                    'high_risk_count': 0,
                    'unique_addresses': set()
                }
            
            blockchain_stats[txn.blockchain]['transaction_count'] += 1
            blockchain_stats[txn.blockchain]['total_volume'] += txn.amount
            blockchain_stats[txn.blockchain]['unique_addresses'].add(txn.from_address)
            blockchain_stats[txn.blockchain]['unique_addresses'].add(txn.to_address)
            
            if txn.risk_score > 0.7:
                blockchain_stats[txn.blockchain]['high_risk_count'] += 1
        
        # Convert sets to counts
        for blockchain, stats in blockchain_stats.items():
            stats['unique_addresses'] = len(stats['unique_addresses'])
        
        # Alert statistics
        alert_stats = {}
        for alert in alerts:
            alert_type = alert.alert_type
            if alert_type not in alert_stats:
                alert_stats[alert_type] = 0
            alert_stats[alert_type] += 1
        
        report = {
            'report_id': f"CRYPTO_RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'blockchain_filter': blockchain,
            'summary': {
                'total_transactions': len(transactions),
                'total_volume': total_volume,
                'high_risk_transactions': len(high_risk_txns),
                'alerts_generated': len(alerts),
                'blockchains_analyzed': len(blockchain_stats),
                'unique_addresses': len(self.crypto_addresses)
            },
            'blockchain_breakdown': blockchain_stats,
            'alert_breakdown': alert_stats,
            'top_risk_addresses': [
                {
                    'address': addr.address,
                    'blockchain': addr.blockchain,
                    'risk_level': addr.risk_level,
                    'total_volume': addr.total_received + addr.total_sent,
                    'transaction_count': addr.transaction_count
                }
                for addr in sorted(self.crypto_addresses.values(), 
                                 key=lambda x: x.total_received + x.total_sent, reverse=True)[:10]
            ],
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'blockchain': alert.blockchain,
                    'risk_score': alert.risk_score,
                    'description': alert.description,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:20]
            ]
        }
        
        return report
    
    def export_crypto_data(self, output_dir: str = "crypto_analysis/"):
        """Export cryptocurrency analysis data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export transactions
        transactions_data = [
            {
                'tx_hash': txn.tx_hash,
                'blockchain': txn.blockchain,
                'from_address': txn.from_address,
                'to_address': txn.to_address,
                'amount': txn.amount,
                'currency': txn.currency,
                'timestamp': txn.timestamp.isoformat(),
                'risk_score': txn.risk_score,
                'risk_factors': txn.risk_factors
            }
            for txn in self.crypto_transactions
        ]
        
        with open(output_path / f"crypto_transactions_{timestamp}.json", 'w') as f:
            json.dump(transactions_data, f, indent=2)
        
        # Export addresses
        addresses_data = [
            {
                'address': addr.address,
                'blockchain': addr.blockchain,
                'address_type': addr.address_type,
                'risk_level': addr.risk_level,
                'total_received': addr.total_received,
                'total_sent': addr.total_sent,
                'transaction_count': addr.transaction_count,
                'first_seen': addr.first_seen.isoformat(),
                'last_seen': addr.last_seen.isoformat(),
                'labels': addr.labels,
                'cluster_id': addr.cluster_id
            }
            for addr in self.crypto_addresses.values()
        ]
        
        with open(output_path / f"crypto_addresses_{timestamp}.json", 'w') as f:
            json.dump(addresses_data, f, indent=2)
        
        # Export alerts
        alerts_data = [
            {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'blockchain': alert.blockchain,
                'addresses_involved': alert.addresses_involved,
                'transactions_involved': alert.transactions_involved,
                'description': alert.description,
                'risk_score': alert.risk_score,
                'evidence': alert.evidence,
                'recommended_actions': alert.recommended_actions,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity
            }
            for alert in self.crypto_alerts
        ]
        
        with open(output_path / f"crypto_alerts_{timestamp}.json", 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        print(f"📁 Cryptocurrency analysis data exported to {output_path}")
    
    def get_crypto_summary(self) -> str:
        """Generate human-readable cryptocurrency analysis summary."""
        total_transactions = len(self.crypto_transactions)
        total_alerts = len(self.crypto_alerts)
        high_risk_addresses = len([addr for addr in self.crypto_addresses.values() 
                                 if addr.risk_level == 'high'])
        
        summary = f"""
CRYPTOCURRENCY ANALYSIS SUMMARY
===============================

Total Transactions Analyzed: {total_transactions}
High-Risk Addresses Identified: {high_risk_addresses}
Alerts Generated: {total_alerts}

Blockchain Coverage:
"""
        
        # Blockchain breakdown
        blockchain_counts = {}
        for txn in self.crypto_transactions:
            blockchain_counts[txn.blockchain] = blockchain_counts.get(txn.blockchain, 0) + 1
        
        for blockchain, count in blockchain_counts.items():
            summary += f"  {blockchain.title()}: {count} transactions\n"
        
        # Alert breakdown
        if self.crypto_alerts:
            summary += f"\nAlert Types:\n"
            alert_types = {}
            for alert in self.crypto_alerts:
                alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
            
            for alert_type, count in sorted(alert_types.items(), key=lambda x: x[1], reverse=True):
                summary += f"  {alert_type.replace('_', ' ').title()}: {count}\n"
        
        # Risk assessment
        mixing_alerts = len([a for a in self.crypto_alerts if a.alert_type == 'mixing_service_detected'])
        privacy_alerts = len([a for a in self.crypto_alerts if a.alert_type == 'privacy_coin_usage'])
        
        summary += f"\nRisk Indicators:\n"
        summary += f"  Mixing Services Detected: {mixing_alerts}\n"
        summary += f"  Privacy Coin Usage: {privacy_alerts}\n"
        
        summary += f"\nOverall Risk Level: {'🚨 HIGH RISK' if high_risk_addresses > 0 else '✅ NORMAL'}"
        
        return summary


def main():
    """Main function for testing the Cryptocurrency Specialist."""
    print("Testing Blue Team Cryptocurrency Specialist...")
    
    # Initialize specialist
    crypto_specialist = CryptocurrencySpecialist()
    
    # Create sample crypto transactions
    sample_crypto_txns = [
        {
            'tx_hash': '1a2b3c4d5e6f...',
            'blockchain': 'bitcoin',
            'from_address': '1BTC123...',
            'to_address': '1BTC456...',
            'amount': 1.5,
            'currency': 'BTC',
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'tx_hash': '2b3c4d5e6f7a...',
            'blockchain': 'ethereum',
            'from_address': '0xETH123...',
            'to_address': '0xETH456...',
            'amount': 10.0,
            'currency': 'ETH',
            'timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'tx_hash': '3c4d5e6f7a8b...',
            'blockchain': 'monero',
            'from_address': '4Monero123...',
            'to_address': '4Monero456...',
            'amount': 5.0,
            'currency': 'XMR',
            'timestamp': datetime.now()
        }
    ]
    
    # Add suspicious mixing service transaction
    sample_crypto_txns.append({
        'tx_hash': '4d5e6f7a8b9c...',
        'blockchain': 'bitcoin',
        'from_address': '1BTC789...',
        'to_address': '1TornadoCash...',
        'amount': 0.5,
        'currency': 'BTC',
        'timestamp': datetime.now() - timedelta(minutes=30)
    })
    
    # Analyze transactions
    print("\nAnalyzing cryptocurrency transactions...")
    analysis = crypto_specialist.analyze_crypto_transactions(sample_crypto_txns)
    
    # Test fund tracing
    print("\nTesting fund tracing...")
    trace_result = crypto_specialist.trace_crypto_funds('1BTC123...', 'bitcoin', max_hops=2)
    
    # Generate report
    print("\nGenerating cryptocurrency report...")
    report = crypto_specialist.generate_crypto_report()
    
    # Display results
    print("\nCryptocurrency Analysis Summary:")
    print("=" * 50)
    print(crypto_specialist.get_crypto_summary())
    
    # Export data
    crypto_specialist.export_crypto_data()
    
    print("\n✅ Cryptocurrency Specialist test completed!")


if __name__ == "__main__":
    main() 