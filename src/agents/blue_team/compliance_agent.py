"""
Blue Team Compliance Agent

This agent monitors regulatory compliance in real-time, tracks regulatory limits,
generates compliance alerts, and ensures adherence to multiple jurisdictional
requirements for AML-FT operations.
"""

import json
import yaml
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


@dataclass
class ComplianceAlert:
    """Data class for compliance alerts."""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    entity_id: str
    description: str
    regulatory_reference: str
    threshold_value: float
    current_value: float
    jurisdiction: str
    timestamp: datetime
    status: str = "active"
    remediation_required: bool = True


@dataclass
class RegulatoryLimit:
    """Data class for regulatory limits."""
    limit_id: str
    limit_type: str
    jurisdiction: str
    threshold_amount: float
    time_period: str  # 'daily', 'monthly', 'annual'
    description: str
    regulatory_reference: str
    penalty_description: str
    monitoring_enabled: bool = True


@dataclass
class ComplianceReport:
    """Data class for compliance reports."""
    report_id: str
    report_type: str
    jurisdiction: str
    period_start: datetime
    period_end: datetime
    compliance_status: str
    violations_count: int
    alerts_generated: int
    entities_monitored: int
    recommendations: List[str]
    created_at: datetime


class ComplianceAgent:
    """
    Monitors regulatory compliance in real-time for AML-FT operations.
    
    This agent tracks regulatory limits, generates compliance alerts,
    monitors jurisdictional requirements, and ensures adherence to
    multiple regulatory frameworks.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Compliance Agent."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.regulatory_limits = self._load_regulatory_limits()
        self.compliance_alerts = []
        self.compliance_reports = []
        self.entity_tracking = {}
        self.jurisdiction_rules = self._load_jurisdiction_rules()
        
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
                'temperature': 0.1,
                'max_tokens': 2000
            },
            'compliance': {
                'monitoring_enabled': True,
                'alert_threshold': 0.8,
                'jurisdictions': ['US', 'UK', 'EU', 'BR'],
                'reporting_frequency': 'daily',
                'auto_remediation': False
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for compliance analysis."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def _load_regulatory_limits(self) -> List[RegulatoryLimit]:
        """Load regulatory limits for different jurisdictions."""
        limits = []
        
        # US FinCEN Limits
        limits.extend([
            RegulatoryLimit(
                limit_id="US_CTR_DAILY",
                limit_type="currency_transaction_reporting",
                jurisdiction="US",
                threshold_amount=10000.0,
                time_period="daily",
                description="Currency Transaction Report threshold",
                regulatory_reference="31 CFR 1010.311",
                penalty_description="Civil penalties up to $25,000 per violation"
            ),
            RegulatoryLimit(
                limit_id="US_SAR_THRESHOLD",
                limit_type="suspicious_activity_reporting",
                jurisdiction="US",
                threshold_amount=5000.0,
                time_period="transaction",
                description="Suspicious Activity Report threshold",
                regulatory_reference="31 CFR 1020.320",
                penalty_description="Criminal penalties up to $500,000 and 5 years imprisonment"
            ),
            RegulatoryLimit(
                limit_id="US_STRUCTURING",
                limit_type="structuring_detection",
                jurisdiction="US",
                threshold_amount=10000.0,
                time_period="daily",
                description="Structuring to avoid CTR reporting",
                regulatory_reference="31 USC 5324",
                penalty_description="Criminal penalties up to $250,000 and 5 years imprisonment"
            )
        ])
        
        # UK FCA Limits
        limits.extend([
            RegulatoryLimit(
                limit_id="UK_SAR_THRESHOLD",
                limit_type="suspicious_activity_reporting",
                jurisdiction="UK",
                threshold_amount=1000.0,
                time_period="transaction",
                description="Suspicious Activity Report threshold",
                regulatory_reference="POCA 2002 Section 330",
                penalty_description="Up to 5 years imprisonment"
            ),
            RegulatoryLimit(
                limit_id="UK_CASH_DECLARATION",
                limit_type="cash_declaration",
                jurisdiction="UK",
                threshold_amount=10000.0,
                time_period="transport",
                description="Cash declaration requirement",
                regulatory_reference="POCA 2002 Section 294",
                penalty_description="Forfeiture of undeclared cash"
            )
        ])
        
        # EU Limits
        limits.extend([
            RegulatoryLimit(
                limit_id="EU_CDD_THRESHOLD",
                limit_type="customer_due_diligence",
                jurisdiction="EU",
                threshold_amount=15000.0,
                time_period="transaction",
                description="Customer Due Diligence threshold",
                regulatory_reference="4AMLD Article 11",
                penalty_description="Administrative sanctions up to €5,000,000"
            ),
            RegulatoryLimit(
                limit_id="EU_CASH_PAYMENT",
                limit_type="cash_payment_limit",
                jurisdiction="EU",
                threshold_amount=10000.0,
                time_period="transaction",
                description="Cash payment restriction",
                regulatory_reference="4AMLD Article 12",
                penalty_description="Administrative fines"
            )
        ])
        
        # Brazil COAF Limits
        limits.extend([
            RegulatoryLimit(
                limit_id="BR_COAF_THRESHOLD",
                limit_type="suspicious_transaction_reporting",
                jurisdiction="BR",
                threshold_amount=10000.0,
                time_period="transaction",
                description="COAF reporting threshold",
                regulatory_reference="Lei 9.613/1998",
                penalty_description="Multa de R$ 200.000 a R$ 20.000.000"
            ),
            RegulatoryLimit(
                limit_id="BR_CASH_LIMIT",
                limit_type="cash_transaction_limit",
                jurisdiction="BR",
                threshold_amount=2000.0,
                time_period="transaction",
                description="Cash transaction limit",
                regulatory_reference="Lei 12.683/2012",
                penalty_description="Multa e prisão de 3 a 10 anos"
            )
        ])
        
        return limits
    
    def _load_jurisdiction_rules(self) -> Dict[str, Dict]:
        """Load jurisdiction-specific rules and requirements."""
        return {
            'US': {
                'primary_regulator': 'FinCEN',
                'reporting_currency': 'USD',
                'business_day_cutoff': '17:00 EST',
                'weekend_processing': False,
                'enhanced_due_diligence_threshold': 25000.0,
                'politically_exposed_person_monitoring': True
            },
            'UK': {
                'primary_regulator': 'FCA',
                'reporting_currency': 'GBP',
                'business_day_cutoff': '17:00 GMT',
                'weekend_processing': False,
                'enhanced_due_diligence_threshold': 15000.0,
                'politically_exposed_person_monitoring': True
            },
            'EU': {
                'primary_regulator': 'EBA',
                'reporting_currency': 'EUR',
                'business_day_cutoff': '17:00 CET',
                'weekend_processing': False,
                'enhanced_due_diligence_threshold': 15000.0,
                'politically_exposed_person_monitoring': True
            },
            'BR': {
                'primary_regulator': 'COAF',
                'reporting_currency': 'BRL',
                'business_day_cutoff': '18:00 BRT',
                'weekend_processing': False,
                'enhanced_due_diligence_threshold': 50000.0,
                'politically_exposed_person_monitoring': True
            }
        }
    
    def monitor_transactions(self, transactions: pd.DataFrame, 
                           jurisdiction: str = 'US') -> List[ComplianceAlert]:
        """
        Monitor transactions for compliance violations.
        
        Args:
            transactions: DataFrame containing transaction data
            jurisdiction: Jurisdiction to apply rules for
            
        Returns:
            List of compliance alerts generated
        """
        print(f"🔍 Monitoring {len(transactions)} transactions for {jurisdiction} compliance...")
        
        alerts = []
        
        # Get relevant limits for jurisdiction
        jurisdiction_limits = [
            limit for limit in self.regulatory_limits 
            if limit.jurisdiction == jurisdiction and limit.monitoring_enabled
        ]
        
        for limit in jurisdiction_limits:
            limit_alerts = self._check_regulatory_limit(transactions, limit)
            alerts.extend(limit_alerts)
        
        # Additional compliance checks
        alerts.extend(self._check_structuring_patterns(transactions, jurisdiction))
        alerts.extend(self._check_unusual_patterns(transactions, jurisdiction))
        alerts.extend(self._check_high_risk_entities(transactions, jurisdiction))
        
        # Store alerts
        self.compliance_alerts.extend(alerts)
        
        print(f"✅ Generated {len(alerts)} compliance alerts")
        return alerts
    
    def _check_regulatory_limit(self, transactions: pd.DataFrame, 
                              limit: RegulatoryLimit) -> List[ComplianceAlert]:
        """Check transactions against a specific regulatory limit."""
        alerts = []
        
        if limit.limit_type == "currency_transaction_reporting":
            # Check for transactions at or above CTR threshold
            high_value_txns = transactions[transactions['amount'] >= limit.threshold_amount]
            
            for _, txn in high_value_txns.iterrows():
                alert = ComplianceAlert(
                    alert_id=f"CTR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{txn['transaction_id']}",
                    alert_type="currency_transaction_reporting",
                    severity="high",
                    entity_id=txn['sender_id'],
                    description=f"Transaction of ${txn['amount']:,.2f} exceeds CTR threshold",
                    regulatory_reference=limit.regulatory_reference,
                    threshold_value=limit.threshold_amount,
                    current_value=txn['amount'],
                    jurisdiction=limit.jurisdiction,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        elif limit.limit_type == "structuring_detection":
            # Check for potential structuring
            alerts.extend(self._detect_structuring(transactions, limit))
        
        elif limit.limit_type == "suspicious_activity_reporting":
            # Check for suspicious patterns
            alerts.extend(self._detect_suspicious_activity(transactions, limit))
        
        return alerts
    
    def _detect_structuring(self, transactions: pd.DataFrame, 
                           limit: RegulatoryLimit) -> List[ComplianceAlert]:
        """Detect potential structuring violations."""
        alerts = []
        
        # Group transactions by sender and date
        transactions['date'] = pd.to_datetime(transactions['timestamp']).dt.date
        
        daily_totals = transactions.groupby(['sender_id', 'date'])['amount'].agg(['sum', 'count']).reset_index()
        
        # Look for patterns just below the threshold
        threshold = limit.threshold_amount
        potential_structuring = daily_totals[
            (daily_totals['sum'] >= threshold * 0.7) &  # 70% of threshold
            (daily_totals['sum'] < threshold) &  # Below threshold
            (daily_totals['count'] >= 2)  # Multiple transactions
        ]
        
        for _, row in potential_structuring.iterrows():
            alert = ComplianceAlert(
                alert_id=f"STRUCT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{row['sender_id']}",
                alert_type="structuring_detection",
                severity="critical",
                entity_id=row['sender_id'],
                description=f"Potential structuring: ${row['sum']:,.2f} in {row['count']} transactions",
                regulatory_reference=limit.regulatory_reference,
                threshold_value=threshold,
                current_value=row['sum'],
                jurisdiction=limit.jurisdiction,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def _detect_suspicious_activity(self, transactions: pd.DataFrame, 
                                  limit: RegulatoryLimit) -> List[ComplianceAlert]:
        """Detect suspicious activity patterns."""
        alerts = []
        
        # Check for unusual patterns
        suspicious_patterns = [
            # Round amounts just below threshold
            (transactions['amount'] % 1000 == 0) & 
            (transactions['amount'] >= limit.threshold_amount * 0.9),
            
            # Rapid succession of transactions
            transactions.duplicated(subset=['sender_id'], keep=False),
            
            # Transactions at unusual times
            pd.to_datetime(transactions['timestamp']).dt.hour.isin([0, 1, 2, 3, 4, 5, 22, 23])
        ]
        
        for i, pattern in enumerate(suspicious_patterns):
            suspicious_txns = transactions[pattern]
            
            for _, txn in suspicious_txns.iterrows():
                alert = ComplianceAlert(
                    alert_id=f"SUSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{txn['transaction_id']}",
                    alert_type="suspicious_activity",
                    severity="medium",
                    entity_id=txn['sender_id'],
                    description=f"Suspicious pattern detected in transaction {txn['transaction_id']}",
                    regulatory_reference=limit.regulatory_reference,
                    threshold_value=limit.threshold_amount,
                    current_value=txn['amount'],
                    jurisdiction=limit.jurisdiction,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_structuring_patterns(self, transactions: pd.DataFrame, 
                                  jurisdiction: str) -> List[ComplianceAlert]:
        """Advanced structuring pattern detection."""
        alerts = []
        
        # Get structuring threshold for jurisdiction
        threshold = 10000.0  # Default threshold
        for limit in self.regulatory_limits:
            if (limit.jurisdiction == jurisdiction and 
                limit.limit_type == "structuring_detection"):
                threshold = limit.threshold_amount
                break
        
        # Multi-day structuring detection
        transactions['date'] = pd.to_datetime(transactions['timestamp']).dt.date
        
        # Look for patterns over multiple days
        entity_patterns = transactions.groupby('sender_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'date': 'nunique'
        }).reset_index()
        
        entity_patterns.columns = ['sender_id', 'total_amount', 'txn_count', 'avg_amount', 'days_active']
        
        # Identify potential structuring
        structuring_entities = entity_patterns[
            (entity_patterns['total_amount'] >= threshold * 0.8) &
            (entity_patterns['avg_amount'] < threshold) &
            (entity_patterns['txn_count'] >= 3) &
            (entity_patterns['days_active'] >= 2)
        ]
        
        for _, entity in structuring_entities.iterrows():
            alert = ComplianceAlert(
                alert_id=f"MULTI_STRUCT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity['sender_id']}",
                alert_type="multi_day_structuring",
                severity="critical",
                entity_id=entity['sender_id'],
                description=f"Multi-day structuring pattern: ${entity['total_amount']:,.2f} over {entity['days_active']} days",
                regulatory_reference="31 USC 5324",
                threshold_value=threshold,
                current_value=entity['total_amount'],
                jurisdiction=jurisdiction,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_unusual_patterns(self, transactions: pd.DataFrame, 
                               jurisdiction: str) -> List[ComplianceAlert]:
        """Check for unusual transaction patterns."""
        alerts = []
        
        # Velocity checks
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        transactions_sorted = transactions.sort_values(['sender_id', 'timestamp'])
        
        # Calculate time differences between consecutive transactions
        transactions_sorted['time_diff'] = transactions_sorted.groupby('sender_id')['timestamp'].diff()
        
        # Find rapid-fire transactions (less than 1 minute apart)
        rapid_transactions = transactions_sorted[
            transactions_sorted['time_diff'] < pd.Timedelta(minutes=1)
        ]
        
        for _, txn in rapid_transactions.iterrows():
            alert = ComplianceAlert(
                alert_id=f"RAPID_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{txn['transaction_id']}",
                alert_type="rapid_transactions",
                severity="medium",
                entity_id=txn['sender_id'],
                description=f"Rapid transaction pattern detected",
                regulatory_reference="General AML monitoring",
                threshold_value=60.0,  # seconds
                current_value=txn['time_diff'].total_seconds(),
                jurisdiction=jurisdiction,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_high_risk_entities(self, transactions: pd.DataFrame, 
                                 jurisdiction: str) -> List[ComplianceAlert]:
        """Check transactions involving high-risk entities."""
        alerts = []
        
        # Simulate high-risk entity list (in real implementation, this would be external)
        high_risk_entities = [
            'SHELL_001', 'MULE_001', 'CRIMINAL_001', 'SUSPICIOUS_001'
        ]
        
        high_risk_txns = transactions[
            (transactions['sender_id'].isin(high_risk_entities)) |
            (transactions['receiver_id'].isin(high_risk_entities))
        ]
        
        for _, txn in high_risk_txns.iterrows():
            alert = ComplianceAlert(
                alert_id=f"HIGH_RISK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{txn['transaction_id']}",
                alert_type="high_risk_entity",
                severity="high",
                entity_id=txn['sender_id'],
                description=f"Transaction involving high-risk entity",
                regulatory_reference="Enhanced Due Diligence requirements",
                threshold_value=0.0,
                current_value=txn['amount'],
                jurisdiction=jurisdiction,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def generate_compliance_report(self, jurisdiction: str, 
                                 period_days: int = 30) -> ComplianceReport:
        """Generate compliance report for a specific jurisdiction."""
        print(f"📊 Generating compliance report for {jurisdiction}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Filter alerts for the period
        period_alerts = [
            alert for alert in self.compliance_alerts
            if (alert.jurisdiction == jurisdiction and 
                start_date <= alert.timestamp <= end_date)
        ]
        
        # Count violations by severity
        violations_count = len([alert for alert in period_alerts if alert.severity in ['high', 'critical']])
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(period_alerts, jurisdiction)
        
        report = ComplianceReport(
            report_id=f"COMP_RPT_{jurisdiction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="periodic_compliance",
            jurisdiction=jurisdiction,
            period_start=start_date,
            period_end=end_date,
            compliance_status="compliant" if violations_count == 0 else "violations_detected",
            violations_count=violations_count,
            alerts_generated=len(period_alerts),
            entities_monitored=len(set(alert.entity_id for alert in period_alerts)),
            recommendations=recommendations,
            created_at=datetime.now()
        )
        
        self.compliance_reports.append(report)
        
        print(f"✅ Compliance report generated: {violations_count} violations, {len(period_alerts)} alerts")
        return report
    
    def _generate_compliance_recommendations(self, alerts: List[ComplianceAlert], 
                                           jurisdiction: str) -> List[str]:
        """Generate compliance recommendations based on alerts."""
        recommendations = []
        
        # Analyze alert patterns
        alert_types = {}
        for alert in alerts:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
        
        # Generate recommendations based on patterns
        if alert_types.get('structuring_detection', 0) > 5:
            recommendations.append("Implement enhanced structuring detection algorithms")
            recommendations.append("Increase monitoring frequency for high-risk entities")
        
        if alert_types.get('suspicious_activity', 0) > 10:
            recommendations.append("Review and update suspicious activity detection rules")
            recommendations.append("Provide additional training to compliance staff")
        
        if alert_types.get('high_risk_entity', 0) > 3:
            recommendations.append("Update high-risk entity lists")
            recommendations.append("Implement enhanced due diligence procedures")
        
        # Jurisdiction-specific recommendations
        if jurisdiction == 'US':
            recommendations.append("Ensure timely CTR and SAR filings with FinCEN")
        elif jurisdiction == 'UK':
            recommendations.append("Maintain compliance with FCA AML requirements")
        elif jurisdiction == 'EU':
            recommendations.append("Ensure 4AMLD/5AMLD compliance")
        elif jurisdiction == 'BR':
            recommendations.append("Maintain COAF reporting requirements")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get real-time compliance dashboard data."""
        current_time = datetime.now()
        last_24h = current_time - timedelta(hours=24)
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.compliance_alerts
            if alert.timestamp >= last_24h
        ]
        
        # Alert statistics
        alert_stats = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
            'high_alerts': len([a for a in recent_alerts if a.severity == 'high']),
            'medium_alerts': len([a for a in recent_alerts if a.severity == 'medium']),
            'low_alerts': len([a for a in recent_alerts if a.severity == 'low'])
        }
        
        # Jurisdiction breakdown
        jurisdiction_stats = {}
        for alert in recent_alerts:
            jurisdiction_stats[alert.jurisdiction] = jurisdiction_stats.get(alert.jurisdiction, 0) + 1
        
        # Alert type breakdown
        type_stats = {}
        for alert in recent_alerts:
            type_stats[alert.alert_type] = type_stats.get(alert.alert_type, 0) + 1
        
        return {
            'timestamp': current_time.isoformat(),
            'monitoring_status': 'active',
            'alert_statistics': alert_stats,
            'jurisdiction_breakdown': jurisdiction_stats,
            'alert_type_breakdown': type_stats,
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'entity_id': alert.entity_id,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def export_compliance_data(self, output_dir: str = "compliance_reports/"):
        """Export compliance data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export alerts
        alerts_data = [
            {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'entity_id': alert.entity_id,
                'description': alert.description,
                'regulatory_reference': alert.regulatory_reference,
                'threshold_value': alert.threshold_value,
                'current_value': alert.current_value,
                'jurisdiction': alert.jurisdiction,
                'timestamp': alert.timestamp.isoformat(),
                'status': alert.status
            }
            for alert in self.compliance_alerts
        ]
        
        with open(output_path / f"compliance_alerts_{timestamp}.json", 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        # Export reports
        reports_data = [
            {
                'report_id': report.report_id,
                'report_type': report.report_type,
                'jurisdiction': report.jurisdiction,
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'compliance_status': report.compliance_status,
                'violations_count': report.violations_count,
                'alerts_generated': report.alerts_generated,
                'entities_monitored': report.entities_monitored,
                'recommendations': report.recommendations,
                'created_at': report.created_at.isoformat()
            }
            for report in self.compliance_reports
        ]
        
        with open(output_path / f"compliance_reports_{timestamp}.json", 'w') as f:
            json.dump(reports_data, f, indent=2)
        
        print(f"📁 Compliance data exported to {output_path}")
    
    def get_compliance_summary(self) -> str:
        """Generate human-readable compliance summary."""
        total_alerts = len(self.compliance_alerts)
        critical_alerts = len([a for a in self.compliance_alerts if a.severity == 'critical'])
        
        summary = f"""
COMPLIANCE MONITORING SUMMARY
============================

Total Alerts Generated: {total_alerts}
Critical Alerts: {critical_alerts}
Reports Generated: {len(self.compliance_reports)}

Jurisdictions Monitored: {len(set(a.jurisdiction for a in self.compliance_alerts))}
Regulatory Limits Tracked: {len(self.regulatory_limits)}

Alert Types:
"""
        
        # Count alert types
        alert_types = {}
        for alert in self.compliance_alerts:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
        
        for alert_type, count in sorted(alert_types.items(), key=lambda x: x[1], reverse=True):
            summary += f"  {alert_type}: {count}\n"
        
        summary += f"\nCompliance Status: {'⚠️ VIOLATIONS DETECTED' if critical_alerts > 0 else '✅ COMPLIANT'}"
        
        return summary


def main():
    """Main function for testing the Compliance Agent."""
    print("Testing Blue Team Compliance Agent...")
    
    # Initialize agent
    compliance_agent = ComplianceAgent()
    
    # Create sample transaction data
    sample_transactions = pd.DataFrame({
        'transaction_id': [f'TX_{i:06d}' for i in range(100)],
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'sender_id': [f'SENDER_{i%10:03d}' for i in range(100)],
        'receiver_id': [f'RECEIVER_{i%8:03d}' for i in range(100)],
        'amount': np.random.lognormal(8, 1, 100)
    })
    
    # Add some compliance violations
    # Large transaction (CTR threshold)
    sample_transactions.loc[0, 'amount'] = 15000.0
    
    # Structuring pattern
    for i in range(5):
        sample_transactions.loc[len(sample_transactions)] = {
            'transaction_id': f'STRUCT_{i:03d}',
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
            'sender_id': 'SUSPICIOUS_001',
            'receiver_id': f'RECEIVER_{i:03d}',
            'amount': 9500.0,
        }
    
    # Monitor transactions
    print("\nMonitoring transactions for compliance...")
    alerts = compliance_agent.monitor_transactions(sample_transactions, jurisdiction='US')
    
    # Generate compliance report
    print("\nGenerating compliance report...")
    report = compliance_agent.generate_compliance_report('US', period_days=30)
    
    # Display results
    print("\nCompliance Summary:")
    print("=" * 50)
    print(compliance_agent.get_compliance_summary())
    
    # Export data
    compliance_agent.export_compliance_data()
    
    print("\n✅ Compliance Agent test completed!")


if __name__ == "__main__":
    main() 