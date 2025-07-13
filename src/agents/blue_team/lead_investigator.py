"""
Blue Team Lead Investigator Agent

This agent acts as the senior investigator who connects the dots between
transaction analysis and OSINT findings to construct coherent criminal
narratives and determine the most likely money laundering scenarios.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path

# LLM imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class Evidence:
    """Data class for evidence pieces."""
    evidence_id: str
    source: str
    evidence_type: str
    entity_id: str
    description: str
    confidence: float
    timestamp: datetime
    supporting_data: Dict[str, Any]


@dataclass
class CriminalNarrative:
    """Data class for criminal narrative construction."""
    narrative_id: str
    title: str
    summary: str
    criminal_technique: str
    entities_involved: List[str]
    evidence_pieces: List[Evidence]
    timeline: List[Dict[str, Any]]
    confidence_score: float
    risk_level: str
    recommended_actions: List[str]
    created_at: datetime


class LeadInvestigator:
    """
    Senior investigator who constructs criminal narratives from evidence.
    
    This agent combines transaction analysis results with OSINT findings
    to build coherent stories about potential money laundering operations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Lead Investigator."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.evidence_vault = []
        self.narratives = []
        self.investigation_context = {}
        
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
                'temperature': 0.4,
                'max_tokens': 4000
            },
            'investigation': {
                'min_evidence_pieces': 3,
                'confidence_threshold': 0.6,
                'narrative_depth': 'thorough',
                'include_alternative_scenarios': True
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for narrative construction."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        elif provider == 'anthropic' and anthropic:
            return anthropic.Anthropic(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def investigate_case(self, 
                        transaction_analysis: Dict,
                        osint_results: Dict[str, List]) -> List[CriminalNarrative]:
        """
        Conduct comprehensive investigation and build criminal narratives.
        
        Args:
            transaction_analysis: Results from Transaction Analyst
            osint_results: Results from OSINT Agent
            
        Returns:
            List of criminal narratives constructed
        """
        print("🕵️ Lead Investigator starting comprehensive case analysis...")
        
        # Step 1: Collect and organize evidence
        print("   📋 Collecting and organizing evidence...")
        self._collect_evidence(transaction_analysis, osint_results)
        
        # Step 2: Identify patterns and connections
        print("   🔗 Identifying patterns and connections...")
        connections = self._identify_connections()
        
        # Step 3: Construct criminal narratives
        print("   📖 Constructing criminal narratives...")
        narratives = self._construct_narratives(connections)
        
        # Step 4: Validate and rank narratives
        print("   ⚖️ Validating and ranking narratives...")
        validated_narratives = self._validate_narratives(narratives)
        
        # Step 5: Generate recommendations
        print("   💡 Generating investigation recommendations...")
        self._generate_recommendations(validated_narratives)
        
        self.narratives = validated_narratives
        
        print(f"✅ Investigation completed. {len(validated_narratives)} criminal narratives constructed.")
        
        return validated_narratives
    
    def _collect_evidence(self, transaction_analysis: Dict, osint_results: Dict[str, List]):
        """Collect evidence from all sources."""
        evidence_id_counter = 1
        
        # Collect transaction-based evidence
        suspicious_entities = transaction_analysis.get('suspicious_entities', [])
        for entity in suspicious_entities:
            evidence = Evidence(
                evidence_id=f"TXN_{evidence_id_counter:03d}",
                source='Transaction Analysis',
                evidence_type='suspicious_pattern',
                entity_id=entity.get('entity_id', 'unknown'),
                description=f"Entity flagged for {entity.get('reason', 'unknown reason')}",
                confidence=entity.get('risk_score', 0.5),
                timestamp=datetime.now(),
                supporting_data=entity
            )
            self.evidence_vault.append(evidence)
            evidence_id_counter += 1
        
        # Collect anomaly evidence
        anomalies = transaction_analysis.get('anomalies_detected', {})
        if anomalies.get('transactions'):
            for anomaly in anomalies['transactions'][:10]:  # Limit to top 10
                evidence = Evidence(
                    evidence_id=f"ANM_{evidence_id_counter:03d}",
                    source='Anomaly Detection',
                    evidence_type='statistical_anomaly',
                    entity_id=anomaly.get('sender_id', 'unknown'),
                    description=f"Anomalous transaction pattern detected",
                    confidence=abs(anomaly.get('anomaly_score', 0.5)),
                    timestamp=datetime.now(),
                    supporting_data=anomaly
                )
                self.evidence_vault.append(evidence)
                evidence_id_counter += 1
        
        # Collect network evidence
        network_analysis = transaction_analysis.get('network_analysis', {})
        suspicious_subgraphs = network_analysis.get('suspicious_subgraphs', [])
        for subgraph in suspicious_subgraphs:
            evidence = Evidence(
                evidence_id=f"NET_{evidence_id_counter:03d}",
                source='Network Analysis',
                evidence_type='network_pattern',
                entity_id=','.join(subgraph.get('nodes', [])),
                description=f"Suspicious network community detected: {subgraph.get('suspicion_reason', 'unknown')}",
                confidence=0.7,
                timestamp=datetime.now(),
                supporting_data=subgraph
            )
            self.evidence_vault.append(evidence)
            evidence_id_counter += 1
        
        # Collect OSINT evidence
        for entity_id, osint_findings in osint_results.items():
            for finding in osint_findings:
                evidence = Evidence(
                    evidence_id=f"OSI_{evidence_id_counter:03d}",
                    source='OSINT',
                    evidence_type=finding.information_type,
                    entity_id=entity_id,
                    description=finding.content,
                    confidence=finding.relevance_score,
                    timestamp=finding.timestamp,
                    supporting_data={
                        'source': finding.source,
                        'url': finding.url,
                        'confidence': getattr(finding, 'confidence', 0.0)
                    }
                )
                self.evidence_vault.append(evidence)
                evidence_id_counter += 1
        
        print(f"   📊 Collected {len(self.evidence_vault)} pieces of evidence")
    
    def _identify_connections(self) -> List[Dict[str, Any]]:
        """Identify connections between evidence pieces."""
        connections = []
        
        # Group evidence by entity
        entity_evidence = {}
        for evidence in self.evidence_vault:
            entity_ids = evidence.entity_id.split(',')
            for entity_id in entity_ids:
                entity_id = entity_id.strip()
                if entity_id not in entity_evidence:
                    entity_evidence[entity_id] = []
                entity_evidence[entity_id].append(evidence)
        
        # Find entities with multiple evidence pieces
        for entity_id, evidence_list in entity_evidence.items():
            if len(evidence_list) >= 2:
                connection = {
                    'connection_type': 'entity_multi_evidence',
                    'entity_id': entity_id,
                    'evidence_pieces': evidence_list,
                    'connection_strength': min(1.0, len(evidence_list) * 0.3),
                    'description': f"Multiple evidence pieces for entity {entity_id}"
                }
                connections.append(connection)
        
        # Find temporal connections
        temporal_connections = self._find_temporal_connections()
        connections.extend(temporal_connections)
        
        # Find technique-based connections
        technique_connections = self._find_technique_connections()
        connections.extend(technique_connections)
        
        return connections
    
    def _find_temporal_connections(self) -> List[Dict[str, Any]]:
        """Find evidence pieces that are temporally related."""
        connections = []
        
        # Sort evidence by timestamp
        sorted_evidence = sorted(self.evidence_vault, key=lambda x: x.timestamp)
        
        # Look for evidence within short time windows
        for i in range(len(sorted_evidence)):
            for j in range(i + 1, len(sorted_evidence)):
                evidence1 = sorted_evidence[i]
                evidence2 = sorted_evidence[j]
                
                time_diff = abs((evidence1.timestamp - evidence2.timestamp).total_seconds())
                
                # If evidence is within 1 hour and involves different entities
                if time_diff <= 3600 and evidence1.entity_id != evidence2.entity_id:
                    connection = {
                        'connection_type': 'temporal_correlation',
                        'evidence_pieces': [evidence1, evidence2],
                        'connection_strength': max(0.1, 1.0 - (time_diff / 3600)),
                        'description': f"Temporal correlation between {evidence1.entity_id} and {evidence2.entity_id}",
                        'time_difference_seconds': time_diff
                    }
                    connections.append(connection)
        
        return connections
    
    def _find_technique_connections(self) -> List[Dict[str, Any]]:
        """Find evidence pieces that suggest the same criminal technique."""
        connections = []
        
        # Group evidence by suspected technique
        technique_evidence = {
            'smurfing': [],
            'shell_companies': [],
            'money_mules': [],
            'cash_businesses': [],
            'layering': [],
            'structuring': []
        }
        
        for evidence in self.evidence_vault:
            # Classify evidence by technique indicators
            description = evidence.description.lower()
            
            if 'structuring' in description or 'multiple small' in description:
                technique_evidence['structuring'].append(evidence)
            elif 'shell' in description or 'fake business' in description:
                technique_evidence['shell_companies'].append(evidence)
            elif 'mule' in description or 'third party' in description:
                technique_evidence['money_mules'].append(evidence)
            elif 'cash' in description or 'business' in description:
                technique_evidence['cash_businesses'].append(evidence)
            elif 'rapid' in description or 'layering' in description:
                technique_evidence['layering'].append(evidence)
        
        # Create connections for techniques with multiple evidence pieces
        for technique, evidence_list in technique_evidence.items():
            if len(evidence_list) >= 2:
                connection = {
                    'connection_type': 'technique_pattern',
                    'technique': technique,
                    'evidence_pieces': evidence_list,
                    'connection_strength': min(1.0, len(evidence_list) * 0.4),
                    'description': f"Multiple evidence pieces suggest {technique} technique"
                }
                connections.append(connection)
        
        return connections
    
    def _construct_narratives(self, connections: List[Dict[str, Any]]) -> List[CriminalNarrative]:
        """Construct criminal narratives from connections."""
        narratives = []
        
        # Group connections by strength and type
        high_strength_connections = [c for c in connections if c['connection_strength'] > 0.7]
        
        # Construct narratives for each strong connection group
        for connection in high_strength_connections:
            if connection['connection_type'] == 'technique_pattern':
                narrative = self._construct_technique_narrative(connection)
                if narrative:
                    narratives.append(narrative)
            elif connection['connection_type'] == 'entity_multi_evidence':
                narrative = self._construct_entity_narrative(connection)
                if narrative:
                    narratives.append(narrative)
        
        # Use LLM to enhance narratives
        if self.llm_client:
            enhanced_narratives = self._enhance_narratives_with_llm(narratives)
            return enhanced_narratives
        
        return narratives
    
    def _construct_technique_narrative(self, connection: Dict[str, Any]) -> Optional[CriminalNarrative]:
        """Construct narrative based on technique pattern."""
        technique = connection['technique']
        evidence_pieces = connection['evidence_pieces']
        
        # Extract entities involved
        entities_involved = list(set(ev.entity_id for ev in evidence_pieces))
        
        # Create timeline
        timeline = []
        for evidence in sorted(evidence_pieces, key=lambda x: x.timestamp):
            timeline.append({
                'timestamp': evidence.timestamp,
                'event': evidence.description,
                'entity': evidence.entity_id,
                'source': evidence.source
            })
        
        # Generate narrative based on technique
        technique_descriptions = {
            'structuring': 'Systematic breaking of large amounts into smaller transactions to avoid reporting thresholds',
            'shell_companies': 'Use of fake or dormant companies to legitimize illegal funds',
            'money_mules': 'Recruitment of individuals to transfer money and obscure the paper trail',
            'cash_businesses': 'Integration of illegal funds through cash-intensive legitimate businesses',
            'layering': 'Complex series of transactions to distance funds from their illegal source'
        }
        
        summary = f"Investigation reveals evidence of {technique} money laundering technique. "
        summary += technique_descriptions.get(technique, 'Unknown technique employed.')
        
        narrative = CriminalNarrative(
            narrative_id=f"NARR_{technique.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Money Laundering Operation: {technique.title()} Technique",
            summary=summary,
            criminal_technique=technique,
            entities_involved=entities_involved,
            evidence_pieces=evidence_pieces,
            timeline=timeline,
            confidence_score=connection['connection_strength'],
            risk_level=self._assess_risk_level(connection['connection_strength']),
            recommended_actions=[],
            created_at=datetime.now()
        )
        
        return narrative
    
    def _construct_entity_narrative(self, connection: Dict[str, Any]) -> Optional[CriminalNarrative]:
        """Construct narrative focused on specific entity."""
        entity_id = connection['entity_id']
        evidence_pieces = connection['evidence_pieces']
        
        # Create timeline
        timeline = []
        for evidence in sorted(evidence_pieces, key=lambda x: x.timestamp):
            timeline.append({
                'timestamp': evidence.timestamp,
                'event': evidence.description,
                'entity': evidence.entity_id,
                'source': evidence.source
            })
        
        # Determine primary technique
        techniques = []
        for evidence in evidence_pieces:
            if 'structuring' in evidence.description.lower():
                techniques.append('structuring')
            elif 'shell' in evidence.description.lower():
                techniques.append('shell_companies')
            elif 'mule' in evidence.description.lower():
                techniques.append('money_mules')
        
        primary_technique = max(set(techniques), key=techniques.count) if techniques else 'unknown'
        
        summary = f"Comprehensive investigation of entity {entity_id} reveals multiple suspicious indicators. "
        summary += f"Evidence suggests involvement in {primary_technique} activities."
        
        narrative = CriminalNarrative(
            narrative_id=f"NARR_ENTITY_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Entity Investigation: {entity_id}",
            summary=summary,
            criminal_technique=primary_technique,
            entities_involved=[entity_id],
            evidence_pieces=evidence_pieces,
            timeline=timeline,
            confidence_score=connection['connection_strength'],
            risk_level=self._assess_risk_level(connection['connection_strength']),
            recommended_actions=[],
            created_at=datetime.now()
        )
        
        return narrative
    
    def _enhance_narratives_with_llm(self, narratives: List[CriminalNarrative]) -> List[CriminalNarrative]:
        """Enhance narratives using LLM reasoning."""
        enhanced_narratives = []
        
        for narrative in narratives:
            try:
                # Prepare context for LLM
                context = self._prepare_narrative_context(narrative)
                
                # Generate enhanced narrative
                enhanced_content = self._generate_enhanced_narrative(context, narrative)
                
                if enhanced_content:
                    # Update narrative with enhanced content
                    narrative.summary = enhanced_content.get('summary', narrative.summary)
                    narrative.recommended_actions = enhanced_content.get('recommended_actions', [])
                    narrative.confidence_score = min(1.0, narrative.confidence_score * enhanced_content.get('confidence_multiplier', 1.0))
                
                enhanced_narratives.append(narrative)
                
            except Exception as e:
                print(f"   ⚠️ Failed to enhance narrative {narrative.narrative_id}: {str(e)}")
                enhanced_narratives.append(narrative)
        
        return enhanced_narratives
    
    def _prepare_narrative_context(self, narrative: CriminalNarrative) -> str:
        """Prepare context for LLM narrative enhancement."""
        context = f"Criminal Narrative Analysis\n"
        context += f"========================\n\n"
        context += f"Technique: {narrative.criminal_technique}\n"
        context += f"Entities Involved: {', '.join(narrative.entities_involved)}\n"
        context += f"Current Confidence: {narrative.confidence_score:.2f}\n\n"
        
        context += f"Evidence Summary:\n"
        for i, evidence in enumerate(narrative.evidence_pieces, 1):
            context += f"{i}. {evidence.source}: {evidence.description} (Confidence: {evidence.confidence:.2f})\n"
        
        context += f"\nTimeline:\n"
        for i, event in enumerate(narrative.timeline, 1):
            context += f"{i}. {event['timestamp'].strftime('%Y-%m-%d %H:%M')}: {event['event']}\n"
        
        return context
    
    def _generate_enhanced_narrative(self, context: str, narrative: CriminalNarrative) -> Optional[Dict]:
        """Generate enhanced narrative using LLM."""
        prompt = f"""
As a senior financial crime investigator, analyze the following evidence and enhance the criminal narrative:

{context}

Please provide:
1. An enhanced summary that connects all evidence pieces into a coherent story
2. Specific recommended actions for investigators
3. Assessment of the confidence level (multiplier between 0.8 and 1.2)
4. Identification of any gaps in the evidence
5. Potential alternative explanations

Format your response as JSON:
{{
    "summary": "Enhanced narrative summary",
    "recommended_actions": ["action1", "action2", "action3"],
    "confidence_multiplier": 1.1,
    "evidence_gaps": ["gap1", "gap2"],
    "alternative_explanations": ["explanation1", "explanation2"]
}}
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are a senior financial crime investigator with expertise in money laundering detection."},
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
                return json.loads(json_str)
            
        except Exception as e:
            print(f"   ⚠️ LLM enhancement failed: {str(e)}")
        
        return None
    
    def _validate_narratives(self, narratives: List[CriminalNarrative]) -> List[CriminalNarrative]:
        """Validate and rank narratives by confidence and evidence quality."""
        validated = []
        
        for narrative in narratives:
            # Validate minimum evidence requirement
            if len(narrative.evidence_pieces) < self.config['investigation']['min_evidence_pieces']:
                continue
            
            # Validate confidence threshold
            if narrative.confidence_score < self.config['investigation']['confidence_threshold']:
                continue
            
            # Generate recommendations if not already done
            if not narrative.recommended_actions:
                narrative.recommended_actions = self._generate_recommendations_for_narrative(narrative)
            
            validated.append(narrative)
        
        # Sort by confidence score
        validated.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return validated
    
    def _generate_recommendations_for_narrative(self, narrative: CriminalNarrative) -> List[str]:
        """Generate specific recommendations for a narrative."""
        recommendations = []
        
        # Risk-based recommendations
        if narrative.risk_level == 'high':
            recommendations.extend([
                "File Suspicious Activity Report (SAR) immediately",
                "Freeze accounts pending investigation",
                "Coordinate with law enforcement"
            ])
        elif narrative.risk_level == 'medium':
            recommendations.extend([
                "Enhanced monitoring of involved entities",
                "Gather additional evidence",
                "Consider SAR filing if more evidence emerges"
            ])
        
        # Technique-specific recommendations
        if narrative.criminal_technique == 'structuring':
            recommendations.append("Review all transactions below $10,000 from involved entities")
        elif narrative.criminal_technique == 'shell_companies':
            recommendations.append("Investigate business registration and beneficial ownership")
        elif narrative.criminal_technique == 'money_mules':
            recommendations.append("Trace funds to ultimate beneficiaries")
        
        return recommendations
    
    def _assess_risk_level(self, confidence_score: float) -> str:
        """Assess risk level based on confidence score."""
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, narratives: List[CriminalNarrative]):
        """Generate overall investigation recommendations."""
        if not narratives:
            return
        
        # Update investigation context
        self.investigation_context = {
            'total_narratives': len(narratives),
            'high_risk_narratives': len([n for n in narratives if n.risk_level == 'high']),
            'primary_techniques': list(set(n.criminal_technique for n in narratives)),
            'entities_of_interest': list(set(entity for n in narratives for entity in n.entities_involved))
        }
    
    def get_investigation_report(self) -> str:
        """Generate comprehensive investigation report."""
        if not self.narratives:
            return "No investigation has been conducted yet."
        
        report = f"""
LEAD INVESTIGATOR REPORT
=======================

Investigation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Evidence Pieces: {len(self.evidence_vault)}
Criminal Narratives Constructed: {len(self.narratives)}

EXECUTIVE SUMMARY
================
"""
        
        # High-level summary
        high_risk_narratives = [n for n in self.narratives if n.risk_level == 'high']
        if high_risk_narratives:
            report += f"🚨 {len(high_risk_narratives)} HIGH-RISK criminal narratives identified requiring immediate action.\n"
        
        medium_risk_narratives = [n for n in self.narratives if n.risk_level == 'medium']
        if medium_risk_narratives:
            report += f"⚠️ {len(medium_risk_narratives)} MEDIUM-RISK narratives require enhanced monitoring.\n"
        
        # Primary techniques
        techniques = list(set(n.criminal_technique for n in self.narratives))
        report += f"\nPrimary techniques identified: {', '.join(techniques)}\n"
        
        # Entities of interest
        all_entities = list(set(entity for n in self.narratives for entity in n.entities_involved))
        report += f"Entities of interest: {len(all_entities)}\n"
        
        report += f"\nCRIMINAL NARRATIVES\n"
        report += f"==================\n"
        
        # Detailed narratives
        for i, narrative in enumerate(self.narratives, 1):
            report += f"\n{i}. {narrative.title}\n"
            report += f"   Risk Level: {narrative.risk_level.upper()}\n"
            report += f"   Confidence: {narrative.confidence_score:.2f}\n"
            report += f"   Technique: {narrative.criminal_technique}\n"
            report += f"   Entities: {', '.join(narrative.entities_involved)}\n"
            report += f"   Evidence Pieces: {len(narrative.evidence_pieces)}\n"
            report += f"   Summary: {narrative.summary}\n"
            
            if narrative.recommended_actions:
                report += f"   Recommended Actions:\n"
                for action in narrative.recommended_actions:
                    report += f"     • {action}\n"
        
        report += f"\nOVERALL RECOMMENDATIONS\n"
        report += f"======================\n"
        
        if high_risk_narratives:
            report += f"1. IMMEDIATE ACTION REQUIRED for {len(high_risk_narratives)} high-risk cases\n"
            report += f"2. File SARs for all high-risk entities\n"
            report += f"3. Coordinate with law enforcement\n"
        
        if medium_risk_narratives:
            report += f"4. Enhanced monitoring for {len(medium_risk_narratives)} medium-risk cases\n"
        
        report += f"5. Continue investigation to gather additional evidence\n"
        report += f"6. Regular review of all flagged entities\n"
        
        return report
    
    def export_narratives(self, output_file: str):
        """Export narratives to JSON file."""
        export_data = {
            'investigation_timestamp': datetime.now().isoformat(),
            'total_evidence_pieces': len(self.evidence_vault),
            'narratives': [
                {
                    'narrative_id': n.narrative_id,
                    'title': n.title,
                    'summary': n.summary,
                    'criminal_technique': n.criminal_technique,
                    'entities_involved': n.entities_involved,
                    'confidence_score': n.confidence_score,
                    'risk_level': n.risk_level,
                    'recommended_actions': n.recommended_actions,
                    'evidence_count': len(n.evidence_pieces),
                    'timeline_events': len(n.timeline),
                    'created_at': n.created_at.isoformat()
                }
                for n in self.narratives
            ],
            'investigation_context': self.investigation_context
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Investigation narratives exported to {output_file}")


def main():
    """Main function for testing the Lead Investigator."""
    print("Testing Blue Team Lead Investigator...")
    
    # Sample transaction analysis results
    transaction_analysis = {
        'suspicious_entities': [
            {
                'entity_id': 'SHELL_001',
                'reason': 'shell_company_pattern',
                'risk_score': 0.85,
                'detection_method': 'business_analysis'
            },
            {
                'entity_id': 'MULE_002',
                'reason': 'rapid_money_movement',
                'risk_score': 0.75,
                'detection_method': 'transaction_analysis'
            }
        ],
        'anomalies_detected': {
            'transactions': [
                {
                    'sender_id': 'SHELL_001',
                    'anomaly_score': -0.8,
                    'description': 'Unusual transaction pattern'
                }
            ]
        },
        'network_analysis': {
            'suspicious_subgraphs': [
                {
                    'nodes': ['SHELL_001', 'MULE_002'],
                    'suspicion_reason': 'highly_connected_community'
                }
            ]
        }
    }
    
    # Sample OSINT results
    osint_results = {
        'SHELL_001': [
            type('OSINTResult', (), {
                'information_type': 'business_registration',
                'content': 'Shell company with minimal legitimate activity',
                'relevance_score': 0.9,
                'timestamp': datetime.now(),
                'source': 'Business Registry',
                'url': 'https://registry.example.com'
            })()
        ]
    }
    
    # Initialize Lead Investigator
    investigator = LeadInvestigator()
    
    # Conduct investigation
    print("\nConducting investigation...")
    narratives = investigator.investigate_case(transaction_analysis, osint_results)
    
    # Display results
    print("\nInvestigation Report:")
    print("=" * 80)
    print(investigator.get_investigation_report())
    
    # Export results
    investigator.export_narratives("investigation_narratives.json")


if __name__ == "__main__":
    main() 