"""
Blue Team OSINT Agent

This agent performs Open Source Intelligence (OSINT) gathering on suspicious entities
identified by the Transaction Analyst. It searches for external information that can
provide context and additional evidence for investigations.
"""

import requests
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path

# For web scraping (if needed)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# For LLM integration
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None


@dataclass
class OSINTResult:
    """Data class for OSINT search results."""
    entity_id: str
    entity_name: str
    source: str
    information_type: str
    content: str
    relevance_score: float
    timestamp: datetime
    url: Optional[str] = None
    confidence: float = 0.0


class OSINTAgent:
    """
    Performs Open Source Intelligence gathering on suspicious entities.
    
    This agent searches various sources for information about entities flagged
    as suspicious by the Transaction Analyst, providing context and additional
    evidence for investigations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the OSINT Agent."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.search_results = []
        self.entity_profiles = {}
        
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
                'max_tokens': 2000
            },
            'osint': {
                'max_results_per_entity': 10,
                'search_timeout': 30,
                'relevance_threshold': 0.5,
                'sources_enabled': [
                    'sanctions_lists',
                    'news_search',
                    'business_registry',
                    'social_media',
                    'court_records'
                ]
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for content analysis."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def investigate_entities(self, suspicious_entities: List[Dict]) -> Dict[str, List[OSINTResult]]:
        """
        Investigate a list of suspicious entities.
        
        Args:
            suspicious_entities: List of entities from Transaction Analyst
            
        Returns:
            Dictionary mapping entity IDs to their OSINT results
        """
        print(f"🔍 Starting OSINT investigation on {len(suspicious_entities)} entities...")
        
        investigation_results = {}
        
        for entity in suspicious_entities:
            entity_id = entity.get('entity_id', 'unknown')
            entity_name = self._extract_entity_name(entity)
            
            print(f"   🔎 Investigating {entity_name} ({entity_id})...")
            
            # Perform OSINT searches
            results = self._investigate_single_entity(entity_id, entity_name, entity)
            investigation_results[entity_id] = results
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        print(f"✅ OSINT investigation completed. Found {sum(len(results) for results in investigation_results.values())} pieces of intelligence.")
        
        return investigation_results
    
    def _extract_entity_name(self, entity: Dict) -> str:
        """Extract entity name from entity data."""
        # Try various possible name fields
        name_fields = ['name', 'entity_name', 'sender_name', 'receiver_name', 'business_name']
        
        for field in name_fields:
            if field in entity and entity[field]:
                return entity[field]
        
        # Fallback to entity ID
        return entity.get('entity_id', 'Unknown Entity')
    
    def _investigate_single_entity(self, entity_id: str, entity_name: str, entity_data: Dict) -> List[OSINTResult]:
        """Investigate a single entity across multiple sources."""
        results = []
        
        # Search different sources
        for source in self.config['osint']['sources_enabled']:
            try:
                if source == 'sanctions_lists':
                    source_results = self._search_sanctions_lists(entity_id, entity_name)
                elif source == 'news_search':
                    source_results = self._search_news(entity_id, entity_name)
                elif source == 'business_registry':
                    source_results = self._search_business_registry(entity_id, entity_name)
                elif source == 'social_media':
                    source_results = self._search_social_media(entity_id, entity_name)
                elif source == 'court_records':
                    source_results = self._search_court_records(entity_id, entity_name)
                else:
                    source_results = []
                
                results.extend(source_results)
                
            except Exception as e:
                print(f"   ⚠️ Error searching {source}: {str(e)}")
                continue
        
        # Analyze and score results
        analyzed_results = self._analyze_results(results, entity_data)
        
        # Store in entity profiles
        self._build_entity_profile(entity_id, entity_name, analyzed_results)
        
        return analyzed_results
    
    def _search_sanctions_lists(self, entity_id: str, entity_name: str) -> List[OSINTResult]:
        """Search international sanctions lists."""
        results = []
        
        # Simulate searching major sanctions lists
        sanctions_lists = [
            "OFAC Specially Designated Nationals List",
            "UN Security Council Consolidated List",
            "EU Consolidated List",
            "UK HM Treasury Consolidated List",
            "FATF High-Risk Jurisdictions"
        ]
        
        # Simulate potential matches (in real implementation, this would query actual APIs)
        for sanctions_list in sanctions_lists:
            # Simulate low probability of sanctions match
            if random.random() < 0.05:  # 5% chance of match
                match_type = random.choice(['exact_match', 'partial_match', 'alias_match'])
                
                result = OSINTResult(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    source=sanctions_list,
                    information_type='sanctions_match',
                    content=f"Potential {match_type} found in {sanctions_list}. Entity may be subject to financial sanctions.",
                    relevance_score=0.95,
                    timestamp=datetime.now(),
                    url=f"https://sanctions.example.com/search?q={entity_name.replace(' ', '+')}"
                )
                results.append(result)
        
        return results
    
    def _search_news(self, entity_id: str, entity_name: str) -> List[OSINTResult]:
        """Search news articles for entity mentions."""
        results = []
        
        # Simulate news search results
        news_scenarios = [
            {
                'type': 'financial_crime',
                'content': f"News article mentions {entity_name} in connection with financial irregularities investigation.",
                'relevance': 0.8
            },
            {
                'type': 'business_news',
                'content': f"Business article discusses {entity_name}'s recent corporate activities and partnerships.",
                'relevance': 0.6
            },
            {
                'type': 'regulatory_action',
                'content': f"Regulatory filing mentions {entity_name} in compliance-related matter.",
                'relevance': 0.7
            },
            {
                'type': 'court_case',
                'content': f"Court documents reference {entity_name} in civil litigation proceedings.",
                'relevance': 0.75
            }
        ]
        
        # Simulate finding 0-2 news mentions
        num_results = random.randint(0, 2)
        for _ in range(num_results):
            scenario = random.choice(news_scenarios)
            
            result = OSINTResult(
                entity_id=entity_id,
                entity_name=entity_name,
                source='News Search',
                information_type=scenario['type'],
                content=scenario['content'],
                relevance_score=scenario['relevance'],
                timestamp=datetime.now() - timedelta(days=random.randint(1, 365)),
                url=f"https://news.example.com/article/{random.randint(1000, 9999)}"
            )
            results.append(result)
        
        return results
    
    def _search_business_registry(self, entity_id: str, entity_name: str) -> List[OSINTResult]:
        """Search business registries for entity information."""
        results = []
        
        # Simulate business registry search
        if random.random() < 0.7:  # 70% chance of finding business info
            registry_info = {
                'registration_date': datetime.now() - timedelta(days=random.randint(30, 3650)),
                'business_type': random.choice(['LLC', 'Corporation', 'Partnership', 'Sole Proprietorship']),
                'status': random.choice(['Active', 'Inactive', 'Suspended', 'Dissolved']),
                'registered_address': f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'First', 'Second'])} St",
                'officers': [f"Officer {i+1}" for i in range(random.randint(1, 3))]
            }
            
            content = f"Business registry information for {entity_name}:\n"
            content += f"- Registration Date: {registry_info['registration_date'].strftime('%Y-%m-%d')}\n"
            content += f"- Business Type: {registry_info['business_type']}\n"
            content += f"- Status: {registry_info['status']}\n"
            content += f"- Address: {registry_info['registered_address']}\n"
            content += f"- Officers: {', '.join(registry_info['officers'])}"
            
            # Higher relevance if business is inactive/suspended
            relevance = 0.9 if registry_info['status'] in ['Inactive', 'Suspended', 'Dissolved'] else 0.6
            
            result = OSINTResult(
                entity_id=entity_id,
                entity_name=entity_name,
                source='Business Registry',
                information_type='business_registration',
                content=content,
                relevance_score=relevance,
                timestamp=datetime.now(),
                url=f"https://registry.example.com/business/{entity_id}"
            )
            results.append(result)
        
        return results
    
    def _search_social_media(self, entity_id: str, entity_name: str) -> List[OSINTResult]:
        """Search social media for entity mentions."""
        results = []
        
        # Simulate social media search
        social_platforms = ['LinkedIn', 'Twitter', 'Facebook', 'Instagram']
        
        for platform in social_platforms:
            if random.random() < 0.3:  # 30% chance of finding social media presence
                social_info = random.choice([
                    f"Professional profile found on {platform} for {entity_name}",
                    f"Recent posts on {platform} mention financial difficulties",
                    f"Network connections on {platform} include known high-risk individuals",
                    f"Profile shows inconsistent location information",
                    f"Recent activity suggests business operations in high-risk jurisdictions"
                ])
                
                result = OSINTResult(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    source=f'Social Media ({platform})',
                    information_type='social_media_profile',
                    content=social_info,
                    relevance_score=0.4,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 90)),
                    url=f"https://{platform.lower()}.com/profile/{entity_name.replace(' ', '')}"
                )
                results.append(result)
        
        return results
    
    def _search_court_records(self, entity_id: str, entity_name: str) -> List[OSINTResult]:
        """Search court records for entity involvement."""
        results = []
        
        # Simulate court records search
        if random.random() < 0.2:  # 20% chance of court involvement
            court_cases = [
                {
                    'type': 'civil_litigation',
                    'content': f"{entity_name} involved in civil litigation regarding breach of contract",
                    'relevance': 0.5
                },
                {
                    'type': 'bankruptcy',
                    'content': f"Bankruptcy filing found for {entity_name}",
                    'relevance': 0.8
                },
                {
                    'type': 'criminal_case',
                    'content': f"{entity_name} mentioned in criminal case as witness or associate",
                    'relevance': 0.9
                },
                {
                    'type': 'regulatory_violation',
                    'content': f"Regulatory violation case involving {entity_name}",
                    'relevance': 0.85
                }
            ]
            
            case = random.choice(court_cases)
            
            result = OSINTResult(
                entity_id=entity_id,
                entity_name=entity_name,
                source='Court Records',
                information_type=case['type'],
                content=case['content'],
                relevance_score=case['relevance'],
                timestamp=datetime.now() - timedelta(days=random.randint(30, 1095)),
                url=f"https://courts.example.com/case/{random.randint(1000, 9999)}"
            )
            results.append(result)
        
        return results
    
    def _analyze_results(self, results: List[OSINTResult], entity_data: Dict) -> List[OSINTResult]:
        """Analyze and enhance OSINT results using LLM."""
        if not self.llm_client or not results:
            return results
        
        # Prepare context for LLM analysis
        context = f"Entity: {results[0].entity_name}\n"
        context += f"Risk Score: {entity_data.get('risk_score', 'Unknown')}\n"
        context += f"Detection Reason: {entity_data.get('reason', 'Unknown')}\n\n"
        context += "OSINT Findings:\n"
        
        for result in results:
            context += f"- {result.source}: {result.content}\n"
        
        # Analyze with LLM
        analysis_prompt = f"""
Analyze the following OSINT findings for a suspicious financial entity:

{context}

Please provide:
1. Overall risk assessment (Low/Medium/High)
2. Key risk indicators identified
3. Recommended follow-up actions
4. Confidence level in the assessment

Format your response as JSON with the following structure:
{{
    "risk_assessment": "High/Medium/Low",
    "risk_indicators": ["indicator1", "indicator2"],
    "recommended_actions": ["action1", "action2"],
    "confidence_level": 0.8,
    "summary": "Brief summary of findings"
}}
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are a financial crime analyst reviewing OSINT findings."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )
            
            analysis_content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                json_start = analysis_content.find('{')
                json_end = analysis_content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    analysis_json = json.loads(analysis_content[json_start:json_end])
                    
                    # Add analysis as a special result
                    analysis_result = OSINTResult(
                        entity_id=results[0].entity_id,
                        entity_name=results[0].entity_name,
                        source='LLM Analysis',
                        information_type='comprehensive_analysis',
                        content=analysis_json.get('summary', 'Analysis completed'),
                        relevance_score=1.0,
                        timestamp=datetime.now(),
                        confidence=analysis_json.get('confidence_level', 0.5)
                    )
                    results.append(analysis_result)
                    
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            print(f"   ⚠️ LLM analysis failed: {str(e)}")
        
        return results
    
    def _build_entity_profile(self, entity_id: str, entity_name: str, results: List[OSINTResult]):
        """Build comprehensive entity profile from OSINT results."""
        profile = {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'investigation_date': datetime.now(),
            'total_sources': len(set(result.source for result in results)),
            'total_findings': len(results),
            'highest_relevance': max([result.relevance_score for result in results], default=0),
            'risk_indicators': [],
            'information_types': list(set(result.information_type for result in results)),
            'sources_searched': list(set(result.source for result in results)),
            'findings_summary': {}
        }
        
        # Categorize findings
        for result in results:
            if result.information_type not in profile['findings_summary']:
                profile['findings_summary'][result.information_type] = []
            profile['findings_summary'][result.information_type].append({
                'source': result.source,
                'content': result.content,
                'relevance': result.relevance_score,
                'timestamp': result.timestamp
            })
        
        # Identify risk indicators
        for result in results:
            if result.relevance_score > 0.7:
                profile['risk_indicators'].append({
                    'indicator': result.information_type,
                    'source': result.source,
                    'score': result.relevance_score
                })
        
        self.entity_profiles[entity_id] = profile
    
    def get_investigation_summary(self, entity_id: str) -> str:
        """Generate human-readable investigation summary."""
        if entity_id not in self.entity_profiles:
            return f"No OSINT investigation found for entity {entity_id}"
        
        profile = self.entity_profiles[entity_id]
        
        summary = f"""
OSINT Investigation Summary
==========================

Entity: {profile['entity_name']} ({profile['entity_id']})
Investigation Date: {profile['investigation_date'].strftime('%Y-%m-%d %H:%M:%S')}

Overview:
- Sources Searched: {profile['total_sources']}
- Total Findings: {profile['total_findings']}
- Highest Relevance Score: {profile['highest_relevance']:.2f}

Information Types Found:
{chr(10).join(f"  • {info_type}" for info_type in profile['information_types'])}

Risk Indicators:
"""
        
        if profile['risk_indicators']:
            for indicator in profile['risk_indicators']:
                summary += f"  • {indicator['indicator']} (Score: {indicator['score']:.2f}, Source: {indicator['source']})\n"
        else:
            summary += "  • No high-risk indicators identified\n"
        
        summary += "\nDetailed Findings:\n"
        for info_type, findings in profile['findings_summary'].items():
            summary += f"\n{info_type.replace('_', ' ').title()}:\n"
            for finding in findings:
                summary += f"  - {finding['source']}: {finding['content'][:100]}...\n"
        
        return summary
    
    def get_all_results(self) -> List[OSINTResult]:
        """Get all OSINT results collected."""
        return self.search_results
    
    def export_results(self, output_file: str):
        """Export OSINT results to JSON file."""
        export_data = {
            'investigation_timestamp': datetime.now().isoformat(),
            'entity_profiles': {k: {**v, 'investigation_date': v['investigation_date'].isoformat()} 
                             for k, v in self.entity_profiles.items()},
            'total_entities_investigated': len(self.entity_profiles),
            'total_findings': sum(profile['total_findings'] for profile in self.entity_profiles.values())
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ OSINT results exported to {output_file}")


def main():
    """Main function for testing the OSINT Agent."""
    print("Testing Blue Team OSINT Agent...")
    
    # Create sample suspicious entities
    suspicious_entities = [
        {
            'entity_id': 'SUSPICIOUS_001',
            'entity_name': 'Phantom Consulting LLC',
            'risk_score': 0.85,
            'reason': 'shell_company_pattern',
            'detection_method': 'transaction_analysis'
        },
        {
            'entity_id': 'MULE_ABC123',
            'entity_name': 'John Doe',
            'risk_score': 0.72,
            'reason': 'money_mule_behavior',
            'detection_method': 'network_analysis'
        }
    ]
    
    # Initialize OSINT agent
    osint_agent = OSINTAgent()
    
    # Perform investigation
    print("\nPerforming OSINT investigation...")
    results = osint_agent.investigate_entities(suspicious_entities)
    
    # Display results
    print("\nInvestigation Results:")
    print("=" * 60)
    
    for entity_id, entity_results in results.items():
        print(f"\n{entity_id}:")
        print(osint_agent.get_investigation_summary(entity_id))
    
    # Export results
    osint_agent.export_results("osint_results.json")


if __name__ == "__main__":
    main() 