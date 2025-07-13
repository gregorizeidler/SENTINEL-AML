"""
Red Team Social Engineering Agent

This agent performs sophisticated social engineering attacks to complement
the technical money laundering schemes. It creates convincing personas,
designs phishing campaigns, and exploits human vulnerabilities to enhance
the realism of Red Team operations.
"""

import json
import yaml
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from faker import Faker
import uuid

# LLM imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None


@dataclass
class SocialPersona:
    """Data class for social engineering personas."""
    persona_id: str
    name: str
    role: str
    organization: str
    background_story: str
    personality_traits: List[str]
    communication_style: str
    credentials: Dict[str, str]
    social_media_profiles: Dict[str, str]
    credibility_score: float
    created_at: datetime
    last_used: Optional[datetime] = None


@dataclass
class SocialEngineeringCampaign:
    """Data class for social engineering campaigns."""
    campaign_id: str
    campaign_type: str  # 'phishing', 'pretexting', 'baiting', 'tailgating'
    target_profile: str
    personas_used: List[str]
    attack_vector: str
    narrative: str
    success_metrics: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    resources_required: List[str]
    risk_level: str
    created_at: datetime
    status: str = "planned"


@dataclass
class SocialEngineeringAttack:
    """Data class for social engineering attacks."""
    attack_id: str
    campaign_id: str
    attack_type: str
    target_entity: str
    persona_used: str
    attack_method: str
    payload: Dict[str, Any]
    success_probability: float
    executed_at: datetime
    outcome: str = "pending"
    evidence_collected: List[str] = None
    follow_up_actions: List[str] = None


class SocialEngineeringAgent:
    """
    Performs sophisticated social engineering attacks for Red Team operations.
    
    This agent creates convincing personas, designs social engineering campaigns,
    and executes human-based attacks to support money laundering operations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Social Engineering Agent."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.fake = Faker()
        self.personas = {}
        self.campaigns = []
        self.attacks = []
        self.target_profiles = self._load_target_profiles()
        self.attack_templates = self._load_attack_templates()
        self.social_tactics = self._load_social_tactics()
        
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
                'temperature': 0.7,
                'max_tokens': 3000
            },
            'social_engineering': {
                'persona_types': ['executive', 'it_support', 'auditor', 'vendor', 'customer'],
                'attack_sophistication': 'high',
                'target_research_depth': 'thorough',
                'campaign_duration_days': 14,
                'success_threshold': 0.7
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for social engineering."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def _load_target_profiles(self) -> Dict[str, Dict]:
        """Load target profiles for social engineering."""
        return {
            'bank_employee': {
                'role_types': ['teller', 'manager', 'compliance_officer', 'it_admin'],
                'vulnerabilities': ['authority_compliance', 'urgency_pressure', 'helpful_nature'],
                'communication_channels': ['email', 'phone', 'in_person'],
                'typical_requests': ['password_reset', 'account_access', 'system_maintenance']
            },
            'customer': {
                'role_types': ['individual', 'business_owner', 'elderly', 'tech_savvy'],
                'vulnerabilities': ['trust_authority', 'fear_loss', 'greed'],
                'communication_channels': ['email', 'phone', 'sms', 'social_media'],
                'typical_requests': ['account_verification', 'security_update', 'reward_claim']
            },
            'vendor': {
                'role_types': ['it_support', 'consultant', 'auditor', 'maintenance'],
                'vulnerabilities': ['business_relationship', 'deadline_pressure', 'technical_complexity'],
                'communication_channels': ['email', 'phone', 'remote_access'],
                'typical_requests': ['system_access', 'data_export', 'configuration_change']
            },
            'regulator': {
                'role_types': ['examiner', 'investigator', 'compliance_reviewer'],
                'vulnerabilities': ['authority_respect', 'compliance_fear', 'urgency'],
                'communication_channels': ['official_email', 'phone', 'secure_portal'],
                'typical_requests': ['data_request', 'system_access', 'document_submission']
            }
        }
    
    def _load_attack_templates(self) -> Dict[str, Dict]:
        """Load social engineering attack templates."""
        return {
            'phishing_email': {
                'description': 'Fraudulent email to steal credentials or information',
                'components': ['sender_spoofing', 'urgent_subject', 'credible_content', 'malicious_link'],
                'success_indicators': ['click_rate', 'credential_harvest', 'malware_install'],
                'difficulty': 'medium'
            },
            'vishing_call': {
                'description': 'Voice phishing to extract information over phone',
                'components': ['caller_id_spoofing', 'authority_persona', 'urgency_script', 'information_request'],
                'success_indicators': ['information_disclosure', 'action_compliance', 'callback_success'],
                'difficulty': 'high'
            },
            'pretexting': {
                'description': 'Fabricated scenario to gain trust and information',
                'components': ['believable_story', 'authority_figure', 'logical_request', 'trust_building'],
                'success_indicators': ['trust_establishment', 'information_sharing', 'action_compliance'],
                'difficulty': 'high'
            },
            'baiting': {
                'description': 'Offering something enticing to trigger malicious action',
                'components': ['attractive_offer', 'urgency_factor', 'legitimate_appearance', 'payload_delivery'],
                'success_indicators': ['engagement_rate', 'payload_execution', 'data_exfiltration'],
                'difficulty': 'medium'
            },
            'tailgating': {
                'description': 'Following authorized person into secure area',
                'components': ['timing_coordination', 'distraction_technique', 'authority_appearance', 'confidence_display'],
                'success_indicators': ['access_gained', 'detection_avoided', 'objective_achieved'],
                'difficulty': 'low'
            },
            'watering_hole': {
                'description': 'Compromising websites frequently visited by targets',
                'components': ['target_research', 'website_compromise', 'payload_injection', 'user_exploitation'],
                'success_indicators': ['site_compromise', 'user_infection', 'data_collection'],
                'difficulty': 'very_high'
            }
        }
    
    def _load_social_tactics(self) -> Dict[str, Dict]:
        """Load social engineering tactics and techniques."""
        return {
            'authority': {
                'description': 'Impersonating authority figures to gain compliance',
                'effectiveness': 0.8,
                'personas': ['executive', 'regulator', 'it_admin', 'auditor'],
                'techniques': ['title_dropping', 'urgent_demands', 'consequence_threats']
            },
            'urgency': {
                'description': 'Creating time pressure to bypass normal verification',
                'effectiveness': 0.7,
                'personas': ['executive', 'it_support', 'vendor'],
                'techniques': ['deadline_pressure', 'emergency_scenario', 'limited_time_offer']
            },
            'reciprocity': {
                'description': 'Offering help or gifts to create obligation',
                'effectiveness': 0.6,
                'personas': ['vendor', 'consultant', 'customer'],
                'techniques': ['free_service', 'helpful_gesture', 'information_sharing']
            },
            'social_proof': {
                'description': 'Showing others have already complied',
                'effectiveness': 0.6,
                'personas': ['consultant', 'auditor', 'vendor'],
                'techniques': ['peer_compliance', 'industry_standard', 'common_practice']
            },
            'liking': {
                'description': 'Building rapport and similarity to gain trust',
                'effectiveness': 0.7,
                'personas': ['customer', 'vendor', 'consultant'],
                'techniques': ['common_interests', 'shared_experiences', 'compliments']
            },
            'scarcity': {
                'description': 'Creating perception of limited availability',
                'effectiveness': 0.5,
                'personas': ['vendor', 'consultant', 'customer'],
                'techniques': ['limited_offer', 'exclusive_access', 'rare_opportunity']
            }
        }
    
    def create_social_persona(self, persona_type: str, target_context: str = None) -> SocialPersona:
        """
        Create a convincing social engineering persona.
        
        Args:
            persona_type: Type of persona to create
            target_context: Context for targeting specific organizations
            
        Returns:
            SocialPersona object
        """
        print(f"🎭 Creating social engineering persona: {persona_type}")
        
        persona_id = f"PERSONA_{uuid.uuid4().hex[:8].upper()}"
        
        # Generate basic persona details
        if persona_type == 'executive':
            persona = self._create_executive_persona(persona_id, target_context)
        elif persona_type == 'it_support':
            persona = self._create_it_support_persona(persona_id, target_context)
        elif persona_type == 'auditor':
            persona = self._create_auditor_persona(persona_id, target_context)
        elif persona_type == 'vendor':
            persona = self._create_vendor_persona(persona_id, target_context)
        elif persona_type == 'customer':
            persona = self._create_customer_persona(persona_id, target_context)
        else:
            persona = self._create_generic_persona(persona_id, persona_type, target_context)
        
        # Enhance persona with LLM if available
        if self.llm_client:
            persona = self._enhance_persona_with_llm(persona)
        
        # Store persona
        self.personas[persona_id] = persona
        
        print(f"✅ Created persona: {persona.name} ({persona.role})")
        return persona
    
    def _create_executive_persona(self, persona_id: str, target_context: str) -> SocialPersona:
        """Create executive persona."""
        profile = self.fake.profile()
        
        executive_titles = [
            'Chief Executive Officer', 'Chief Financial Officer', 'Chief Technology Officer',
            'Vice President', 'Senior Vice President', 'Managing Director',
            'Regional Director', 'Division Head', 'Senior Manager'
        ]
        
        organizations = [
            'Global Financial Services Inc.', 'International Banking Group',
            'Corporate Finance Solutions', 'Executive Consulting Partners',
            'Strategic Investment Holdings', 'Premier Financial Advisory'
        ]
        
        return SocialPersona(
            persona_id=persona_id,
            name=profile['name'],
            role=random.choice(executive_titles),
            organization=target_context or random.choice(organizations),
            background_story=f"Experienced {random.choice(['finance', 'banking', 'investment'])} executive with {random.randint(10, 25)} years of industry experience.",
            personality_traits=['authoritative', 'decisive', 'results-oriented', 'demanding'],
            communication_style='direct and commanding',
            credentials={
                'education': f"{random.choice(['MBA', 'CFA', 'CPA'])} from {random.choice(['Harvard', 'Wharton', 'Stanford'])}",
                'certifications': random.choice(['CFA', 'CPA', 'FRM', 'PMP']),
                'experience': f"{random.randint(15, 30)} years in financial services"
            },
            social_media_profiles={
                'linkedin': f"linkedin.com/in/{profile['username']}",
                'twitter': f"@{profile['username']}_exec"
            },
            credibility_score=0.9,
            created_at=datetime.now()
        )
    
    def _create_it_support_persona(self, persona_id: str, target_context: str) -> SocialPersona:
        """Create IT support persona."""
        profile = self.fake.profile()
        
        it_roles = [
            'IT Support Specialist', 'System Administrator', 'Network Engineer',
            'Security Analyst', 'Help Desk Technician', 'Database Administrator',
            'Cloud Engineer', 'DevOps Engineer', 'IT Manager'
        ]
        
        return SocialPersona(
            persona_id=persona_id,
            name=profile['name'],
            role=random.choice(it_roles),
            organization=target_context or f"{random.choice(['TechCorp', 'InfoSystems', 'DataTech'])} IT Services",
            background_story=f"IT professional specializing in {random.choice(['network security', 'system administration', 'database management'])} with {random.randint(5, 15)} years of experience.",
            personality_traits=['technical', 'helpful', 'methodical', 'patient'],
            communication_style='technical and supportive',
            credentials={
                'education': f"{random.choice(['BS Computer Science', 'BS Information Technology', 'BS Engineering'])}",
                'certifications': random.choice(['CISSP', 'CCNA', 'CompTIA Security+', 'AWS Certified']),
                'experience': f"{random.randint(5, 20)} years in IT support"
            },
            social_media_profiles={
                'linkedin': f"linkedin.com/in/{profile['username']}",
                'github': f"github.com/{profile['username']}"
            },
            credibility_score=0.8,
            created_at=datetime.now()
        )
    
    def _create_auditor_persona(self, persona_id: str, target_context: str) -> SocialPersona:
        """Create auditor persona."""
        profile = self.fake.profile()
        
        auditor_roles = [
            'Senior Auditor', 'Compliance Auditor', 'Internal Auditor',
            'Risk Auditor', 'IT Auditor', 'Financial Auditor',
            'Regulatory Examiner', 'Audit Manager', 'Principal Auditor'
        ]
        
        audit_firms = [
            'Deloitte', 'PwC', 'EY', 'KPMG', 'BDO', 'Grant Thornton',
            'RSM', 'Crowe', 'Mazars', 'Baker Tilly'
        ]
        
        return SocialPersona(
            persona_id=persona_id,
            name=profile['name'],
            role=random.choice(auditor_roles),
            organization=target_context or random.choice(audit_firms),
            background_story=f"Professional auditor with expertise in {random.choice(['financial auditing', 'compliance', 'risk management'])} and {random.randint(8, 20)} years of experience.",
            personality_traits=['detail-oriented', 'methodical', 'authoritative', 'persistent'],
            communication_style='formal and thorough',
            credentials={
                'education': f"{random.choice(['CPA', 'CA', 'ACCA'])} qualification",
                'certifications': random.choice(['CPA', 'CIA', 'CISA', 'CFE']),
                'experience': f"{random.randint(8, 25)} years in auditing"
            },
            social_media_profiles={
                'linkedin': f"linkedin.com/in/{profile['username']}",
                'professional_org': f"Member of {random.choice(['AICPA', 'IIA', 'ISACA'])}"
            },
            credibility_score=0.85,
            created_at=datetime.now()
        )
    
    def _create_vendor_persona(self, persona_id: str, target_context: str) -> SocialPersona:
        """Create vendor persona."""
        profile = self.fake.profile()
        
        vendor_roles = [
            'Account Manager', 'Sales Representative', 'Technical Consultant',
            'Implementation Specialist', 'Customer Success Manager', 'Support Engineer',
            'Business Development Manager', 'Solution Architect', 'Project Manager'
        ]
        
        vendor_companies = [
            'Microsoft', 'Oracle', 'IBM', 'Salesforce', 'SAP', 'Cisco',
            'Dell', 'HP', 'VMware', 'Adobe', 'ServiceNow', 'Workday'
        ]
        
        return SocialPersona(
            persona_id=persona_id,
            name=profile['name'],
            role=random.choice(vendor_roles),
            organization=target_context or random.choice(vendor_companies),
            background_story=f"Vendor representative with {random.randint(5, 15)} years of experience in {random.choice(['technology sales', 'customer support', 'implementation services'])}.",
            personality_traits=['persuasive', 'customer-focused', 'knowledgeable', 'persistent'],
            communication_style='professional and engaging',
            credentials={
                'education': f"{random.choice(['BS Business', 'BS Marketing', 'BS Engineering'])}",
                'certifications': random.choice(['Salesforce Certified', 'Microsoft Certified', 'AWS Partner']),
                'experience': f"{random.randint(5, 20)} years in vendor relations"
            },
            social_media_profiles={
                'linkedin': f"linkedin.com/in/{profile['username']}",
                'company_email': f"{profile['username']}@{random.choice(['microsoft.com', 'oracle.com', 'salesforce.com'])}"
            },
            credibility_score=0.75,
            created_at=datetime.now()
        )
    
    def _create_customer_persona(self, persona_id: str, target_context: str) -> SocialPersona:
        """Create customer persona."""
        profile = self.fake.profile()
        
        customer_types = [
            'Individual Customer', 'Business Owner', 'Corporate Treasurer',
            'Account Holder', 'Premium Customer', 'Long-term Client',
            'High-value Customer', 'Retail Customer', 'Private Banking Client'
        ]
        
        return SocialPersona(
            persona_id=persona_id,
            name=profile['name'],
            role=random.choice(customer_types),
            organization=target_context or f"{profile['company']} (Customer)",
            background_story=f"Long-standing customer with {random.randint(5, 20)} years of banking relationship and {random.choice(['personal', 'business', 'investment'])} accounts.",
            personality_traits=['trusting', 'concerned', 'loyal', 'security-conscious'],
            communication_style='polite and concerned',
            credentials={
                'account_type': random.choice(['Premium', 'Business', 'Private Banking', 'Standard']),
                'relationship_length': f"{random.randint(5, 25)} years",
                'account_value': f"${random.randint(50000, 2000000):,}"
            },
            social_media_profiles={
                'email': profile['mail'],
                'phone': self.fake.phone_number()
            },
            credibility_score=0.7,
            created_at=datetime.now()
        )
    
    def _create_generic_persona(self, persona_id: str, persona_type: str, target_context: str) -> SocialPersona:
        """Create generic persona for other types."""
        profile = self.fake.profile()
        
        return SocialPersona(
            persona_id=persona_id,
            name=profile['name'],
            role=persona_type.replace('_', ' ').title(),
            organization=target_context or profile['company'],
            background_story=f"Professional with experience in {persona_type.replace('_', ' ')}.",
            personality_traits=['professional', 'knowledgeable', 'helpful'],
            communication_style='professional',
            credentials={
                'experience': f"{random.randint(3, 15)} years in {persona_type.replace('_', ' ')}"
            },
            social_media_profiles={
                'email': profile['mail']
            },
            credibility_score=0.6,
            created_at=datetime.now()
        )
    
    def _enhance_persona_with_llm(self, persona: SocialPersona) -> SocialPersona:
        """Enhance persona details using LLM."""
        if not self.llm_client:
            return persona
        
        enhancement_prompt = f"""
Enhance the following social engineering persona to make it more convincing and detailed:

Name: {persona.name}
Role: {persona.role}
Organization: {persona.organization}
Current Background: {persona.background_story}

Please provide:
1. Enhanced background story with specific details
2. Improved personality traits
3. Communication style refinements
4. Additional credentials or achievements
5. Potential conversation topics or interests

Format your response as JSON:
{{
    "enhanced_background": "detailed background story",
    "personality_traits": ["trait1", "trait2", "trait3"],
    "communication_style": "refined communication style",
    "additional_credentials": {{"credential_type": "credential_value"}},
    "conversation_topics": ["topic1", "topic2", "topic3"],
    "credibility_enhancements": ["enhancement1", "enhancement2"]
}}
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are an expert in creating convincing social engineering personas for security testing."},
                    {"role": "user", "content": enhancement_prompt}
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
                enhancement = json.loads(json_str)
                
                # Apply enhancements
                persona.background_story = enhancement.get('enhanced_background', persona.background_story)
                persona.personality_traits = enhancement.get('personality_traits', persona.personality_traits)
                persona.communication_style = enhancement.get('communication_style', persona.communication_style)
                
                # Add additional credentials
                additional_creds = enhancement.get('additional_credentials', {})
                persona.credentials.update(additional_creds)
                
                # Increase credibility score
                persona.credibility_score = min(persona.credibility_score + 0.1, 1.0)
                
        except Exception as e:
            print(f"⚠️ LLM persona enhancement failed: {str(e)}")
        
        return persona
    
    def design_social_campaign(self, target_profile: str, objective: str,
                             attack_types: List[str] = None) -> SocialEngineeringCampaign:
        """
        Design a social engineering campaign.
        
        Args:
            target_profile: Type of target (bank_employee, customer, etc.)
            objective: Campaign objective
            attack_types: List of attack types to use
            
        Returns:
            SocialEngineeringCampaign object
        """
        print(f"📋 Designing social engineering campaign for: {target_profile}")
        
        campaign_id = f"CAMPAIGN_{uuid.uuid4().hex[:8].upper()}"
        
        # Select attack types if not provided
        if not attack_types:
            attack_types = self._select_optimal_attack_types(target_profile)
        
        # Create personas for the campaign
        personas_needed = self._determine_personas_needed(target_profile, attack_types)
        personas_used = []
        
        for persona_type in personas_needed:
            persona = self.create_social_persona(persona_type, target_profile)
            personas_used.append(persona.persona_id)
        
        # Design attack vector
        attack_vector = self._design_attack_vector(target_profile, attack_types, objective)
        
        # Create campaign narrative
        narrative = self._create_campaign_narrative(target_profile, attack_types, objective)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(attack_types, objective)
        
        # Create timeline
        timeline = self._create_campaign_timeline(attack_types, objective)
        
        # Determine resources required
        resources_required = self._determine_resources_required(attack_types)
        
        # Assess risk level
        risk_level = self._assess_campaign_risk(attack_types, target_profile)
        
        campaign = SocialEngineeringCampaign(
            campaign_id=campaign_id,
            campaign_type='multi_vector',
            target_profile=target_profile,
            personas_used=personas_used,
            attack_vector=attack_vector,
            narrative=narrative,
            success_metrics=success_metrics,
            timeline=timeline,
            resources_required=resources_required,
            risk_level=risk_level,
            created_at=datetime.now()
        )
        
        self.campaigns.append(campaign)
        
        print(f"✅ Campaign designed: {len(attack_types)} attack vectors, {len(personas_used)} personas")
        return campaign
    
    def _select_optimal_attack_types(self, target_profile: str) -> List[str]:
        """Select optimal attack types for target profile."""
        target_data = self.target_profiles.get(target_profile, {})
        vulnerabilities = target_data.get('vulnerabilities', [])
        
        # Map vulnerabilities to attack types
        attack_mapping = {
            'authority_compliance': ['pretexting', 'vishing_call'],
            'urgency_pressure': ['phishing_email', 'vishing_call'],
            'helpful_nature': ['pretexting', 'baiting'],
            'trust_authority': ['pretexting', 'phishing_email'],
            'fear_loss': ['phishing_email', 'vishing_call'],
            'greed': ['baiting', 'phishing_email'],
            'business_relationship': ['pretexting', 'phishing_email'],
            'technical_complexity': ['pretexting', 'baiting']
        }
        
        selected_attacks = set()
        for vulnerability in vulnerabilities:
            selected_attacks.update(attack_mapping.get(vulnerability, []))
        
        return list(selected_attacks)[:3]  # Limit to 3 attack types
    
    def _determine_personas_needed(self, target_profile: str, attack_types: List[str]) -> List[str]:
        """Determine personas needed for the campaign."""
        personas_needed = set()
        
        # Map attack types to personas
        attack_persona_mapping = {
            'phishing_email': ['it_support', 'executive', 'vendor'],
            'vishing_call': ['it_support', 'auditor', 'executive'],
            'pretexting': ['auditor', 'vendor', 'executive'],
            'baiting': ['vendor', 'it_support'],
            'tailgating': ['vendor', 'it_support'],
            'watering_hole': ['it_support', 'vendor']
        }
        
        for attack_type in attack_types:
            attack_personas = attack_persona_mapping.get(attack_type, [])
            if attack_personas:
                personas_needed.add(random.choice(attack_personas))
        
        return list(personas_needed)
    
    def _design_attack_vector(self, target_profile: str, attack_types: List[str], objective: str) -> str:
        """Design the attack vector for the campaign."""
        target_data = self.target_profiles.get(target_profile, {})
        channels = target_data.get('communication_channels', ['email'])
        
        vector_components = []
        
        for attack_type in attack_types:
            if attack_type == 'phishing_email':
                vector_components.append("Email-based credential harvesting")
            elif attack_type == 'vishing_call':
                vector_components.append("Voice-based information extraction")
            elif attack_type == 'pretexting':
                vector_components.append("Scenario-based trust building")
            elif attack_type == 'baiting':
                vector_components.append("Incentive-based payload delivery")
            elif attack_type == 'tailgating':
                vector_components.append("Physical access through social manipulation")
        
        return f"Multi-channel attack using {', '.join(vector_components)} targeting {target_profile}"
    
    def _create_campaign_narrative(self, target_profile: str, attack_types: List[str], objective: str) -> str:
        """Create campaign narrative."""
        narrative_parts = [
            f"Target: {target_profile.replace('_', ' ').title()}",
            f"Objective: {objective}",
            f"Attack Methods: {', '.join(attack_types)}",
            f"Expected Outcome: Successful {objective.lower()} through social manipulation"
        ]
        
        return " | ".join(narrative_parts)
    
    def _define_success_metrics(self, attack_types: List[str], objective: str) -> Dict[str, Any]:
        """Define success metrics for the campaign."""
        metrics = {
            'primary_objective': objective,
            'success_threshold': 0.7,
            'metrics': []
        }
        
        for attack_type in attack_types:
            template = self.attack_templates.get(attack_type, {})
            success_indicators = template.get('success_indicators', [])
            metrics['metrics'].extend(success_indicators)
        
        # Remove duplicates
        metrics['metrics'] = list(set(metrics['metrics']))
        
        return metrics
    
    def _create_campaign_timeline(self, attack_types: List[str], objective: str) -> List[Dict[str, Any]]:
        """Create campaign timeline."""
        timeline = []
        current_date = datetime.now()
        
        # Phase 1: Reconnaissance
        timeline.append({
            'phase': 'reconnaissance',
            'start_date': current_date,
            'end_date': current_date + timedelta(days=2),
            'activities': ['Target research', 'Persona development', 'Attack vector preparation'],
            'deliverables': ['Target profiles', 'Persona documents', 'Attack templates']
        })
        
        # Phase 2: Initial Contact
        timeline.append({
            'phase': 'initial_contact',
            'start_date': current_date + timedelta(days=3),
            'end_date': current_date + timedelta(days=5),
            'activities': ['First contact attempts', 'Trust building', 'Relationship establishment'],
            'deliverables': ['Contact logs', 'Response analysis', 'Relationship status']
        })
        
        # Phase 3: Exploitation
        timeline.append({
            'phase': 'exploitation',
            'start_date': current_date + timedelta(days=6),
            'end_date': current_date + timedelta(days=10),
            'activities': ['Attack execution', 'Payload delivery', 'Objective achievement'],
            'deliverables': ['Attack results', 'Data collected', 'Access gained']
        })
        
        # Phase 4: Cleanup
        timeline.append({
            'phase': 'cleanup',
            'start_date': current_date + timedelta(days=11),
            'end_date': current_date + timedelta(days=14),
            'activities': ['Evidence cleanup', 'Trail removal', 'Documentation'],
            'deliverables': ['Clean exit', 'Final report', 'Lessons learned']
        })
        
        return timeline
    
    def _determine_resources_required(self, attack_types: List[str]) -> List[str]:
        """Determine resources required for the campaign."""
        resources = set()
        
        resource_mapping = {
            'phishing_email': ['Email infrastructure', 'Domain registration', 'SSL certificates'],
            'vishing_call': ['Phone system', 'Caller ID spoofing', 'Voice recording'],
            'pretexting': ['Research tools', 'Communication channels', 'Documentation'],
            'baiting': ['Payload creation', 'Delivery mechanism', 'Monitoring tools'],
            'tailgating': ['Physical access', 'Disguise materials', 'Timing coordination'],
            'watering_hole': ['Web hosting', 'Exploit kits', 'Traffic analysis']
        }
        
        for attack_type in attack_types:
            resources.update(resource_mapping.get(attack_type, []))
        
        return list(resources)
    
    def _assess_campaign_risk(self, attack_types: List[str], target_profile: str) -> str:
        """Assess campaign risk level."""
        risk_scores = []
        
        for attack_type in attack_types:
            template = self.attack_templates.get(attack_type, {})
            difficulty = template.get('difficulty', 'medium')
            
            difficulty_scores = {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'very_high': 1.0
            }
            
            risk_scores.append(difficulty_scores.get(difficulty, 0.5))
        
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        
        if avg_risk > 0.8:
            return 'high'
        elif avg_risk > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def execute_social_attack(self, campaign_id: str, attack_type: str, 
                            target_entity: str) -> SocialEngineeringAttack:
        """
        Execute a social engineering attack.
        
        Args:
            campaign_id: ID of the campaign
            attack_type: Type of attack to execute
            target_entity: Target entity ID
            
        Returns:
            SocialEngineeringAttack object
        """
        print(f"🎯 Executing social engineering attack: {attack_type}")
        
        # Find campaign
        campaign = None
        for c in self.campaigns:
            if c.campaign_id == campaign_id:
                campaign = c
                break
        
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        # Select persona for attack
        persona_id = random.choice(campaign.personas_used)
        persona = self.personas[persona_id]
        
        # Create attack payload
        payload = self._create_attack_payload(attack_type, persona, target_entity)
        
        # Calculate success probability
        success_probability = self._calculate_success_probability(
            attack_type, persona, target_entity, campaign.target_profile
        )
        
        # Execute attack
        attack = SocialEngineeringAttack(
            attack_id=f"ATTACK_{uuid.uuid4().hex[:8].upper()}",
            campaign_id=campaign_id,
            attack_type=attack_type,
            target_entity=target_entity,
            persona_used=persona_id,
            attack_method=self._determine_attack_method(attack_type),
            payload=payload,
            success_probability=success_probability,
            executed_at=datetime.now()
        )
        
        # Simulate attack execution
        outcome = self._simulate_attack_execution(attack)
        attack.outcome = outcome
        
        # Generate evidence and follow-up actions
        attack.evidence_collected = self._generate_evidence(attack)
        attack.follow_up_actions = self._generate_follow_up_actions(attack)
        
        self.attacks.append(attack)
        
        print(f"✅ Attack executed: {outcome} (probability: {success_probability:.2f})")
        return attack
    
    def _create_attack_payload(self, attack_type: str, persona: SocialPersona, 
                             target_entity: str) -> Dict[str, Any]:
        """Create attack payload."""
        if attack_type == 'phishing_email':
            return {
                'sender': f"{persona.name} <{persona.name.lower().replace(' ', '.')}@{persona.organization.lower().replace(' ', '')}.com>",
                'subject': f"Urgent: {random.choice(['Security Update Required', 'Account Verification Needed', 'System Maintenance Alert'])}",
                'body': f"Dear {target_entity},\n\nThis is {persona.name} from {persona.organization}. We need your immediate attention regarding your account security...",
                'malicious_link': f"https://secure-{persona.organization.lower().replace(' ', '')}.com/verify",
                'urgency_level': 'high'
            }
        
        elif attack_type == 'vishing_call':
            return {
                'caller_id': f"{persona.organization} - {persona.name}",
                'opening_script': f"Hello, this is {persona.name} from {persona.organization}. I'm calling about an urgent security matter...",
                'information_request': ['account_number', 'verification_code', 'password'],
                'authority_indicators': [persona.role, persona.organization],
                'urgency_factors': ['security_breach', 'account_compromise', 'immediate_action_required']
            }
        
        elif attack_type == 'pretexting':
            return {
                'scenario': f"Routine {random.choice(['audit', 'compliance review', 'security assessment'])}",
                'authority_figure': f"{persona.name}, {persona.role}",
                'legitimacy_proof': [persona.credentials, persona.organization],
                'information_request': ['system_access', 'user_credentials', 'process_documentation'],
                'trust_building_elements': ['professional_knowledge', 'industry_references', 'mutual_contacts']
            }
        
        else:
            return {
                'attack_type': attack_type,
                'persona': persona.name,
                'target': target_entity,
                'method': 'social_manipulation'
            }
    
    def _calculate_success_probability(self, attack_type: str, persona: SocialPersona,
                                     target_entity: str, target_profile: str) -> float:
        """Calculate attack success probability."""
        base_probability = 0.5
        
        # Persona credibility factor
        credibility_factor = persona.credibility_score * 0.3
        
        # Attack type effectiveness
        template = self.attack_templates.get(attack_type, {})
        difficulty = template.get('difficulty', 'medium')
        
        difficulty_modifiers = {
            'low': 0.2,
            'medium': 0.0,
            'high': -0.2,
            'very_high': -0.4
        }
        
        difficulty_modifier = difficulty_modifiers.get(difficulty, 0.0)
        
        # Target profile vulnerability
        target_data = self.target_profiles.get(target_profile, {})
        vulnerabilities = target_data.get('vulnerabilities', [])
        vulnerability_factor = len(vulnerabilities) * 0.05
        
        # Calculate final probability
        final_probability = base_probability + credibility_factor + difficulty_modifier + vulnerability_factor
        
        return max(0.1, min(0.9, final_probability))
    
    def _determine_attack_method(self, attack_type: str) -> str:
        """Determine attack method."""
        method_mapping = {
            'phishing_email': 'email_deception',
            'vishing_call': 'voice_manipulation',
            'pretexting': 'scenario_fabrication',
            'baiting': 'incentive_manipulation',
            'tailgating': 'physical_manipulation',
            'watering_hole': 'website_compromise'
        }
        
        return method_mapping.get(attack_type, 'social_manipulation')
    
    def _simulate_attack_execution(self, attack: SocialEngineeringAttack) -> str:
        """Simulate attack execution."""
        # Use success probability to determine outcome
        if random.random() < attack.success_probability:
            outcomes = ['successful', 'partially_successful']
            return random.choice(outcomes)
        else:
            outcomes = ['failed', 'detected', 'blocked']
            return random.choice(outcomes)
    
    def _generate_evidence(self, attack: SocialEngineeringAttack) -> List[str]:
        """Generate evidence collected from attack."""
        evidence = []
        
        if attack.outcome in ['successful', 'partially_successful']:
            evidence.extend([
                f"Target response: {random.choice(['compliant', 'suspicious', 'helpful'])}",
                f"Information disclosed: {random.choice(['partial', 'complete', 'minimal'])}",
                f"Trust level achieved: {random.choice(['high', 'medium', 'low'])}",
                f"Follow-up potential: {random.choice(['high', 'medium', 'low'])}"
            ])
        
        if attack.outcome == 'failed':
            evidence.extend([
                f"Failure reason: {random.choice(['suspicious_target', 'poor_execution', 'bad_timing'])}",
                f"Detection level: {random.choice(['none', 'minimal', 'high'])}",
                f"Countermeasures: {random.choice(['none', 'basic', 'advanced'])}"
            ])
        
        return evidence
    
    def _generate_follow_up_actions(self, attack: SocialEngineeringAttack) -> List[str]:
        """Generate follow-up actions."""
        actions = []
        
        if attack.outcome == 'successful':
            actions.extend([
                "Exploit gained access immediately",
                "Establish persistence mechanisms",
                "Collect additional intelligence",
                "Prepare for next phase"
            ])
        elif attack.outcome == 'partially_successful':
            actions.extend([
                "Analyze partial success factors",
                "Refine approach for next attempt",
                "Build on established trust",
                "Gather more target intelligence"
            ])
        else:
            actions.extend([
                "Analyze failure factors",
                "Adjust persona or approach",
                "Wait for detection to subside",
                "Consider alternative attack vectors"
            ])
        
        return actions
    
    def get_social_engineering_dashboard(self) -> Dict[str, Any]:
        """Get social engineering dashboard data."""
        current_time = datetime.now()
        
        # Campaign statistics
        active_campaigns = len([c for c in self.campaigns if c.status == 'active'])
        completed_campaigns = len([c for c in self.campaigns if c.status == 'completed'])
        
        # Attack statistics
        recent_attacks = [a for a in self.attacks 
                         if (current_time - a.executed_at).days < 7]
        
        successful_attacks = len([a for a in recent_attacks if a.outcome == 'successful'])
        
        # Persona statistics
        active_personas = len(self.personas)
        
        return {
            'timestamp': current_time.isoformat(),
            'campaigns': {
                'active': active_campaigns,
                'completed': completed_campaigns,
                'total': len(self.campaigns)
            },
            'attacks': {
                'recent': len(recent_attacks),
                'successful': successful_attacks,
                'success_rate': successful_attacks / len(recent_attacks) if recent_attacks else 0.0,
                'total': len(self.attacks)
            },
            'personas': {
                'active': active_personas,
                'types': self._get_persona_type_distribution()
            },
            'attack_types': self._get_attack_type_distribution(),
            'target_profiles': self._get_target_profile_distribution()
        }
    
    def _get_persona_type_distribution(self) -> Dict[str, int]:
        """Get persona type distribution."""
        distribution = {}
        for persona in self.personas.values():
            persona_type = persona.role.lower().replace(' ', '_')
            distribution[persona_type] = distribution.get(persona_type, 0) + 1
        return distribution
    
    def _get_attack_type_distribution(self) -> Dict[str, int]:
        """Get attack type distribution."""
        distribution = {}
        for attack in self.attacks:
            attack_type = attack.attack_type
            distribution[attack_type] = distribution.get(attack_type, 0) + 1
        return distribution
    
    def _get_target_profile_distribution(self) -> Dict[str, int]:
        """Get target profile distribution."""
        distribution = {}
        for campaign in self.campaigns:
            target_profile = campaign.target_profile
            distribution[target_profile] = distribution.get(target_profile, 0) + 1
        return distribution
    
    def export_social_engineering_data(self, output_dir: str = "social_engineering/"):
        """Export social engineering data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export personas
        personas_data = []
        for persona in self.personas.values():
            personas_data.append({
                'persona_id': persona.persona_id,
                'name': persona.name,
                'role': persona.role,
                'organization': persona.organization,
                'background_story': persona.background_story,
                'personality_traits': persona.personality_traits,
                'communication_style': persona.communication_style,
                'credentials': persona.credentials,
                'social_media_profiles': persona.social_media_profiles,
                'credibility_score': persona.credibility_score,
                'created_at': persona.created_at.isoformat(),
                'last_used': persona.last_used.isoformat() if persona.last_used else None
            })
        
        with open(output_path / f"social_personas_{timestamp}.json", 'w') as f:
            json.dump(personas_data, f, indent=2)
        
        # Export campaigns
        campaigns_data = []
        for campaign in self.campaigns:
            campaigns_data.append({
                'campaign_id': campaign.campaign_id,
                'campaign_type': campaign.campaign_type,
                'target_profile': campaign.target_profile,
                'personas_used': campaign.personas_used,
                'attack_vector': campaign.attack_vector,
                'narrative': campaign.narrative,
                'success_metrics': campaign.success_metrics,
                'timeline': [
                    {
                        **phase,
                        'start_date': phase['start_date'].isoformat(),
                        'end_date': phase['end_date'].isoformat()
                    }
                    for phase in campaign.timeline
                ],
                'resources_required': campaign.resources_required,
                'risk_level': campaign.risk_level,
                'created_at': campaign.created_at.isoformat(),
                'status': campaign.status
            })
        
        with open(output_path / f"social_campaigns_{timestamp}.json", 'w') as f:
            json.dump(campaigns_data, f, indent=2)
        
        # Export attacks
        attacks_data = []
        for attack in self.attacks:
            attacks_data.append({
                'attack_id': attack.attack_id,
                'campaign_id': attack.campaign_id,
                'attack_type': attack.attack_type,
                'target_entity': attack.target_entity,
                'persona_used': attack.persona_used,
                'attack_method': attack.attack_method,
                'payload': attack.payload,
                'success_probability': attack.success_probability,
                'executed_at': attack.executed_at.isoformat(),
                'outcome': attack.outcome,
                'evidence_collected': attack.evidence_collected,
                'follow_up_actions': attack.follow_up_actions
            })
        
        with open(output_path / f"social_attacks_{timestamp}.json", 'w') as f:
            json.dump(attacks_data, f, indent=2)
        
        print(f"📁 Social engineering data exported to {output_path}")
    
    def get_social_engineering_summary(self) -> str:
        """Generate human-readable social engineering summary."""
        total_personas = len(self.personas)
        total_campaigns = len(self.campaigns)
        total_attacks = len(self.attacks)
        
        successful_attacks = len([a for a in self.attacks if a.outcome == 'successful'])
        success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0
        
        summary = f"""
SOCIAL ENGINEERING SUMMARY
==========================

Personas Created: {total_personas}
Campaigns Designed: {total_campaigns}
Attacks Executed: {total_attacks}
Success Rate: {success_rate:.1f}%

Persona Types:
"""
        
        # Persona type breakdown
        persona_types = {}
        for persona in self.personas.values():
            role = persona.role
            persona_types[role] = persona_types.get(role, 0) + 1
        
        for role, count in sorted(persona_types.items(), key=lambda x: x[1], reverse=True):
            summary += f"  {role}: {count}\n"
        
        # Attack type breakdown
        if self.attacks:
            summary += f"\nAttack Types:\n"
            attack_types = {}
            for attack in self.attacks:
                attack_types[attack.attack_type] = attack_types.get(attack.attack_type, 0) + 1
            
            for attack_type, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
                summary += f"  {attack_type.replace('_', ' ').title()}: {count}\n"
        
        summary += f"\nOverall Effectiveness: {'🎯 HIGH' if success_rate > 60 else '⚡ MEDIUM' if success_rate > 30 else '🔄 DEVELOPING'}"
        
        return summary


def main():
    """Main function for testing the Social Engineering Agent."""
    print("Testing Red Team Social Engineering Agent...")
    
    # Initialize agent
    social_agent = SocialEngineeringAgent()
    
    # Test persona creation
    print("\nCreating social engineering personas...")
    personas = []
    persona_types = ['executive', 'it_support', 'auditor', 'vendor', 'customer']
    
    for persona_type in persona_types:
        persona = social_agent.create_social_persona(persona_type, 'Financial Services Corp')
        personas.append(persona)
        print(f"  Created: {persona.name} ({persona.role})")
    
    # Test campaign design
    print("\nDesigning social engineering campaign...")
    campaign = social_agent.design_social_campaign(
        target_profile='bank_employee',
        objective='credential_harvesting',
        attack_types=['phishing_email', 'vishing_call', 'pretexting']
    )
    
    # Test attack execution
    print("\nExecuting social engineering attacks...")
    for attack_type in ['phishing_email', 'vishing_call']:
        attack = social_agent.execute_social_attack(
            campaign.campaign_id,
            attack_type,
            'TARGET_EMPLOYEE_001'
        )
        print(f"  {attack_type}: {attack.outcome}")
    
    # Display results
    print("\nSocial Engineering Summary:")
    print("=" * 50)
    print(social_agent.get_social_engineering_summary())
    
    # Export data
    social_agent.export_social_engineering_data()
    
    print("\n✅ Social Engineering Agent test completed!")


if __name__ == "__main__":
    main() 