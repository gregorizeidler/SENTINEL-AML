"""
Shell Company Creator Agent - Especialista em criação de empresas de fachada

Este agente simula a criação de empresas fictícias ou de fachada para facilitar
operações de lavagem de dinheiro através de estruturas corporativas complexas.
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ShellCompanyCreator:
    """
    Agente especializado na criação de empresas de fachada e estruturas corporativas
    complexas para operações de lavagem de dinheiro.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.created_companies = []
        self.corporate_structures = []
        
        # Tipos de empresas de fachada
        self.company_types = {
            'consulting': {
                'legitimacy_score': 0.8,
                'setup_complexity': 'low',
                'transaction_volume': 'high',
                'regulatory_scrutiny': 'low',
                'typical_activities': ['Business consulting', 'Management advisory', 'Strategy consulting']
            },
            'trading': {
                'legitimacy_score': 0.6,
                'setup_complexity': 'medium',
                'transaction_volume': 'very_high',
                'regulatory_scrutiny': 'medium',
                'typical_activities': ['Import/export', 'Commodity trading', 'International trade']
            },
            'real_estate': {
                'legitimacy_score': 0.7,
                'setup_complexity': 'high',
                'transaction_volume': 'high',
                'regulatory_scrutiny': 'high',
                'typical_activities': ['Property development', 'Real estate investment', 'Property management']
            },
            'technology': {
                'legitimacy_score': 0.9,
                'setup_complexity': 'medium',
                'transaction_volume': 'medium',
                'regulatory_scrutiny': 'low',
                'typical_activities': ['Software development', 'IT services', 'Digital solutions']
            },
            'holding': {
                'legitimacy_score': 0.5,
                'setup_complexity': 'high',
                'transaction_volume': 'very_high',
                'regulatory_scrutiny': 'high',
                'typical_activities': ['Investment holding', 'Asset management', 'Portfolio management']
            },
            'logistics': {
                'legitimacy_score': 0.7,
                'setup_complexity': 'medium',
                'transaction_volume': 'high',
                'regulatory_scrutiny': 'medium',
                'typical_activities': ['Transportation', 'Warehousing', 'Supply chain management']
            }
        }
        
        # Jurisdições para incorporação
        self.jurisdictions = {
            'delaware_us': {
                'privacy_level': 0.6,
                'setup_speed': 'fast',
                'cost': 'low',
                'regulatory_burden': 'low',
                'international_treaties': 'high'
            },
            'british_virgin_islands': {
                'privacy_level': 0.9,
                'setup_speed': 'fast',
                'cost': 'medium',
                'regulatory_burden': 'very_low',
                'international_treaties': 'medium'
            },
            'cayman_islands': {
                'privacy_level': 0.8,
                'setup_speed': 'medium',
                'cost': 'high',
                'regulatory_burden': 'low',
                'international_treaties': 'high'
            },
            'panama': {
                'privacy_level': 0.9,
                'setup_speed': 'fast',
                'cost': 'low',
                'regulatory_burden': 'very_low',
                'international_treaties': 'low'
            },
            'seychelles': {
                'privacy_level': 0.8,
                'setup_speed': 'fast',
                'cost': 'low',
                'regulatory_burden': 'very_low',
                'international_treaties': 'low'
            },
            'singapore': {
                'privacy_level': 0.4,
                'setup_speed': 'medium',
                'cost': 'high',
                'regulatory_burden': 'high',
                'international_treaties': 'very_high'
            },
            'hong_kong': {
                'privacy_level': 0.5,
                'setup_speed': 'medium',
                'cost': 'medium',
                'regulatory_burden': 'medium',
                'international_treaties': 'high'
            }
        }
        
        # Estruturas corporativas
        self.corporate_structures = {
            'simple_shell': {
                'complexity': 'low',
                'layers': 1,
                'detection_difficulty': 0.3,
                'setup_time': 'fast'
            },
            'layered_structure': {
                'complexity': 'medium',
                'layers': 3,
                'detection_difficulty': 0.6,
                'setup_time': 'medium'
            },
            'complex_web': {
                'complexity': 'high',
                'layers': 5,
                'detection_difficulty': 0.8,
                'setup_time': 'slow'
            },
            'circular_ownership': {
                'complexity': 'very_high',
                'layers': 4,
                'detection_difficulty': 0.9,
                'setup_time': 'slow'
            }
        }
    
    def design_corporate_structure(self, target_amount: float, risk_tolerance: str = 'medium') -> Dict[str, Any]:
        """
        Projeta uma estrutura corporativa para lavagem de dinheiro
        
        Args:
            target_amount: Valor alvo para lavagem
            risk_tolerance: Tolerância ao risco (low, medium, high)
            
        Returns:
            Dict com detalhes da estrutura corporativa
        """
        try:
            # Determinar complexidade baseada no valor e risco
            structure_type = self._select_structure_type(target_amount, risk_tolerance)
            
            # Selecionar jurisdições
            jurisdictions = self._select_jurisdictions(structure_type, risk_tolerance)
            
            # Criar empresas para a estrutura
            companies = self._design_company_hierarchy(structure_type, jurisdictions)
            
            # Definir fluxos de fundos
            fund_flows = self._design_fund_flows(companies)
            
            # Criar documentação de suporte
            supporting_docs = self._create_supporting_documentation(companies)
            
            structure = {
                'structure_id': f"STRUCT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'structure_type': structure_type,
                'target_amount': target_amount,
                'risk_tolerance': risk_tolerance,
                'jurisdictions': jurisdictions,
                'companies': companies,
                'fund_flows': fund_flows,
                'supporting_documentation': supporting_docs,
                'complexity_score': self._calculate_complexity_score(structure_type, len(companies)),
                'detection_difficulty': self._calculate_detection_difficulty(structure_type, jurisdictions),
                'estimated_setup_time': self._estimate_setup_time(structure_type, len(companies)),
                'estimated_cost': self._estimate_setup_cost(companies, jurisdictions),
                'created_at': datetime.now().isoformat()
            }
            
            self.corporate_structures.append(structure)
            logger.info(f"Estrutura corporativa criada: {structure['structure_id']}")
            
            return structure
            
        except Exception as e:
            logger.error(f"Erro ao criar estrutura corporativa: {str(e)}")
            return {}
    
    def create_shell_company(self, company_type: str, jurisdiction: str, purpose: str = 'general') -> Dict[str, Any]:
        """
        Cria uma empresa de fachada individual
        
        Args:
            company_type: Tipo da empresa
            jurisdiction: Jurisdição de incorporação
            purpose: Propósito específico da empresa
            
        Returns:
            Dict com detalhes da empresa criada
        """
        try:
            # Gerar nome da empresa
            company_name = self._generate_company_name(company_type)
            
            # Criar estrutura de propriedade
            ownership_structure = self._create_ownership_structure()
            
            # Definir diretores e oficiais
            directors_officers = self._create_directors_officers(jurisdiction)
            
            # Criar endereços e contatos
            addresses = self._create_company_addresses(jurisdiction)
            
            # Definir atividades comerciais
            business_activities = self._define_business_activities(company_type)
            
            company = {
                'company_id': f"SHELL_{len(self.created_companies) + 1:04d}",
                'company_name': company_name,
                'company_type': company_type,
                'jurisdiction': jurisdiction,
                'purpose': purpose,
                'incorporation_date': datetime.now().isoformat(),
                'ownership_structure': ownership_structure,
                'directors_officers': directors_officers,
                'addresses': addresses,
                'business_activities': business_activities,
                'financial_details': self._create_financial_profile(company_type),
                'compliance_status': self._create_compliance_profile(jurisdiction),
                'legitimacy_indicators': self._create_legitimacy_indicators(company_type),
                'risk_factors': self._assess_company_risk_factors(company_type, jurisdiction),
                'created_at': datetime.now().isoformat()
            }
            
            self.created_companies.append(company)
            logger.info(f"Empresa de fachada criada: {company['company_name']}")
            
            return company
            
        except Exception as e:
            logger.error(f"Erro ao criar empresa de fachada: {str(e)}")
            return {}
    
    def create_nominee_services(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cria serviços de nominee para ocultar propriedade real
        
        Args:
            company: Empresa para aplicar serviços nominee
            
        Returns:
            Dict com detalhes dos serviços nominee
        """
        try:
            # Criar nominee directors
            nominee_directors = self._create_nominee_directors()
            
            # Criar nominee shareholders
            nominee_shareholders = self._create_nominee_shareholders()
            
            # Criar trust arrangements
            trust_arrangements = self._create_trust_arrangements()
            
            # Criar power of attorney
            power_of_attorney = self._create_power_of_attorney()
            
            nominee_services = {
                'company_id': company['company_id'],
                'nominee_directors': nominee_directors,
                'nominee_shareholders': nominee_shareholders,
                'trust_arrangements': trust_arrangements,
                'power_of_attorney': power_of_attorney,
                'service_provider': self._select_service_provider(company['jurisdiction']),
                'confidentiality_level': self._assess_confidentiality_level(company['jurisdiction']),
                'annual_cost': self._calculate_nominee_cost(company['jurisdiction']),
                'legal_protections': self._assess_legal_protections(company['jurisdiction']),
                'created_at': datetime.now().isoformat()
            }
            
            return nominee_services
            
        except Exception as e:
            logger.error(f"Erro ao criar serviços nominee: {str(e)}")
            return {}
    
    def _select_structure_type(self, target_amount: float, risk_tolerance: str) -> str:
        """Seleciona tipo de estrutura baseado no valor e risco"""
        if target_amount < 100000:
            return 'simple_shell'
        elif target_amount < 1000000:
            if risk_tolerance == 'low':
                return 'layered_structure'
            else:
                return 'simple_shell'
        elif target_amount < 10000000:
            if risk_tolerance == 'low':
                return 'complex_web'
            else:
                return 'layered_structure'
        else:
            return 'circular_ownership'
    
    def _select_jurisdictions(self, structure_type: str, risk_tolerance: str) -> List[str]:
        """Seleciona jurisdições apropriadas"""
        structure_info = self.corporate_structures[structure_type]
        num_jurisdictions = min(structure_info['layers'], 3)
        
        if risk_tolerance == 'low':
            # Preferir jurisdições com mais privacidade
            preferred = ['british_virgin_islands', 'panama', 'seychelles']
        elif risk_tolerance == 'high':
            # Preferir jurisdições mais respeitáveis
            preferred = ['delaware_us', 'singapore', 'hong_kong']
        else:
            preferred = list(self.jurisdictions.keys())
        
        return random.sample(preferred, min(num_jurisdictions, len(preferred)))
    
    def _design_company_hierarchy(self, structure_type: str, jurisdictions: List[str]) -> List[Dict[str, Any]]:
        """Projeta hierarquia de empresas"""
        structure_info = self.corporate_structures[structure_type]
        companies = []
        
        for i in range(structure_info['layers']):
            jurisdiction = jurisdictions[i % len(jurisdictions)]
            company_type = random.choice(list(self.company_types.keys()))
            
            company = self.create_shell_company(company_type, jurisdiction, f'layer_{i+1}')
            companies.append(company)
        
        return companies
    
    def _design_fund_flows(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Projeta fluxos de fundos entre empresas"""
        fund_flows = []
        
        for i in range(len(companies) - 1):
            source = companies[i]
            destination = companies[i + 1]
            
            flow = {
                'source_company': source['company_id'],
                'destination_company': destination['company_id'],
                'flow_type': random.choice(['loan', 'investment', 'service_fee', 'dividend']),
                'amount_range': self._calculate_flow_amount(source, destination),
                'frequency': random.choice(['one_time', 'monthly', 'quarterly', 'annual']),
                'justification': self._create_flow_justification(source, destination)
            }
            
            fund_flows.append(flow)
        
        return fund_flows
    
    def _create_supporting_documentation(self, companies: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Cria documentação de suporte"""
        return {
            'incorporation_documents': [f"Certificate of Incorporation - {company['company_name']}" for company in companies],
            'board_resolutions': [f"Board Resolution - {company['company_name']}" for company in companies],
            'shareholder_agreements': [f"Shareholder Agreement - {company['company_name']}" for company in companies],
            'service_agreements': [f"Service Agreement between {companies[i]['company_name']} and {companies[i+1]['company_name']}" for i in range(len(companies)-1)],
            'financial_statements': [f"Financial Statements - {company['company_name']}" for company in companies],
            'audit_reports': [f"Audit Report - {company['company_name']}" for company in companies if random.random() > 0.5]
        }
    
    def _generate_company_name(self, company_type: str) -> str:
        """Gera nome realista para empresa"""
        prefixes = ['Global', 'International', 'Premier', 'Strategic', 'Advanced', 'Elite', 'Professional']
        suffixes = ['Holdings', 'Ventures', 'Solutions', 'Partners', 'Group', 'Corp', 'Ltd']
        
        type_words = {
            'consulting': ['Consulting', 'Advisory', 'Management', 'Strategy'],
            'trading': ['Trading', 'Commerce', 'Import', 'Export'],
            'real_estate': ['Properties', 'Realty', 'Development', 'Estates'],
            'technology': ['Tech', 'Systems', 'Digital', 'Innovation'],
            'holding': ['Holdings', 'Investments', 'Capital', 'Assets'],
            'logistics': ['Logistics', 'Transport', 'Supply', 'Distribution']
        }
        
        prefix = random.choice(prefixes)
        type_word = random.choice(type_words.get(company_type, ['Business']))
        suffix = random.choice(suffixes)
        
        return f"{prefix} {type_word} {suffix}"
    
    def _create_ownership_structure(self) -> Dict[str, Any]:
        """Cria estrutura de propriedade"""
        return {
            'authorized_shares': random.randint(1000, 100000),
            'issued_shares': random.randint(100, 10000),
            'share_classes': ['Common', 'Preferred'] if random.random() > 0.5 else ['Common'],
            'shareholder_structure': 'nominee_held' if random.random() > 0.3 else 'direct_ownership',
            'voting_rights': 'standard' if random.random() > 0.7 else 'modified'
        }
    
    def _create_directors_officers(self, jurisdiction: str) -> List[Dict[str, Any]]:
        """Cria diretores e oficiais"""
        directors = []
        
        # Número de diretores baseado na jurisdição
        min_directors = 1 if jurisdiction in ['british_virgin_islands', 'panama'] else 2
        num_directors = random.randint(min_directors, 3)
        
        for i in range(num_directors):
            director = {
                'name': f"Director {i+1}",
                'role': random.choice(['Director', 'Chairman', 'Secretary']),
                'nationality': self._get_jurisdiction_nationality(jurisdiction),
                'appointment_date': datetime.now().isoformat(),
                'is_nominee': random.random() > 0.4
            }
            directors.append(director)
        
        return directors
    
    def _create_company_addresses(self, jurisdiction: str) -> Dict[str, str]:
        """Cria endereços da empresa"""
        address_templates = {
            'delaware_us': '1209 Orange Street, Wilmington, DE 19801, USA',
            'british_virgin_islands': 'Craigmuir Chambers, Road Town, Tortola, BVI',
            'cayman_islands': 'Harbour Centre, North Church Street, George Town, Cayman Islands',
            'panama': 'Obarrio Business Center, Panama City, Panama',
            'seychelles': 'Global Gateway 8, Rue de la Perle, Providence, Seychelles',
            'singapore': '1 Raffles Place, Singapore 048616',
            'hong_kong': '16/F, Tower 1, Admiralty Centre, Hong Kong'
        }
        
        return {
            'registered_address': address_templates.get(jurisdiction, 'Generic Business Address'),
            'business_address': address_templates.get(jurisdiction, 'Generic Business Address'),
            'mailing_address': address_templates.get(jurisdiction, 'Generic Business Address')
        }
    
    def _define_business_activities(self, company_type: str) -> List[str]:
        """Define atividades comerciais"""
        return self.company_types[company_type]['typical_activities']
    
    def _create_financial_profile(self, company_type: str) -> Dict[str, Any]:
        """Cria perfil financeiro"""
        return {
            'initial_capital': random.randint(10000, 1000000),
            'expected_annual_revenue': random.randint(100000, 10000000),
            'transaction_volume': self.company_types[company_type]['transaction_volume'],
            'banking_relationships': random.randint(1, 3),
            'credit_rating': random.choice(['A', 'B', 'C', 'Not Rated'])
        }
    
    def _create_compliance_profile(self, jurisdiction: str) -> Dict[str, Any]:
        """Cria perfil de compliance"""
        return {
            'regulatory_requirements': self.jurisdictions[jurisdiction]['regulatory_burden'],
            'reporting_obligations': random.choice(['Annual', 'Quarterly', 'Monthly', 'None']),
            'audit_requirements': random.random() > 0.5,
            'tax_obligations': random.choice(['Standard', 'Reduced', 'Exempt']),
            'aml_requirements': random.choice(['Standard', 'Enhanced', 'Minimal'])
        }
    
    def _create_legitimacy_indicators(self, company_type: str) -> Dict[str, Any]:
        """Cria indicadores de legitimidade"""
        return {
            'website_presence': random.random() > 0.3,
            'business_licenses': random.random() > 0.4,
            'professional_memberships': random.random() > 0.6,
            'insurance_coverage': random.random() > 0.5,
            'employee_count': random.randint(0, 10),
            'legitimacy_score': self.company_types[company_type]['legitimacy_score']
        }
    
    def _assess_company_risk_factors(self, company_type: str, jurisdiction: str) -> Dict[str, float]:
        """Avalia fatores de risco da empresa"""
        return {
            'regulatory_risk': 1.0 - self.jurisdictions[jurisdiction]['privacy_level'],
            'operational_risk': random.uniform(0.2, 0.8),
            'reputational_risk': 1.0 - self.company_types[company_type]['legitimacy_score'],
            'detection_risk': random.uniform(0.1, 0.9),
            'overall_risk': random.uniform(0.3, 0.7)
        }
    
    def _calculate_complexity_score(self, structure_type: str, num_companies: int) -> float:
        """Calcula score de complexidade"""
        base_score = {'simple_shell': 0.2, 'layered_structure': 0.5, 'complex_web': 0.8, 'circular_ownership': 0.9}
        return base_score[structure_type] + (num_companies * 0.1)
    
    def _calculate_detection_difficulty(self, structure_type: str, jurisdictions: List[str]) -> float:
        """Calcula dificuldade de detecção"""
        base_difficulty = self.corporate_structures[structure_type]['detection_difficulty']
        jurisdiction_bonus = sum(self.jurisdictions[j]['privacy_level'] for j in jurisdictions) / len(jurisdictions)
        return min(0.95, base_difficulty + jurisdiction_bonus * 0.2)
    
    def _estimate_setup_time(self, structure_type: str, num_companies: int) -> str:
        """Estima tempo de configuração"""
        base_days = {'simple_shell': 7, 'layered_structure': 21, 'complex_web': 45, 'circular_ownership': 60}
        total_days = base_days[structure_type] + (num_companies * 5)
        return f"{total_days} days"
    
    def _estimate_setup_cost(self, companies: List[Dict[str, Any]], jurisdictions: List[str]) -> Dict[str, float]:
        """Estima custos de configuração"""
        jurisdiction_costs = {
            'delaware_us': 500, 'british_virgin_islands': 2000, 'cayman_islands': 5000,
            'panama': 1500, 'seychelles': 1000, 'singapore': 3000, 'hong_kong': 2500
        }
        
        total_cost = sum(jurisdiction_costs.get(j, 1000) for j in jurisdictions)
        annual_cost = total_cost * 0.3
        
        return {
            'setup_cost': total_cost,
            'annual_maintenance': annual_cost,
            'total_first_year': total_cost + annual_cost
        }
    
    def _create_nominee_directors(self) -> List[Dict[str, Any]]:
        """Cria diretores nominee"""
        return [
            {
                'name': f"Nominee Director {i+1}",
                'service_provider': 'Professional Services Ltd',
                'nationality': random.choice(['Seychelles', 'BVI', 'Panama']),
                'fee': random.randint(1000, 5000),
                'confidentiality_agreement': True
            }
            for i in range(random.randint(1, 2))
        ]
    
    def _create_nominee_shareholders(self) -> List[Dict[str, Any]]:
        """Cria acionistas nominee"""
        return [
            {
                'name': f"Nominee Shareholder {i+1}",
                'share_percentage': random.randint(10, 100),
                'service_provider': 'Trust Services Corp',
                'fee': random.randint(2000, 8000),
                'declaration_of_trust': True
            }
            for i in range(random.randint(1, 2))
        ]
    
    def _create_trust_arrangements(self) -> Dict[str, Any]:
        """Cria arranjos de trust"""
        return {
            'trust_type': random.choice(['Discretionary', 'Fixed', 'Charitable']),
            'trustee': 'Professional Trustee Services',
            'beneficiaries': 'Confidential',
            'trust_deed': True,
            'annual_fee': random.randint(5000, 20000)
        }
    
    def _create_power_of_attorney(self) -> Dict[str, Any]:
        """Cria procuração"""
        return {
            'attorney_name': 'Legal Representative',
            'powers_granted': ['Banking', 'Contracting', 'Property'],
            'duration': 'Indefinite',
            'revocable': True,
            'fee': random.randint(1000, 3000)
        }
    
    def _select_service_provider(self, jurisdiction: str) -> str:
        """Seleciona provedor de serviços"""
        providers = {
            'british_virgin_islands': 'BVI Corporate Services',
            'panama': 'Panama Legal Services',
            'seychelles': 'Seychelles Business Services',
            'cayman_islands': 'Cayman Corporate Solutions'
        }
        return providers.get(jurisdiction, 'International Corporate Services')
    
    def _assess_confidentiality_level(self, jurisdiction: str) -> float:
        """Avalia nível de confidencialidade"""
        return self.jurisdictions[jurisdiction]['privacy_level']
    
    def _calculate_nominee_cost(self, jurisdiction: str) -> float:
        """Calcula custo dos serviços nominee"""
        base_costs = {
            'british_virgin_islands': 8000, 'panama': 6000, 'seychelles': 5000,
            'cayman_islands': 12000, 'delaware_us': 3000
        }
        return base_costs.get(jurisdiction, 7000)
    
    def _assess_legal_protections(self, jurisdiction: str) -> List[str]:
        """Avalia proteções legais"""
        protections = {
            'british_virgin_islands': ['Confidentiality laws', 'No public registry', 'Bearer shares allowed'],
            'panama': ['Banking secrecy', 'Anonymous companies', 'No exchange controls'],
            'seychelles': ['Confidentiality protection', 'No public disclosure', 'Flexible structures']
        }
        return protections.get(jurisdiction, ['Standard corporate protections'])
    
    def _calculate_flow_amount(self, source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, float]:
        """Calcula valores de fluxo de fundos"""
        return {
            'minimum': random.randint(10000, 100000),
            'maximum': random.randint(100000, 1000000),
            'typical': random.randint(50000, 500000)
        }
    
    def _create_flow_justification(self, source: Dict[str, Any], destination: Dict[str, Any]) -> str:
        """Cria justificativa para fluxo de fundos"""
        justifications = [
            f"Management fees from {source['company_name']} to {destination['company_name']}",
            f"Loan facility provided by {source['company_name']} to {destination['company_name']}",
            f"Investment return from {destination['company_name']} to {source['company_name']}",
            f"Service fees for consulting services provided by {destination['company_name']}"
        ]
        return random.choice(justifications)
    
    def _get_jurisdiction_nationality(self, jurisdiction: str) -> str:
        """Retorna nacionalidade típica para jurisdição"""
        nationalities = {
            'british_virgin_islands': 'British',
            'panama': 'Panamanian',
            'seychelles': 'Seychellois',
            'cayman_islands': 'Caymanian',
            'delaware_us': 'American'
        }
        return nationalities.get(jurisdiction, 'International')
    
    def get_structure_analytics(self) -> Dict[str, Any]:
        """Retorna análise das estruturas corporativas criadas"""
        return {
            'total_structures': len(self.corporate_structures),
            'total_companies': len(self.created_companies),
            'jurisdictions_used': list(set(company['jurisdiction'] for company in self.created_companies)),
            'company_types_used': list(set(company['company_type'] for company in self.created_companies)),
            'average_complexity': sum(struct['complexity_score'] for struct in self.corporate_structures) / len(self.corporate_structures) if self.corporate_structures else 0,
            'average_detection_difficulty': sum(struct['detection_difficulty'] for struct in self.corporate_structures) / len(self.corporate_structures) if self.corporate_structures else 0,
            'total_estimated_cost': sum(struct['estimated_cost']['total_first_year'] for struct in self.corporate_structures)
        } 