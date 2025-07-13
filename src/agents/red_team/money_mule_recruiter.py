"""
Money Mule Recruiter Agent - Especialista em recrutamento de intermediários para operações de lavagem

Este agente simula táticas de recrutamento de "mulas financeiras" - pessoas que facilitam
transferências de dinheiro ilícito através de suas contas bancárias.
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MoneyMuleRecruiter:
    """
    Agente especializado em recrutamento de intermediários financeiros (money mules)
    para operações de lavagem de dinheiro.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.recruited_mules = []
        self.recruitment_campaigns = []
        
        # Perfis de alvos preferenciais
        self.target_profiles = {
            'students': {
                'vulnerability': 0.8,
                'detection_risk': 0.3,
                'capacity': 'low',
                'recruitment_method': 'social_media'
            },
            'unemployed': {
                'vulnerability': 0.9,
                'detection_risk': 0.4,
                'capacity': 'medium',
                'recruitment_method': 'job_boards'
            },
            'elderly': {
                'vulnerability': 0.7,
                'detection_risk': 0.6,
                'capacity': 'medium',
                'recruitment_method': 'romance_scams'
            },
            'immigrants': {
                'vulnerability': 0.8,
                'detection_risk': 0.5,
                'capacity': 'high',
                'recruitment_method': 'community_networks'
            },
            'debt_holders': {
                'vulnerability': 0.9,
                'detection_risk': 0.4,
                'capacity': 'high',
                'recruitment_method': 'financial_desperation'
            }
        }
        
        # Métodos de recrutamento
        self.recruitment_methods = {
            'social_media': {
                'platforms': ['Instagram', 'TikTok', 'Facebook', 'Twitter'],
                'success_rate': 0.15,
                'detection_risk': 0.3,
                'scale': 'high'
            },
            'job_boards': {
                'platforms': ['Indeed', 'LinkedIn', 'Craigslist'],
                'success_rate': 0.25,
                'detection_risk': 0.4,
                'scale': 'medium'
            },
            'romance_scams': {
                'platforms': ['Dating apps', 'Social media'],
                'success_rate': 0.35,
                'detection_risk': 0.6,
                'scale': 'low'
            },
            'community_networks': {
                'platforms': ['Local communities', 'Religious groups'],
                'success_rate': 0.45,
                'detection_risk': 0.7,
                'scale': 'low'
            },
            'financial_desperation': {
                'platforms': ['Debt forums', 'Financial distress groups'],
                'success_rate': 0.55,
                'detection_risk': 0.5,
                'scale': 'medium'
            }
        }
    
    def design_recruitment_campaign(self, target_amount: float, urgency: str = 'medium') -> Dict[str, Any]:
        """
        Projeta uma campanha de recrutamento de mulas financeiras
        
        Args:
            target_amount: Valor alvo para lavagem
            urgency: Urgência da operação (low, medium, high)
            
        Returns:
            Dict com detalhes da campanha
        """
        try:
            # Determinar número de mulas necessárias
            mules_needed = self._calculate_mules_needed(target_amount)
            
            # Selecionar perfis de alvos
            target_profiles = self._select_target_profiles(mules_needed, urgency)
            
            # Escolher métodos de recrutamento
            recruitment_strategy = self._design_recruitment_strategy(target_profiles, urgency)
            
            # Criar narrativas de recrutamento
            recruitment_narratives = self._create_recruitment_narratives(recruitment_strategy)
            
            campaign = {
                'campaign_id': f"RECRUIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'target_amount': target_amount,
                'mules_needed': mules_needed,
                'urgency': urgency,
                'target_profiles': target_profiles,
                'recruitment_strategy': recruitment_strategy,
                'narratives': recruitment_narratives,
                'timeline': self._create_recruitment_timeline(urgency),
                'risk_assessment': self._assess_recruitment_risks(recruitment_strategy),
                'success_probability': self._calculate_success_probability(recruitment_strategy),
                'created_at': datetime.now().isoformat()
            }
            
            self.recruitment_campaigns.append(campaign)
            logger.info(f"Campanha de recrutamento criada: {campaign['campaign_id']}")
            
            return campaign
            
        except Exception as e:
            logger.error(f"Erro ao criar campanha de recrutamento: {str(e)}")
            return {}
    
    def execute_recruitment(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa uma campanha de recrutamento
        
        Args:
            campaign: Campanha de recrutamento
            
        Returns:
            Dict com resultados da execução
        """
        try:
            recruited_mules = []
            recruitment_attempts = 0
            
            for strategy in campaign['recruitment_strategy']:
                method = strategy['method']
                target_count = strategy['target_count']
                
                for _ in range(target_count):
                    recruitment_attempts += 1
                    
                    # Simular tentativa de recrutamento
                    success = self._simulate_recruitment_attempt(method, strategy)
                    
                    if success:
                        mule = self._create_recruited_mule(method, strategy)
                        recruited_mules.append(mule)
            
            # Calcular métricas de sucesso
            success_rate = len(recruited_mules) / recruitment_attempts if recruitment_attempts > 0 else 0
            
            results = {
                'campaign_id': campaign['campaign_id'],
                'recruitment_attempts': recruitment_attempts,
                'successful_recruitments': len(recruited_mules),
                'success_rate': success_rate,
                'recruited_mules': recruited_mules,
                'total_capacity': sum(mule['capacity'] for mule in recruited_mules),
                'average_detection_risk': sum(mule['detection_risk'] for mule in recruited_mules) / len(recruited_mules) if recruited_mules else 0,
                'execution_date': datetime.now().isoformat()
            }
            
            # Adicionar mulas recrutadas à lista global
            self.recruited_mules.extend(recruited_mules)
            
            logger.info(f"Recrutamento executado: {len(recruited_mules)} mulas recrutadas")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na execução do recrutamento: {str(e)}")
            return {}
    
    def manage_mule_network(self) -> Dict[str, Any]:
        """
        Gerencia a rede de mulas recrutadas
        
        Returns:
            Dict com status da rede
        """
        try:
            active_mules = [mule for mule in self.recruited_mules if mule['status'] == 'active']
            
            # Avaliar performance das mulas
            performance_analysis = self._analyze_mule_performance()
            
            # Identificar mulas de risco
            risk_mules = [mule for mule in active_mules if mule['detection_risk'] > 0.7]
            
            # Recomendar ações
            recommendations = self._generate_management_recommendations(active_mules, risk_mules)
            
            network_status = {
                'total_mules': len(self.recruited_mules),
                'active_mules': len(active_mules),
                'high_risk_mules': len(risk_mules),
                'total_capacity': sum(mule['capacity'] for mule in active_mules),
                'average_detection_risk': sum(mule['detection_risk'] for mule in active_mules) / len(active_mules) if active_mules else 0,
                'performance_analysis': performance_analysis,
                'recommendations': recommendations,
                'last_updated': datetime.now().isoformat()
            }
            
            return network_status
            
        except Exception as e:
            logger.error(f"Erro no gerenciamento da rede: {str(e)}")
            return {}
    
    def _calculate_mules_needed(self, target_amount: float) -> int:
        """Calcula número de mulas necessárias baseado no valor alvo"""
        # Capacidade média por mula: $10,000 - $50,000
        avg_capacity = 30000
        return max(1, int(target_amount / avg_capacity))
    
    def _select_target_profiles(self, mules_needed: int, urgency: str) -> List[Dict[str, Any]]:
        """Seleciona perfis de alvos baseado na necessidade e urgência"""
        profiles = []
        
        if urgency == 'high':
            # Focar em perfis mais vulneráveis
            priority_profiles = ['debt_holders', 'unemployed', 'students']
        elif urgency == 'low':
            # Focar em perfis de menor risco
            priority_profiles = ['students', 'elderly', 'immigrants']
        else:
            priority_profiles = list(self.target_profiles.keys())
        
        for i in range(mules_needed):
            profile_type = random.choice(priority_profiles)
            profile = self.target_profiles[profile_type].copy()
            profile['type'] = profile_type
            profiles.append(profile)
        
        return profiles
    
    def _design_recruitment_strategy(self, target_profiles: List[Dict[str, Any]], urgency: str) -> List[Dict[str, Any]]:
        """Projeta estratégia de recrutamento"""
        strategy = []
        
        # Agrupar por método de recrutamento
        method_groups = {}
        for profile in target_profiles:
            method = profile['recruitment_method']
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(profile)
        
        for method, profiles in method_groups.items():
            strategy.append({
                'method': method,
                'target_count': len(profiles),
                'profiles': profiles,
                'method_details': self.recruitment_methods[method],
                'urgency_modifier': self._get_urgency_modifier(urgency)
            })
        
        return strategy
    
    def _create_recruitment_narratives(self, recruitment_strategy: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Cria narrativas para cada método de recrutamento"""
        narratives = {}
        
        narrative_templates = {
            'social_media': [
                "💰 Ganhe dinheiro fácil trabalhando de casa! Apenas receba transferências!",
                "🏠 Trabalho remoto - Seja nosso representante financeiro local",
                "💳 Ajude pessoas internacionais a receber pagamentos - Ganhe comissão!"
            ],
            'job_boards': [
                "Financial Representative - Work from home, handle international transfers",
                "Payment Processing Agent - Help clients receive funds, earn commission",
                "Remote Financial Assistant - Process transactions for overseas clients"
            ],
            'romance_scams': [
                "Meu amor, preciso da sua ajuda para receber uns fundos aqui no Brasil...",
                "Querida, você poderia me ajudar com uma transação bancária?",
                "Amor, tenho uma oportunidade de negócio que precisa da sua conta..."
            ],
            'community_networks': [
                "Oportunidade na nossa comunidade - Ajudar famílias a receber remessas",
                "Trabalho de confiança - Processar pagamentos para nossa rede",
                "Ganhe extra ajudando compatriotas com transferências"
            ],
            'financial_desperation': [
                "Solução rápida para suas dívidas - Trabalho simples, pagamento garantido",
                "Saia do vermelho hoje - Apenas receba transferências em sua conta",
                "Dinheiro urgente? Ajude com transações financeiras simples"
            ]
        }
        
        for strategy in recruitment_strategy:
            method = strategy['method']
            narratives[method] = random.sample(
                narrative_templates.get(method, []), 
                min(len(narrative_templates.get(method, [])), 2)
            )
        
        return narratives
    
    def _create_recruitment_timeline(self, urgency: str) -> Dict[str, str]:
        """Cria cronograma de recrutamento"""
        now = datetime.now()
        
        if urgency == 'high':
            return {
                'start_date': now.isoformat(),
                'recruitment_phase': (now + timedelta(days=3)).isoformat(),
                'activation_phase': (now + timedelta(days=5)).isoformat(),
                'operation_start': (now + timedelta(days=7)).isoformat()
            }
        elif urgency == 'low':
            return {
                'start_date': now.isoformat(),
                'recruitment_phase': (now + timedelta(days=14)).isoformat(),
                'activation_phase': (now + timedelta(days=21)).isoformat(),
                'operation_start': (now + timedelta(days=30)).isoformat()
            }
        else:
            return {
                'start_date': now.isoformat(),
                'recruitment_phase': (now + timedelta(days=7)).isoformat(),
                'activation_phase': (now + timedelta(days=10)).isoformat(),
                'operation_start': (now + timedelta(days=14)).isoformat()
            }
    
    def _assess_recruitment_risks(self, recruitment_strategy: List[Dict[str, Any]]) -> Dict[str, float]:
        """Avalia riscos da estratégia de recrutamento"""
        total_detection_risk = 0
        total_legal_risk = 0
        total_operational_risk = 0
        
        for strategy in recruitment_strategy:
            method_details = strategy['method_details']
            total_detection_risk += method_details['detection_risk']
            
            # Calcular riscos adicionais
            if strategy['method'] in ['social_media', 'job_boards']:
                total_legal_risk += 0.6
            else:
                total_legal_risk += 0.4
            
            total_operational_risk += 0.3
        
        count = len(recruitment_strategy)
        return {
            'detection_risk': total_detection_risk / count if count > 0 else 0,
            'legal_risk': total_legal_risk / count if count > 0 else 0,
            'operational_risk': total_operational_risk / count if count > 0 else 0,
            'overall_risk': (total_detection_risk + total_legal_risk + total_operational_risk) / (count * 3) if count > 0 else 0
        }
    
    def _calculate_success_probability(self, recruitment_strategy: List[Dict[str, Any]]) -> float:
        """Calcula probabilidade de sucesso da campanha"""
        total_success_rate = 0
        
        for strategy in recruitment_strategy:
            method_details = strategy['method_details']
            total_success_rate += method_details['success_rate']
        
        return total_success_rate / len(recruitment_strategy) if recruitment_strategy else 0
    
    def _simulate_recruitment_attempt(self, method: str, strategy: Dict[str, Any]) -> bool:
        """Simula tentativa de recrutamento"""
        success_rate = self.recruitment_methods[method]['success_rate']
        urgency_modifier = strategy['urgency_modifier']
        
        # Ajustar taxa de sucesso baseada na urgência
        adjusted_success_rate = success_rate * urgency_modifier
        
        return random.random() < adjusted_success_rate
    
    def _create_recruited_mule(self, method: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Cria perfil de mula recrutada"""
        profile = random.choice(strategy['profiles'])
        
        return {
            'mule_id': f"MULE_{len(self.recruited_mules) + 1:04d}",
            'recruitment_method': method,
            'profile_type': profile['type'],
            'capacity': random.randint(5000, 50000),
            'detection_risk': profile['detection_risk'] + random.uniform(-0.1, 0.1),
            'vulnerability_score': profile['vulnerability'],
            'status': 'active',
            'recruitment_date': datetime.now().isoformat(),
            'transactions_processed': 0,
            'total_amount_processed': 0,
            'last_activity': datetime.now().isoformat()
        }
    
    def _get_urgency_modifier(self, urgency: str) -> float:
        """Retorna modificador baseado na urgência"""
        modifiers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3
        }
        return modifiers.get(urgency, 1.0)
    
    def _analyze_mule_performance(self) -> Dict[str, Any]:
        """Analisa performance das mulas"""
        if not self.recruited_mules:
            return {}
        
        active_mules = [mule for mule in self.recruited_mules if mule['status'] == 'active']
        
        return {
            'total_transactions': sum(mule['transactions_processed'] for mule in active_mules),
            'total_amount_processed': sum(mule['total_amount_processed'] for mule in active_mules),
            'average_capacity_utilization': sum(mule['total_amount_processed'] / mule['capacity'] for mule in active_mules) / len(active_mules) if active_mules else 0,
            'top_performers': sorted(active_mules, key=lambda x: x['total_amount_processed'], reverse=True)[:3]
        }
    
    def _generate_management_recommendations(self, active_mules: List[Dict[str, Any]], risk_mules: List[Dict[str, Any]]) -> List[str]:
        """Gera recomendações de gerenciamento"""
        recommendations = []
        
        if len(risk_mules) > len(active_mules) * 0.3:
            recommendations.append("Alto número de mulas de risco - considerar substituição")
        
        if len(active_mules) < 5:
            recommendations.append("Rede pequena - recrutar mais mulas para distribuir risco")
        
        low_performers = [mule for mule in active_mules if mule['transactions_processed'] < 5]
        if len(low_performers) > len(active_mules) * 0.4:
            recommendations.append("Muitas mulas inativas - ativar ou substituir")
        
        return recommendations
    
    def get_recruitment_analytics(self) -> Dict[str, Any]:
        """Retorna análise detalhada das operações de recrutamento"""
        return {
            'total_campaigns': len(self.recruitment_campaigns),
            'total_mules_recruited': len(self.recruited_mules),
            'active_mules': len([m for m in self.recruited_mules if m['status'] == 'active']),
            'recruitment_methods_used': list(set(mule['recruitment_method'] for mule in self.recruited_mules)),
            'average_detection_risk': sum(mule['detection_risk'] for mule in self.recruited_mules) / len(self.recruited_mules) if self.recruited_mules else 0,
            'total_network_capacity': sum(mule['capacity'] for mule in self.recruited_mules),
            'network_utilization': sum(mule['total_amount_processed'] for mule in self.recruited_mules) / sum(mule['capacity'] for mule in self.recruited_mules) if self.recruited_mules else 0
        } 