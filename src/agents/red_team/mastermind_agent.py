"""
Red Team Mastermind Agent

This agent acts as the criminal mastermind that creates sophisticated money 
laundering plans. It uses LLM reasoning to generate realistic criminal strategies
that the Operator Agents will execute.
"""

import json
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta
import yaml
from pathlib import Path

# LLM imports - adjust based on your preferred provider
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class MastermindAgent:
    """
    The criminal mastermind that creates money laundering strategies.
    
    This agent uses advanced LLM reasoning to generate sophisticated
    money laundering plans that simulate real-world criminal behavior.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Mastermind Agent with configuration."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.laundering_techniques = self._load_laundering_techniques()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file doesn't exist."""
        return {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.7,
                'max_tokens': 4000
            },
            'simulation': {
                'red_team': {
                    'target_amount_min': 100000,
                    'target_amount_max': 1000000,
                    'complexity_level': 'medium',
                    'techniques_enabled': [
                        'smurfing', 'shell_companies', 'cash_intensive_businesses',
                        'trade_based_laundering', 'cryptocurrency', 'money_mules'
                    ]
                }
            }
        }
    
    def _initialize_llm(self):
        """Initialize the LLM client based on configuration."""
        provider = self.config['llm']['provider']
        
        if provider == 'openai' and openai:
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        elif provider == 'anthropic' and anthropic:
            return anthropic.Anthropic(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        elif provider == 'google' and genai:
            genai.configure(api_key=self.config['llm'].get('api_key', 'your-api-key'))
            return genai.GenerativeModel(self.config['llm']['model'])
        else:
            print(f"Warning: LLM provider {provider} not available. Using mock responses.")
            return None
    
    def _load_laundering_techniques(self) -> Dict[str, Dict]:
        """Load detailed information about money laundering techniques."""
        return {
            'smurfing': {
                'description': 'Breaking large amounts into smaller transactions below reporting thresholds',
                'typical_amount_range': [5000, 9999],
                'frequency': 'high',
                'detection_difficulty': 'medium',
                'required_entities': ['money_mules', 'multiple_accounts']
            },
            'shell_companies': {
                'description': 'Using fake businesses to legitimize illegal funds',
                'typical_amount_range': [50000, 500000],
                'frequency': 'medium',
                'detection_difficulty': 'high',
                'required_entities': ['fake_businesses', 'business_accounts']
            },
            'cash_intensive_businesses': {
                'description': 'Mixing illegal funds with legitimate cash-heavy businesses',
                'typical_amount_range': [10000, 100000],
                'frequency': 'medium',
                'detection_difficulty': 'high',
                'required_entities': ['restaurants', 'laundromats', 'car_washes']
            },
            'trade_based_laundering': {
                'description': 'Over/under-invoicing international trade transactions',
                'typical_amount_range': [100000, 1000000],
                'frequency': 'low',
                'detection_difficulty': 'very_high',
                'required_entities': ['import_export_companies', 'international_partners']
            },
            'cryptocurrency': {
                'description': 'Using digital currencies to obscure transaction trails',
                'typical_amount_range': [1000, 100000],
                'frequency': 'high',
                'detection_difficulty': 'high',
                'required_entities': ['crypto_exchanges', 'privacy_coins']
            },
            'money_mules': {
                'description': 'Using unwitting individuals to transfer money',
                'typical_amount_range': [1000, 10000],
                'frequency': 'high',
                'detection_difficulty': 'medium',
                'required_entities': ['recruited_individuals', 'personal_accounts']
            }
        }
    
    def create_laundering_plan(self, target_amount: float, 
                             complexity_level: str = None,
                             time_constraint: int = 30) -> Dict:
        """
        Create a comprehensive money laundering plan.
        
        Args:
            target_amount: Amount of money to launder
            complexity_level: 'simple', 'medium', or 'complex'
            time_constraint: Number of days to complete the operation
            
        Returns:
            Dictionary containing the complete laundering plan
        """
        if complexity_level is None:
            complexity_level = self.config['simulation']['red_team']['complexity_level']
        
        # Generate the plan using LLM
        plan_prompt = self._create_planning_prompt(target_amount, complexity_level, time_constraint)
        
        if self.llm_client:
            plan_json = self._query_llm(plan_prompt)
        else:
            # Fallback to rule-based plan if LLM not available
            plan_json = self._create_fallback_plan(target_amount, complexity_level)
        
        # Validate and enhance the plan
        validated_plan = self._validate_and_enhance_plan(plan_json, target_amount)
        
        return validated_plan
    
    def _create_planning_prompt(self, target_amount: float, 
                              complexity_level: str, 
                              time_constraint: int) -> str:
        """Create the prompt for LLM planning."""
        
        enabled_techniques = self.config['simulation']['red_team']['techniques_enabled']
        techniques_info = {k: v for k, v in self.laundering_techniques.items() 
                          if k in enabled_techniques}
        
        prompt = f"""
You are a financial crime expert analyzing money laundering techniques for educational purposes. 
Your task is to create a detailed money laundering plan that demonstrates sophisticated criminal methodology.

**IMPORTANT DISCLAIMER**: This is for educational simulation only. Do not use for actual criminal activities.

**Objective**: Create a plan to launder ${target_amount:,.2f}
**Complexity Level**: {complexity_level}
**Time Constraint**: {time_constraint} days
**Available Techniques**: {', '.join(enabled_techniques)}

**Technique Details**:
{json.dumps(techniques_info, indent=2)}

**Requirements**:
1. The plan must be realistic and sophisticated
2. Use multiple techniques to avoid detection
3. Include specific steps with timing, amounts, and entities
4. Consider detection avoidance strategies
5. Account for transaction costs and losses

**Output Format** (JSON):
{{
    "plan_id": "unique_plan_identifier",
    "objective": "Launder ${target_amount:,.2f}",
    "complexity_level": "{complexity_level}",
    "estimated_duration_days": {time_constraint},
    "techniques_used": ["technique1", "technique2"],
    "total_cost_percentage": 0.15,
    "detection_risk_level": "medium",
    "steps": [
        {{
            "step_number": 1,
            "technique": "smurfing",
            "description": "Break initial amount into smaller deposits",
            "timing": "Days 1-5",
            "entities_involved": ["money_mule_1", "money_mule_2"],
            "transactions": [
                {{
                    "from": "criminal_account",
                    "to": "money_mule_1",
                    "amount": 9500.00,
                    "description": "Cash deposit",
                    "timing": "Day 1"
                }}
            ],
            "risk_factors": ["frequent_deposits", "round_amounts"],
            "mitigation_strategies": ["vary_amounts", "use_different_locations"]
        }}
    ],
    "entities_required": [
        {{
            "entity_type": "money_mule",
            "count": 5,
            "description": "Recruited individuals with clean accounts",
            "risk_level": "medium"
        }}
    ],
    "success_metrics": [
        "funds_successfully_laundered",
        "detection_avoided",
        "time_within_constraint"
    ],
    "contingency_plans": [
        "If detected early, switch to cryptocurrency route",
        "If mules compromised, activate shell company backup"
    ]
}}

**Critical Instructions**:
- Be specific with amounts, timing, and entities
- Include realistic transaction descriptions
- Consider law enforcement detection methods
- Plan for contingencies and backup routes
- Ensure mathematical accuracy (amounts should sum correctly)
- Include operational security measures

Generate a sophisticated, realistic money laundering plan following this format:
"""
        
        return prompt
    
    def _query_llm(self, prompt: str) -> Dict:
        """Query the LLM with the planning prompt."""
        try:
            provider = self.config['llm']['provider']
            
            if provider == 'openai':
                response = self.llm_client.chat.completions.create(
                    model=self.config['llm']['model'],
                    messages=[
                        {"role": "system", "content": "You are a financial crime expert creating educational simulations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens']
                )
                content = response.choices[0].message.content
                
            elif provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model=self.config['llm']['model'],
                    max_tokens=self.config['llm']['max_tokens'],
                    temperature=self.config['llm']['temperature'],
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                
            elif provider == 'google':
                response = self.llm_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config['llm']['temperature'],
                        max_output_tokens=self.config['llm']['max_tokens']
                    )
                )
                content = response.text
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in LLM response")
                
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return self._create_fallback_plan(target_amount=500000, complexity_level='medium')
    
    def _create_fallback_plan(self, target_amount: float, complexity_level: str) -> Dict:
        """Create a fallback plan if LLM is not available."""
        plan_id = f"FALLBACK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simple smurfing plan
        num_mules = min(10, max(3, int(target_amount / 50000)))
        amount_per_mule = target_amount / num_mules
        transactions_per_mule = max(1, int(amount_per_mule / 9500))
        amount_per_transaction = amount_per_mule / transactions_per_mule
        
        return {
            "plan_id": plan_id,
            "objective": f"Launder ${target_amount:,.2f}",
            "complexity_level": complexity_level,
            "estimated_duration_days": 14,
            "techniques_used": ["smurfing", "money_mules"],
            "total_cost_percentage": 0.10,
            "detection_risk_level": "medium",
            "steps": [
                {
                    "step_number": 1,
                    "technique": "smurfing",
                    "description": "Break initial amount into smaller deposits using money mules",
                    "timing": "Days 1-7",
                    "entities_involved": [f"money_mule_{i+1}" for i in range(num_mules)],
                    "transactions": [
                        {
                            "from": "criminal_source",
                            "to": f"money_mule_{i+1}",
                            "amount": round(amount_per_transaction, 2),
                            "description": "Cash deposit",
                            "timing": f"Day {(i % 7) + 1}"
                        }
                        for i in range(num_mules * transactions_per_mule)
                    ],
                    "risk_factors": ["frequent_deposits", "similar_amounts"],
                    "mitigation_strategies": ["vary_timing", "use_different_branches"]
                }
            ],
            "entities_required": [
                {
                    "entity_type": "money_mule",
                    "count": num_mules,
                    "description": "Recruited individuals with clean accounts",
                    "risk_level": "medium"
                }
            ],
            "success_metrics": ["funds_successfully_laundered", "detection_avoided"],
            "contingency_plans": ["Switch to cryptocurrency if detected"]
        }
    
    def _validate_and_enhance_plan(self, plan: Dict, target_amount: float) -> Dict:
        """Validate and enhance the generated plan."""
        # Add metadata
        plan['created_at'] = datetime.now().isoformat()
        plan['target_amount'] = target_amount
        plan['plan_status'] = 'pending'
        
        # Validate amounts
        total_planned = 0
        for step in plan.get('steps', []):
            for transaction in step.get('transactions', []):
                total_planned += transaction.get('amount', 0)
        
        plan['total_planned_amount'] = total_planned
        plan['amount_variance'] = abs(total_planned - target_amount) / target_amount
        
        # Add risk assessment
        plan['risk_assessment'] = self._assess_plan_risk(plan)
        
        # Add execution timeline
        plan['execution_timeline'] = self._create_execution_timeline(plan)
        
        return plan
    
    def _assess_plan_risk(self, plan: Dict) -> Dict:
        """Assess the risk level of the plan."""
        techniques_used = plan.get('techniques_used', [])
        
        # Base risk scores for different techniques
        risk_scores = {
            'smurfing': 0.6,
            'shell_companies': 0.4,
            'cash_intensive_businesses': 0.3,
            'trade_based_laundering': 0.2,
            'cryptocurrency': 0.5,
            'money_mules': 0.7
        }
        
        # Calculate overall risk
        total_risk = sum(risk_scores.get(technique, 0.5) for technique in techniques_used)
        avg_risk = total_risk / len(techniques_used) if techniques_used else 0.5
        
        return {
            'overall_risk_score': round(avg_risk, 2),
            'risk_level': 'high' if avg_risk > 0.7 else 'medium' if avg_risk > 0.4 else 'low',
            'primary_risk_factors': [
                'transaction_patterns',
                'entity_connections',
                'timing_analysis',
                'amount_structuring'
            ],
            'detection_probability': round(avg_risk * 0.8, 2)  # Slightly lower than risk score
        }
    
    def _create_execution_timeline(self, plan: Dict) -> List[Dict]:
        """Create a detailed execution timeline."""
        timeline = []
        
        for step in plan.get('steps', []):
            timeline.append({
                'step_number': step.get('step_number'),
                'technique': step.get('technique'),
                'timing': step.get('timing'),
                'description': step.get('description'),
                'entities_count': len(step.get('entities_involved', [])),
                'transactions_count': len(step.get('transactions', [])),
                'estimated_duration': self._estimate_step_duration(step)
            })
        
        return timeline
    
    def _estimate_step_duration(self, step: Dict) -> int:
        """Estimate duration for a step in days."""
        technique = step.get('technique', '')
        transaction_count = len(step.get('transactions', []))
        
        base_duration = {
            'smurfing': 1,
            'shell_companies': 7,
            'cash_intensive_businesses': 3,
            'trade_based_laundering': 14,
            'cryptocurrency': 2,
            'money_mules': 1
        }
        
        base = base_duration.get(technique, 3)
        return max(1, base + (transaction_count // 10))  # Add time for more transactions
    
    def get_plan_summary(self, plan: Dict) -> str:
        """Generate a human-readable summary of the plan."""
        summary = f"""
Money Laundering Plan Summary
============================

Plan ID: {plan.get('plan_id', 'Unknown')}
Objective: {plan.get('objective', 'Unknown')}
Complexity: {plan.get('complexity_level', 'Unknown')}
Duration: {plan.get('estimated_duration_days', 'Unknown')} days

Techniques Used: {', '.join(plan.get('techniques_used', []))}
Risk Level: {plan.get('risk_assessment', {}).get('risk_level', 'Unknown')}
Detection Probability: {plan.get('risk_assessment', {}).get('detection_probability', 'Unknown')}

Steps Overview:
"""
        
        for step in plan.get('steps', []):
            summary += f"\n{step.get('step_number', '?')}. {step.get('description', 'No description')}"
            summary += f"\n   Technique: {step.get('technique', 'Unknown')}"
            summary += f"\n   Timing: {step.get('timing', 'Unknown')}"
            summary += f"\n   Transactions: {len(step.get('transactions', []))}"
        
        return summary


def main():
    """Main function for testing the Mastermind Agent."""
    print("Initializing Red Team Mastermind Agent...")
    
    mastermind = MastermindAgent()
    
    # Test plan creation
    target_amount = 500000
    print(f"\nCreating laundering plan for ${target_amount:,.2f}...")
    
    plan = mastermind.create_laundering_plan(
        target_amount=target_amount,
        complexity_level='medium',
        time_constraint=21
    )
    
    print("\nGenerated Plan:")
    print("=" * 50)
    print(json.dumps(plan, indent=2, default=str))
    
    print("\nPlan Summary:")
    print("=" * 50)
    print(mastermind.get_plan_summary(plan))


if __name__ == "__main__":
    main() 