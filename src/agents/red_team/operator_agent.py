"""
Red Team Operator Agent

This agent executes the money laundering plans created by the Mastermind Agent.
It generates synthetic criminal transactions and injects them into the normal
transaction flow to simulate real-world money laundering operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import json
from typing import Dict, List, Tuple, Any
import uuid
from pathlib import Path


class OperatorAgent:
    """
    Executes money laundering plans by generating synthetic criminal transactions.
    
    This agent takes the strategic plan from the Mastermind Agent and converts
    it into actual transaction data that gets mixed with normal transactions.
    """
    
    def __init__(self, customers: List[Dict] = None, businesses: List[Dict] = None):
        """
        Initialize the Operator Agent.
        
        Args:
            customers: List of customer entities from normal data
            businesses: List of business entities from normal data
        """
        self.fake = Faker()
        self.customers = customers or []
        self.businesses = businesses or []
        self.criminal_entities = {}
        self.execution_log = []
        
    def execute_plan(self, plan: Dict, normal_transactions: pd.DataFrame = None) -> Dict:
        """
        Execute a money laundering plan and generate criminal transactions.
        
        Args:
            plan: The plan from MastermindAgent
            normal_transactions: DataFrame of normal transactions for context
            
        Returns:
            Dictionary containing execution results and generated transactions
        """
        print(f"Executing plan: {plan.get('plan_id', 'Unknown')}")
        
        # Initialize execution context
        execution_context = {
            'plan_id': plan.get('plan_id'),
            'start_time': datetime.now(),
            'target_amount': plan.get('target_amount', 0),
            'status': 'executing',
            'current_step': 0,
            'total_steps': len(plan.get('steps', [])),
            'transactions_generated': [],
            'entities_created': {},
            'execution_log': []
        }
        
        try:
            # Create required criminal entities
            self._create_criminal_entities(plan, execution_context)
            
            # Execute each step of the plan
            for step in plan.get('steps', []):
                step_result = self._execute_step(step, execution_context, normal_transactions)
                execution_context['execution_log'].append(step_result)
                execution_context['current_step'] += 1
                
                if step_result['status'] == 'failed':
                    execution_context['status'] = 'failed'
                    break
            
            # Finalize execution
            if execution_context['status'] != 'failed':
                execution_context['status'] = 'completed'
                
            execution_context['end_time'] = datetime.now()
            execution_context['duration'] = (
                execution_context['end_time'] - execution_context['start_time']
            ).total_seconds()
            
            # Generate final transaction DataFrame
            transactions_df = self._create_transactions_dataframe(
                execution_context['transactions_generated']
            )
            
            return {
                'execution_context': execution_context,
                'criminal_transactions': transactions_df,
                'entities_created': execution_context['entities_created'],
                'success': execution_context['status'] == 'completed'
            }
            
        except Exception as e:
            execution_context['status'] = 'error'
            execution_context['error'] = str(e)
            execution_context['end_time'] = datetime.now()
            
            return {
                'execution_context': execution_context,
                'criminal_transactions': pd.DataFrame(),
                'entities_created': {},
                'success': False,
                'error': str(e)
            }
    
    def _create_criminal_entities(self, plan: Dict, context: Dict):
        """Create the criminal entities required for the plan."""
        entities_required = plan.get('entities_required', [])
        
        context['entities_created'] = {}
        
        for entity_spec in entities_required:
            entity_type = entity_spec.get('entity_type')
            count = entity_spec.get('count', 1)
            
            if entity_type == 'money_mule':
                entities = self._create_money_mules(count, entity_spec)
            elif entity_type == 'shell_company':
                entities = self._create_shell_companies(count, entity_spec)
            elif entity_type == 'cash_business':
                entities = self._create_cash_businesses(count, entity_spec)
            elif entity_type == 'crypto_account':
                entities = self._create_crypto_accounts(count, entity_spec)
            else:
                entities = self._create_generic_entities(entity_type, count, entity_spec)
            
            context['entities_created'][entity_type] = entities
            
            # Add to criminal entities registry
            if entity_type not in self.criminal_entities:
                self.criminal_entities[entity_type] = []
            self.criminal_entities[entity_type].extend(entities)
    
    def _create_money_mules(self, count: int, spec: Dict) -> List[Dict]:
        """Create money mule entities."""
        mules = []
        
        for i in range(count):
            profile = self.fake.profile()
            
            mule = {
                'entity_id': f"MULE_{uuid.uuid4().hex[:8].upper()}",
                'entity_type': 'money_mule',
                'name': profile['name'],
                'email': profile['mail'],
                'phone': self.fake.phone_number(),
                'address': profile['address'],
                'date_of_birth': profile['birthdate'],
                'account_number': f"ACC_{self.fake.random_int(100000, 999999)}",
                'bank': random.choice(['Chase', 'Bank of America', 'Wells Fargo', 'Citibank']),
                'recruitment_method': random.choice(['online_ad', 'social_media', 'referral']),
                'risk_level': spec.get('risk_level', 'medium'),
                'status': 'active',
                'created_at': datetime.now(),
                'transaction_limit': random.randint(5000, 15000),
                'is_witting': random.choice([True, False])  # Knows they're part of scheme
            }
            mules.append(mule)
        
        return mules
    
    def _create_shell_companies(self, count: int, spec: Dict) -> List[Dict]:
        """Create shell company entities."""
        companies = []
        
        business_types = [
            'Consulting Services LLC', 'Import Export Inc', 'Trading Company LLC',
            'Management Services Corp', 'Investment Holdings LLC', 'Advisory Services Inc'
        ]
        
        for i in range(count):
            company = {
                'entity_id': f"SHELL_{uuid.uuid4().hex[:8].upper()}",
                'entity_type': 'shell_company',
                'name': f"{self.fake.company()} {random.choice(business_types)}",
                'ein': self.fake.ssn(),  # Using SSN format for EIN
                'registration_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'address': self.fake.address(),
                'phone': self.fake.phone_number(),
                'business_type': random.choice(business_types),
                'bank_account': f"BUS_{self.fake.random_int(100000, 999999)}",
                'bank': random.choice(['Chase Business', 'Bank of America Business', 'Wells Fargo Business']),
                'has_website': random.choice([True, False]),
                'has_employees': random.choice([True, False]),
                'legitimate_activity': random.uniform(0.0, 0.3),  # Very low legitimate activity
                'risk_level': spec.get('risk_level', 'high'),
                'status': 'active',
                'created_at': datetime.now()
            }
            companies.append(company)
        
        return companies
    
    def _create_cash_businesses(self, count: int, spec: Dict) -> List[Dict]:
        """Create cash-intensive business entities."""
        businesses = []
        
        cash_business_types = [
            'Restaurant', 'Laundromat', 'Car Wash', 'Convenience Store',
            'Nail Salon', 'Barber Shop', 'Taxi Service', 'Vending Machine Business'
        ]
        
        for i in range(count):
            business = {
                'entity_id': f"CASH_{uuid.uuid4().hex[:8].upper()}",
                'entity_type': 'cash_business',
                'name': f"{self.fake.company()} {random.choice(cash_business_types)}",
                'business_type': random.choice(cash_business_types),
                'address': self.fake.address(),
                'phone': self.fake.phone_number(),
                'bank_account': f"CASH_{self.fake.random_int(100000, 999999)}",
                'bank': random.choice(['Local Bank', 'Community Bank', 'Regional Credit Union']),
                'legitimate_revenue': random.randint(100000, 500000),
                'cash_percentage': random.uniform(0.6, 0.9),  # High cash percentage
                'risk_level': spec.get('risk_level', 'medium'),
                'status': 'active',
                'created_at': datetime.now()
            }
            businesses.append(business)
        
        return businesses
    
    def _create_crypto_accounts(self, count: int, spec: Dict) -> List[Dict]:
        """Create cryptocurrency account entities."""
        accounts = []
        
        exchanges = ['Binance', 'Coinbase', 'Kraken', 'Bitfinex', 'KuCoin']
        cryptocurrencies = ['Bitcoin', 'Ethereum', 'Monero', 'Zcash', 'Litecoin']
        
        for i in range(count):
            account = {
                'entity_id': f"CRYPTO_{uuid.uuid4().hex[:8].upper()}",
                'entity_type': 'crypto_account',
                'exchange': random.choice(exchanges),
                'wallet_address': self.fake.sha256()[:42],  # Simulated wallet address
                'primary_currency': random.choice(cryptocurrencies),
                'account_type': random.choice(['personal', 'business', 'anonymous']),
                'kyc_verified': random.choice([True, False]),
                'risk_level': spec.get('risk_level', 'high'),
                'status': 'active',
                'created_at': datetime.now()
            }
            accounts.append(account)
        
        return accounts
    
    def _create_generic_entities(self, entity_type: str, count: int, spec: Dict) -> List[Dict]:
        """Create generic entities for unknown types."""
        entities = []
        
        for i in range(count):
            entity = {
                'entity_id': f"{entity_type.upper()}_{uuid.uuid4().hex[:8].upper()}",
                'entity_type': entity_type,
                'name': self.fake.name(),
                'description': spec.get('description', f'Generic {entity_type}'),
                'risk_level': spec.get('risk_level', 'medium'),
                'status': 'active',
                'created_at': datetime.now()
            }
            entities.append(entity)
        
        return entities
    
    def _execute_step(self, step: Dict, context: Dict, normal_transactions: pd.DataFrame) -> Dict:
        """Execute a single step of the laundering plan."""
        step_result = {
            'step_number': step.get('step_number'),
            'technique': step.get('technique'),
            'status': 'executing',
            'start_time': datetime.now(),
            'transactions_created': [],
            'entities_used': step.get('entities_involved', []),
            'errors': []
        }
        
        try:
            # Get the technique-specific executor
            technique = step.get('technique')
            
            if technique == 'smurfing':
                transactions = self._execute_smurfing(step, context)
            elif technique == 'shell_companies':
                transactions = self._execute_shell_companies(step, context)
            elif technique == 'cash_intensive_businesses':
                transactions = self._execute_cash_businesses(step, context)
            elif technique == 'cryptocurrency':
                transactions = self._execute_cryptocurrency(step, context)
            elif technique == 'money_mules':
                transactions = self._execute_money_mules(step, context)
            else:
                transactions = self._execute_generic_technique(step, context)
            
            step_result['transactions_created'] = transactions
            context['transactions_generated'].extend(transactions)
            step_result['status'] = 'completed'
            
        except Exception as e:
            step_result['status'] = 'failed'
            step_result['errors'].append(str(e))
        
        step_result['end_time'] = datetime.now()
        step_result['duration'] = (
            step_result['end_time'] - step_result['start_time']
        ).total_seconds()
        
        return step_result
    
    def _execute_smurfing(self, step: Dict, context: Dict) -> List[Dict]:
        """Execute smurfing technique."""
        transactions = []
        
        # Get planned transactions from the step
        planned_transactions = step.get('transactions', [])
        
        for tx_plan in planned_transactions:
            # Create actual transaction
            transaction = self._create_transaction(
                tx_plan, context, 'smurfing'
            )
            transactions.append(transaction)
        
        return transactions
    
    def _execute_shell_companies(self, step: Dict, context: Dict) -> List[Dict]:
        """Execute shell company technique."""
        transactions = []
        
        planned_transactions = step.get('transactions', [])
        
        for tx_plan in planned_transactions:
            transaction = self._create_transaction(
                tx_plan, context, 'shell_companies'
            )
            transactions.append(transaction)
        
        return transactions
    
    def _execute_cash_businesses(self, step: Dict, context: Dict) -> List[Dict]:
        """Execute cash-intensive business technique."""
        transactions = []
        
        planned_transactions = step.get('transactions', [])
        
        for tx_plan in planned_transactions:
            transaction = self._create_transaction(
                tx_plan, context, 'cash_intensive_businesses'
            )
            transactions.append(transaction)
        
        return transactions
    
    def _execute_cryptocurrency(self, step: Dict, context: Dict) -> List[Dict]:
        """Execute cryptocurrency technique."""
        transactions = []
        
        planned_transactions = step.get('transactions', [])
        
        for tx_plan in planned_transactions:
            transaction = self._create_transaction(
                tx_plan, context, 'cryptocurrency'
            )
            transactions.append(transaction)
        
        return transactions
    
    def _execute_money_mules(self, step: Dict, context: Dict) -> List[Dict]:
        """Execute money mules technique."""
        transactions = []
        
        planned_transactions = step.get('transactions', [])
        
        for tx_plan in planned_transactions:
            transaction = self._create_transaction(
                tx_plan, context, 'money_mules'
            )
            transactions.append(transaction)
        
        return transactions
    
    def _execute_generic_technique(self, step: Dict, context: Dict) -> List[Dict]:
        """Execute generic technique."""
        transactions = []
        
        planned_transactions = step.get('transactions', [])
        
        for tx_plan in planned_transactions:
            transaction = self._create_transaction(
                tx_plan, context, 'generic'
            )
            transactions.append(transaction)
        
        return transactions
    
    def _create_transaction(self, tx_plan: Dict, context: Dict, technique: str) -> Dict:
        """Create a single criminal transaction."""
        # Generate unique transaction ID
        tx_id = f"CRM_{uuid.uuid4().hex[:8].upper()}"
        
        # Get sender and receiver entities
        sender = self._resolve_entity(tx_plan.get('from'), context)
        receiver = self._resolve_entity(tx_plan.get('to'), context)
        
        # Generate realistic timestamp
        timestamp = self._generate_criminal_timestamp(tx_plan.get('timing'))
        
        # Create transaction
        transaction = {
            'transaction_id': tx_id,
            'timestamp': timestamp,
            'sender_id': sender['id'],
            'sender_name': sender['name'],
            'sender_type': sender['type'],
            'receiver_id': receiver['id'],
            'receiver_name': receiver['name'],
            'receiver_type': receiver['type'],
            'amount': tx_plan.get('amount', 0),
            'transaction_type': f'criminal_{technique}',
            'currency': 'USD',
            'channel': random.choice(['online', 'atm', 'branch', 'wire']),
            'status': 'completed',
            'description': tx_plan.get('description', f'Criminal transaction - {technique}'),
            'location': self.fake.city(),
            'is_suspicious': True,  # Mark as suspicious for evaluation
            'criminal_technique': technique,
            'plan_id': context.get('plan_id'),
            'step_number': tx_plan.get('step_number'),
            'risk_indicators': self._generate_risk_indicators(technique, tx_plan)
        }
        
        return transaction
    
    def _resolve_entity(self, entity_ref: str, context: Dict) -> Dict:
        """Resolve an entity reference to actual entity data."""
        # Check if it's a criminal entity
        for entity_type, entities in context.get('entities_created', {}).items():
            for entity in entities:
                if entity['entity_id'] == entity_ref or entity.get('name') == entity_ref:
                    return {
                        'id': entity['entity_id'],
                        'name': entity.get('name', entity_ref),
                        'type': entity_type
                    }
        
        # Check if it's a reference to normal entities
        if entity_ref.startswith('CUST_') and self.customers:
            customer = random.choice(self.customers)
            return {
                'id': customer['customer_id'],
                'name': customer['name'],
                'type': 'customer'
            }
        
        if entity_ref.startswith('BUS_') and self.businesses:
            business = random.choice(self.businesses)
            return {
                'id': business['business_id'],
                'name': business['name'],
                'type': 'business'
            }
        
        # Create a placeholder entity
        return {
            'id': f"PLACEHOLDER_{uuid.uuid4().hex[:8].upper()}",
            'name': entity_ref,
            'type': 'unknown'
        }
    
    def _generate_criminal_timestamp(self, timing_spec: str) -> datetime:
        """Generate realistic timestamp for criminal transactions."""
        # Parse timing specification (e.g., "Day 1", "Days 1-5")
        base_date = datetime.now() - timedelta(days=random.randint(1, 30))
        
        if 'Day' in timing_spec:
            try:
                # Extract day number
                day_part = timing_spec.split('Day')[-1].strip()
                if '-' in day_part:
                    # Range like "1-5"
                    start_day, end_day = map(int, day_part.split('-'))
                    day_offset = random.randint(start_day, end_day)
                else:
                    # Single day like "1"
                    day_offset = int(day_part)
                
                target_date = base_date + timedelta(days=day_offset)
            except:
                target_date = base_date
        else:
            target_date = base_date
        
        # Add random time within the day
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return target_date.replace(hour=hour, minute=minute, second=second)
    
    def _generate_risk_indicators(self, technique: str, tx_plan: Dict) -> List[str]:
        """Generate risk indicators for the transaction."""
        indicators = []
        
        # Common indicators
        amount = tx_plan.get('amount', 0)
        if amount > 9000 and amount < 10000:
            indicators.append('amount_just_below_threshold')
        
        if amount % 100 == 0:
            indicators.append('round_amount')
        
        # Technique-specific indicators
        if technique == 'smurfing':
            indicators.extend(['structured_amount', 'frequent_deposits'])
        elif technique == 'shell_companies':
            indicators.extend(['shell_company_involved', 'no_legitimate_business'])
        elif technique == 'cash_intensive_businesses':
            indicators.extend(['cash_business', 'revenue_inconsistency'])
        elif technique == 'cryptocurrency':
            indicators.extend(['crypto_conversion', 'anonymity_enhanced'])
        elif technique == 'money_mules':
            indicators.extend(['third_party_account', 'rapid_movement'])
        
        return indicators
    
    def _create_transactions_dataframe(self, transactions: List[Dict]) -> pd.DataFrame:
        """Convert transaction list to DataFrame with proper formatting."""
        if not transactions:
            return pd.DataFrame()
        
        df = pd.DataFrame(transactions)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add derived features similar to normal transactions
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)
        
        # Amount-based features
        df['amount_rounded'] = df['amount'].round(-2)
        df['is_round_amount'] = df['amount'] % 100 == 0
        df['amount_category'] = pd.cut(df['amount'], 
                                     bins=[0, 100, 1000, 10000, float('inf')],
                                     labels=['small', 'medium', 'large', 'very_large'])
        
        return df
    
    def get_execution_summary(self, execution_result: Dict) -> str:
        """Generate a summary of the execution results."""
        context = execution_result.get('execution_context', {})
        
        summary = f"""
Criminal Plan Execution Summary
==============================

Plan ID: {context.get('plan_id', 'Unknown')}
Status: {context.get('status', 'Unknown')}
Duration: {context.get('duration', 0):.2f} seconds

Target Amount: ${context.get('target_amount', 0):,.2f}
Steps Executed: {context.get('current_step', 0)}/{context.get('total_steps', 0)}

Transactions Generated: {len(context.get('transactions_generated', []))}
Entities Created: {sum(len(entities) for entities in context.get('entities_created', {}).values())}

Success: {execution_result.get('success', False)}
"""
        
        if execution_result.get('error'):
            summary += f"\nError: {execution_result['error']}"
        
        return summary


def main():
    """Main function for testing the Operator Agent."""
    print("Testing Red Team Operator Agent...")
    
    # Create a sample plan (normally this would come from MastermindAgent)
    sample_plan = {
        'plan_id': 'TEST_PLAN_001',
        'target_amount': 100000,
        'steps': [
            {
                'step_number': 1,
                'technique': 'smurfing',
                'description': 'Break amount into smaller deposits',
                'entities_involved': ['money_mule_1', 'money_mule_2'],
                'transactions': [
                    {
                        'from': 'criminal_source',
                        'to': 'money_mule_1',
                        'amount': 9500.00,
                        'description': 'Cash deposit',
                        'timing': 'Day 1'
                    },
                    {
                        'from': 'criminal_source',
                        'to': 'money_mule_2',
                        'amount': 9500.00,
                        'description': 'Cash deposit',
                        'timing': 'Day 2'
                    }
                ]
            }
        ],
        'entities_required': [
            {
                'entity_type': 'money_mule',
                'count': 2,
                'description': 'Recruited individuals',
                'risk_level': 'medium'
            }
        ]
    }
    
    # Initialize operator
    operator = OperatorAgent()
    
    # Execute plan
    print("\nExecuting sample plan...")
    result = operator.execute_plan(sample_plan)
    
    # Display results
    print("\nExecution Results:")
    print("=" * 50)
    print(operator.get_execution_summary(result))
    
    # Show generated transactions
    if not result['criminal_transactions'].empty:
        print("\nGenerated Criminal Transactions:")
        print("=" * 50)
        print(result['criminal_transactions'][['transaction_id', 'timestamp', 'amount', 'criminal_technique']].to_string())


if __name__ == "__main__":
    main() 