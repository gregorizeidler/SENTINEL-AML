"""
Synthetic Transaction Data Generator for AML-FT Simulation

This module generates realistic financial transaction data to serve as the 
background "noise" for the adversarial simulation. The Red Team will inject 
criminal transactions into this normal flow.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
from typing import Dict, List, Tuple
import yaml
from pathlib import Path

class TransactionGenerator:
    """Generates synthetic but realistic financial transaction data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the transaction generator with configuration."""
        self.fake = Faker()
        self.config = self._load_config(config_path)
        self.customers = []
        self.businesses = []
        self.transactions = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            return {
                'data': {
                    'normal_transactions': {
                        'count': 50000,
                        'time_period_days': 365,
                        'customer_count': 5000,
                        'business_count': 500
                    },
                    'transaction_types': [
                        {'type': 'retail_purchase', 'frequency': 0.4, 'amount_range': [10, 500]},
                        {'type': 'salary_payment', 'frequency': 0.15, 'amount_range': [2000, 8000]},
                        {'type': 'bill_payment', 'frequency': 0.2, 'amount_range': [50, 1000]},
                        {'type': 'p2p_transfer', 'frequency': 0.15, 'amount_range': [20, 2000]},
                        {'type': 'business_payment', 'frequency': 0.1, 'amount_range': [1000, 50000]}
                    ]
                }
            }
    
    def generate_customers(self) -> List[Dict]:
        """Generate synthetic customer profiles."""
        customers = []
        customer_count = self.config['data']['normal_transactions']['customer_count']
        
        for i in range(customer_count):
            # Create realistic customer profile
            profile = self.fake.profile()
            
            customer = {
                'customer_id': f"CUST_{i+1:06d}",
                'name': profile['name'],
                'email': profile['mail'],
                'address': profile['address'],
                'phone': self.fake.phone_number(),
                'date_of_birth': profile['birthdate'],
                'account_opened': self.fake.date_between(start_date='-5y', end_date='today'),
                'account_type': random.choice(['checking', 'savings', 'business']),
                'risk_level': random.choices(['low', 'medium', 'high'], weights=[0.8, 0.15, 0.05])[0],
                'occupation': profile['job'],
                'annual_income': random.randint(25000, 150000),
                'credit_score': random.randint(300, 850)
            }
            customers.append(customer)
        
        self.customers = customers
        return customers
    
    def generate_businesses(self) -> List[Dict]:
        """Generate synthetic business profiles."""
        businesses = []
        business_count = self.config['data']['normal_transactions']['business_count']
        
        business_types = [
            'Restaurant', 'Retail Store', 'Consulting Firm', 'Tech Company',
            'Law Firm', 'Medical Practice', 'Construction Company', 'Real Estate',
            'Insurance Agency', 'Auto Dealership', 'Grocery Store', 'Gas Station'
        ]
        
        for i in range(business_count):
            business = {
                'business_id': f"BUS_{i+1:06d}",
                'name': self.fake.company(),
                'business_type': random.choice(business_types),
                'registration_date': self.fake.date_between(start_date='-10y', end_date='today'),
                'address': self.fake.address(),
                'phone': self.fake.phone_number(),
                'ein': self.fake.ssn(),  # Using SSN format for EIN
                'annual_revenue': random.randint(50000, 5000000),
                'employee_count': random.randint(1, 500),
                'risk_level': random.choices(['low', 'medium', 'high'], weights=[0.7, 0.25, 0.05])[0]
            }
            businesses.append(business)
        
        self.businesses = businesses
        return businesses
    
    def generate_transactions(self) -> pd.DataFrame:
        """Generate synthetic transaction data."""
        if not self.customers or not self.businesses:
            self.generate_customers()
            self.generate_businesses()
        
        transactions = []
        transaction_count = self.config['data']['normal_transactions']['count']
        time_period_days = self.config['data']['normal_transactions']['time_period_days']
        
        # Create date range for transactions
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        for i in range(transaction_count):
            # Select transaction type based on frequency weights
            tx_type_config = self._select_transaction_type()
            
            # Generate transaction based on type
            transaction = self._generate_single_transaction(
                i + 1, tx_type_config, start_date, end_date
            )
            transactions.append(transaction)
        
        # Convert to DataFrame and add derived features
        df = pd.DataFrame(transactions)
        df = self._add_derived_features(df)
        
        self.transactions = df
        return df
    
    def _select_transaction_type(self) -> Dict:
        """Select transaction type based on configured frequencies."""
        types = self.config['data']['transaction_types']
        weights = [t['frequency'] for t in types]
        return random.choices(types, weights=weights)[0]
    
    def _generate_single_transaction(self, tx_id: int, tx_type: Dict, 
                                   start_date: datetime, end_date: datetime) -> Dict:
        """Generate a single transaction based on type."""
        
        # Generate timestamp with realistic patterns
        timestamp = self._generate_realistic_timestamp(start_date, end_date, tx_type['type'])
        
        # Generate amount within range
        amount = random.uniform(tx_type['amount_range'][0], tx_type['amount_range'][1])
        
        # Select sender and receiver based on transaction type
        sender, receiver = self._select_transaction_parties(tx_type['type'])
        
        transaction = {
            'transaction_id': f"TX_{tx_id:08d}",
            'timestamp': timestamp,
            'sender_id': sender['id'],
            'sender_name': sender['name'],
            'sender_type': sender['type'],
            'receiver_id': receiver['id'],
            'receiver_name': receiver['name'],
            'receiver_type': receiver['type'],
            'amount': round(amount, 2),
            'transaction_type': tx_type['type'],
            'currency': 'USD',
            'channel': random.choice(['online', 'atm', 'branch', 'mobile']),
            'status': random.choices(['completed', 'pending', 'failed'], 
                                   weights=[0.95, 0.03, 0.02])[0],
            'description': self._generate_transaction_description(tx_type['type']),
            'location': self.fake.city(),
            'is_suspicious': False  # Normal transactions are not suspicious
        }
        
        return transaction
    
    def _generate_realistic_timestamp(self, start_date: datetime, 
                                    end_date: datetime, tx_type: str) -> datetime:
        """Generate realistic timestamps based on transaction type."""
        
        # Generate random date within range
        random_date = self.fake.date_between(start_date=start_date.date(), 
                                           end_date=end_date.date())
        
        # Generate realistic time based on transaction type
        if tx_type == 'salary_payment':
            # Salary payments typically happen on specific days of month
            hour = random.randint(9, 17)  # Business hours
            minute = random.randint(0, 59)
        elif tx_type == 'retail_purchase':
            # Retail purchases happen throughout the day but peak at certain hours
            hour = random.choices(range(24), weights=[
                1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 2, 2, 1, 1
            ])[0]
            minute = random.randint(0, 59)
        else:
            # Default: random time
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
        
        return datetime.combine(random_date, datetime.min.time().replace(hour=hour, minute=minute))
    
    def _select_transaction_parties(self, tx_type: str) -> Tuple[Dict, Dict]:
        """Select sender and receiver based on transaction type."""
        
        if tx_type == 'salary_payment':
            # Business pays customer
            sender = random.choice(self.businesses)
            receiver = random.choice(self.customers)
            return (
                {'id': sender['business_id'], 'name': sender['name'], 'type': 'business'},
                {'id': receiver['customer_id'], 'name': receiver['name'], 'type': 'customer'}
            )
        
        elif tx_type == 'retail_purchase':
            # Customer pays business
            sender = random.choice(self.customers)
            receiver = random.choice(self.businesses)
            return (
                {'id': sender['customer_id'], 'name': sender['name'], 'type': 'customer'},
                {'id': receiver['business_id'], 'name': receiver['name'], 'type': 'business'}
            )
        
        elif tx_type == 'p2p_transfer':
            # Customer to customer
            sender = random.choice(self.customers)
            receiver = random.choice(self.customers)
            while receiver['customer_id'] == sender['customer_id']:
                receiver = random.choice(self.customers)
            return (
                {'id': sender['customer_id'], 'name': sender['name'], 'type': 'customer'},
                {'id': receiver['customer_id'], 'name': receiver['name'], 'type': 'customer'}
            )
        
        elif tx_type == 'business_payment':
            # Business to business
            sender = random.choice(self.businesses)
            receiver = random.choice(self.businesses)
            while receiver['business_id'] == sender['business_id']:
                receiver = random.choice(self.businesses)
            return (
                {'id': sender['business_id'], 'name': sender['name'], 'type': 'business'},
                {'id': receiver['business_id'], 'name': receiver['name'], 'type': 'business'}
            )
        
        else:  # bill_payment or default
            # Customer pays business
            sender = random.choice(self.customers)
            receiver = random.choice(self.businesses)
            return (
                {'id': sender['customer_id'], 'name': sender['name'], 'type': 'customer'},
                {'id': receiver['business_id'], 'name': receiver['name'], 'type': 'business'}
            )
    
    def _generate_transaction_description(self, tx_type: str) -> str:
        """Generate realistic transaction descriptions."""
        
        descriptions = {
            'retail_purchase': [
                'Online purchase', 'Store purchase', 'Grocery shopping', 
                'Gas station', 'Restaurant bill', 'Pharmacy'
            ],
            'salary_payment': [
                'Salary payment', 'Payroll deposit', 'Monthly salary', 
                'Bi-weekly pay', 'Bonus payment'
            ],
            'bill_payment': [
                'Utility bill', 'Phone bill', 'Internet bill', 
                'Insurance payment', 'Loan payment', 'Credit card payment'
            ],
            'p2p_transfer': [
                'Transfer to friend', 'Family support', 'Shared expense', 
                'Loan repayment', 'Gift', 'Rent split'
            ],
            'business_payment': [
                'Vendor payment', 'Service fee', 'Consulting fee', 
                'Equipment purchase', 'Office supplies', 'Marketing expense'
            ]
        }
        
        return random.choice(descriptions.get(tx_type, ['Payment']))
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis."""
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)
        
        # Amount-based features
        df['amount_rounded'] = df['amount'].round(-2)  # Round to nearest 100
        df['is_round_amount'] = df['amount'] % 100 == 0
        df['amount_category'] = pd.cut(df['amount'], 
                                     bins=[0, 100, 1000, 10000, float('inf')],
                                     labels=['small', 'medium', 'large', 'very_large'])
        
        # Frequency features (will be calculated per entity)
        df = df.sort_values('timestamp')
        
        return df
    
    def save_data(self, output_dir: str = "data/generated/"):
        """Save generated data to files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save customers
        if self.customers:
            pd.DataFrame(self.customers).to_csv(f"{output_dir}/customers.csv", index=False)
        
        # Save businesses
        if self.businesses:
            pd.DataFrame(self.businesses).to_csv(f"{output_dir}/businesses.csv", index=False)
        
        # Save transactions
        if isinstance(self.transactions, pd.DataFrame) and not self.transactions.empty:
            self.transactions.to_csv(f"{output_dir}/normal_transactions.csv", index=False)
        
        print(f"Data saved to {output_dir}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of generated data."""
        if isinstance(self.transactions, pd.DataFrame) and not self.transactions.empty:
            return {
                'total_transactions': len(self.transactions),
                'total_customers': len(self.customers),
                'total_businesses': len(self.businesses),
                'date_range': {
                    'start': self.transactions['timestamp'].min(),
                    'end': self.transactions['timestamp'].max()
                },
                'amount_stats': {
                    'total_volume': self.transactions['amount'].sum(),
                    'avg_amount': self.transactions['amount'].mean(),
                    'median_amount': self.transactions['amount'].median(),
                    'max_amount': self.transactions['amount'].max()
                },
                'transaction_types': self.transactions['transaction_type'].value_counts().to_dict()
            }
        return {}


def main():
    """Main function to generate sample data."""
    generator = TransactionGenerator()
    
    print("Generating synthetic financial data...")
    print("1. Creating customer profiles...")
    customers = generator.generate_customers()
    print(f"   Generated {len(customers)} customers")
    
    print("2. Creating business profiles...")
    businesses = generator.generate_businesses()
    print(f"   Generated {len(businesses)} businesses")
    
    print("3. Generating transactions...")
    transactions = generator.generate_transactions()
    print(f"   Generated {len(transactions)} transactions")
    
    print("4. Saving data...")
    generator.save_data()
    
    print("5. Summary statistics:")
    stats = generator.get_summary_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nSynthetic data generation complete!")


if __name__ == "__main__":
    main() 