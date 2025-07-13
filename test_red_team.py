#!/usr/bin/env python3
"""
Test script for Red Team (Criminal Agents)

This script demonstrates how the Red Team works:
1. Generate normal transaction data
2. Create a criminal plan with MastermindAgent
3. Execute the plan with OperatorAgent
4. Show the combined dataset that the Blue Team will analyze
"""

import sys
import os
sys.path.append('src')

from data.transaction_generator import TransactionGenerator
from agents.red_team.mastermind_agent import MastermindAgent
from agents.red_team.operator_agent import OperatorAgent
import pandas as pd
import json


def main():
    print("🔴 RED TEAM ADVERSARIAL SIMULATION TEST")
    print("=" * 60)
    
    # Step 1: Generate normal transaction data
    print("\n1️⃣ GENERATING NORMAL TRANSACTION DATA")
    print("-" * 40)
    
    generator = TransactionGenerator()
    
    print("Creating customer and business profiles...")
    customers = generator.generate_customers()
    businesses = generator.generate_businesses()
    
    print("Generating normal transactions...")
    normal_transactions = generator.generate_transactions()
    
    print(f"✅ Generated {len(normal_transactions)} normal transactions")
    print(f"✅ Created {len(customers)} customers and {len(businesses)} businesses")
    
    # Step 2: Create criminal plan
    print("\n2️⃣ CREATING CRIMINAL PLAN (MASTERMIND AGENT)")
    print("-" * 40)
    
    mastermind = MastermindAgent()
    
    target_amount = 250000  # $250,000 to launder
    print(f"Target amount to launder: ${target_amount:,.2f}")
    
    print("Mastermind Agent creating sophisticated laundering plan...")
    criminal_plan = mastermind.create_laundering_plan(
        target_amount=target_amount,
        complexity_level='medium',
        time_constraint=21
    )
    
    print("✅ Criminal plan created successfully!")
    print(f"Plan ID: {criminal_plan.get('plan_id')}")
    print(f"Techniques: {', '.join(criminal_plan.get('techniques_used', []))}")
    print(f"Risk Level: {criminal_plan.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
    
    # Step 3: Execute criminal plan
    print("\n3️⃣ EXECUTING CRIMINAL PLAN (OPERATOR AGENT)")
    print("-" * 40)
    
    operator = OperatorAgent(customers=customers, businesses=businesses)
    
    print("Operator Agent executing the criminal plan...")
    execution_result = operator.execute_plan(criminal_plan, normal_transactions)
    
    if execution_result['success']:
        print("✅ Criminal plan executed successfully!")
        criminal_transactions = execution_result['criminal_transactions']
        print(f"Generated {len(criminal_transactions)} criminal transactions")
        
        # Step 4: Combine datasets
        print("\n4️⃣ COMBINING NORMAL AND CRIMINAL TRANSACTIONS")
        print("-" * 40)
        
        # Add a flag to distinguish transaction types
        normal_transactions['is_criminal'] = False
        criminal_transactions['is_criminal'] = True
        
        # Ensure both DataFrames have the same columns
        common_columns = set(normal_transactions.columns) & set(criminal_transactions.columns)
        
        # Create combined dataset
        combined_data = pd.concat([
            normal_transactions[list(common_columns)],
            criminal_transactions[list(common_columns)]
        ], ignore_index=True)
        
        # Shuffle the data to mix criminal and normal transactions
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        print(f"✅ Combined dataset created with {len(combined_data)} total transactions")
        print(f"   - Normal transactions: {len(normal_transactions)}")
        print(f"   - Criminal transactions: {len(criminal_transactions)}")
        print(f"   - Criminal percentage: {len(criminal_transactions)/len(combined_data)*100:.2f}%")
        
        # Step 5: Show sample data
        print("\n5️⃣ SAMPLE ANALYSIS")
        print("-" * 40)
        
        print("Sample of criminal transactions:")
        if not criminal_transactions.empty:
            sample_criminal = criminal_transactions[['transaction_id', 'timestamp', 'amount', 'criminal_technique', 'sender_name', 'receiver_name']].head(3)
            print(sample_criminal.to_string(index=False))
        
        print("\nTransaction amount distribution:")
        print(f"Normal transactions - Mean: ${normal_transactions['amount'].mean():.2f}, Median: ${normal_transactions['amount'].median():.2f}")
        print(f"Criminal transactions - Mean: ${criminal_transactions['amount'].mean():.2f}, Median: ${criminal_transactions['amount'].median():.2f}")
        
        print("\nRisk indicators in criminal transactions:")
        if 'risk_indicators' in criminal_transactions.columns:
            all_indicators = []
            for indicators in criminal_transactions['risk_indicators'].dropna():
                if isinstance(indicators, list):
                    all_indicators.extend(indicators)
            
            from collections import Counter
            indicator_counts = Counter(all_indicators)
            for indicator, count in indicator_counts.most_common(5):
                print(f"  - {indicator}: {count} times")
        
        # Step 6: Save results
        print("\n6️⃣ SAVING RESULTS")
        print("-" * 40)
        
        # Create output directory
        os.makedirs('data/test_results', exist_ok=True)
        
        # Save datasets
        normal_transactions.to_csv('data/test_results/normal_transactions.csv', index=False)
        criminal_transactions.to_csv('data/test_results/criminal_transactions.csv', index=False)
        combined_data.to_csv('data/test_results/combined_dataset.csv', index=False)
        
        # Save plan and execution results
        with open('data/test_results/criminal_plan.json', 'w') as f:
            json.dump(criminal_plan, f, indent=2, default=str)
        
        with open('data/test_results/execution_result.json', 'w') as f:
            # Remove DataFrame from execution result for JSON serialization
            result_copy = execution_result.copy()
            result_copy.pop('criminal_transactions', None)
            json.dump(result_copy, f, indent=2, default=str)
        
        print("✅ All results saved to 'data/test_results/' directory")
        
        # Step 7: Summary
        print("\n7️⃣ SUMMARY")
        print("-" * 40)
        
        print("🔴 RED TEAM SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"The Red Team has successfully:")
        print(f"  ✅ Created a sophisticated money laundering plan")
        print(f"  ✅ Generated {len(criminal_transactions)} criminal transactions")
        print(f"  ✅ Mixed them with {len(normal_transactions)} normal transactions")
        print(f"  ✅ Created a realistic dataset for Blue Team analysis")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. The Blue Team will analyze the combined dataset")
        print(f"  2. They must identify the {len(criminal_transactions)} criminal transactions")
        print(f"  3. They must reconstruct the criminal network and plan")
        print(f"  4. They must generate a professional investigation report")
        
        print(f"\n📊 CHALLENGE METRICS:")
        print(f"  - Detection difficulty: {criminal_plan.get('detection_risk_level', 'Unknown')}")
        print(f"  - Techniques used: {len(criminal_plan.get('techniques_used', []))}")
        print(f"  - Criminal entities created: {sum(len(entities) for entities in execution_result.get('entities_created', {}).values())}")
        print(f"  - Needle in haystack ratio: 1:{len(normal_transactions)//len(criminal_transactions)}")
        
    else:
        print("❌ Criminal plan execution failed!")
        print(f"Error: {execution_result.get('error', 'Unknown error')}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Red Team test completed successfully!")
        print("Run this script to generate test data for Blue Team development.")
    else:
        print("\n💥 Red Team test failed!")
        sys.exit(1) 