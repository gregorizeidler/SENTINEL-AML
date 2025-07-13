#!/usr/bin/env python3
"""
AML-FT Adversarial Simulation Demo

This script demonstrates the complete Red Team vs Blue Team simulation:
1. Red Team creates and executes criminal plans
2. Blue Team analyzes the data and detects suspicious activities
3. Shows the adversarial dynamics between the teams

Run this script to see the full simulation in action!
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import our agents
from data.transaction_generator import TransactionGenerator
from agents.red_team.mastermind_agent import MastermindAgent
from agents.red_team.operator_agent import OperatorAgent
from agents.blue_team.transaction_analyst import TransactionAnalyst

import pandas as pd
import numpy as np


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}")


def print_step(step: str, description: str):
    """Print a formatted step."""
    print(f"\n{step} {description}")
    print("-" * 50)


def simulate_pause(seconds: float = 1.0):
    """Simulate processing time for dramatic effect."""
    time.sleep(seconds)


def main():
    """Main demonstration function."""
    print_header("🎯 AML-FT ADVERSARIAL SIMULATION")
    print("🔴 Red Team (Criminals) vs 🔵 Blue Team (Investigators)")
    print("An AI-powered financial crime detection simulation")
    
    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # ==========================================
    # PHASE 1: SETUP THE BATTLEFIELD
    # ==========================================
    print_step("1️⃣", "SETTING UP THE BATTLEFIELD")
    
    print("🏦 Generating realistic financial ecosystem...")
    generator = TransactionGenerator()
    
    # Generate entities
    customers = generator.generate_customers()
    businesses = generator.generate_businesses()
    print(f"   ✅ Created {len(customers)} customers")
    print(f"   ✅ Created {len(businesses)} businesses")
    
    # Generate normal transactions
    print("💳 Generating normal transaction flow...")
    normal_transactions = generator.generate_transactions()
    print(f"   ✅ Generated {len(normal_transactions):,} normal transactions")
    print(f"   ✅ Total volume: ${normal_transactions['amount'].sum():,.2f}")
    
    simulate_pause(1)
    
    # ==========================================
    # PHASE 2: RED TEAM ATTACK
    # ==========================================
    print_step("2️⃣", "🔴 RED TEAM LAUNCHES ATTACK")
    
    # Initialize Red Team
    print("🧠 Mastermind Agent planning criminal operation...")
    mastermind = MastermindAgent()
    
    # Create criminal plan
    target_amount = 500000  # $500K to launder
    print(f"   🎯 Target: ${target_amount:,.2f} to launder")
    
    criminal_plan = mastermind.create_laundering_plan(
        target_amount=target_amount,
        complexity_level='medium',
        time_constraint=21
    )
    
    print(f"   ✅ Criminal plan created: {criminal_plan.get('plan_id')}")
    print(f"   ✅ Techniques: {', '.join(criminal_plan.get('techniques_used', []))}")
    print(f"   ✅ Risk level: {criminal_plan.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
    
    simulate_pause(1.5)
    
    # Execute criminal plan
    print("⚡ Operator Agents executing the plan...")
    operator = OperatorAgent(customers=customers, businesses=businesses)
    
    execution_result = operator.execute_plan(criminal_plan, normal_transactions)
    
    if execution_result['success']:
        criminal_transactions = execution_result['criminal_transactions']
        print(f"   ✅ Plan executed successfully!")
        print(f"   ✅ Generated {len(criminal_transactions)} criminal transactions")
        print(f"   ✅ Created {sum(len(entities) for entities in execution_result.get('entities_created', {}).values())} criminal entities")
        
        # Combine datasets
        print("🔀 Mixing criminal transactions with normal flow...")
        normal_transactions['is_criminal'] = False
        criminal_transactions['is_criminal'] = True
        
        # Ensure compatible columns
        common_columns = list(set(normal_transactions.columns) & set(criminal_transactions.columns))
        
        combined_data = pd.concat([
            normal_transactions[common_columns],
            criminal_transactions[common_columns]
        ], ignore_index=True)
        
        # Shuffle to hide the criminal transactions
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✅ Combined dataset: {len(combined_data):,} total transactions")
        print(f"   ✅ Criminal percentage: {len(criminal_transactions)/len(combined_data)*100:.2f}%")
        
    else:
        print("   ❌ Criminal plan execution failed!")
        print(f"   Error: {execution_result.get('error', 'Unknown error')}")
        return False
    
    simulate_pause(2)
    
    # ==========================================
    # PHASE 3: BLUE TEAM DEFENSE
    # ==========================================
    print_step("3️⃣", "🔵 BLUE TEAM LAUNCHES INVESTIGATION")
    
    # Initialize Blue Team
    print("🔍 Transaction Analyst beginning investigation...")
    analyst = TransactionAnalyst()
    
    # Remove the 'is_criminal' column to simulate real-world scenario
    investigation_data = combined_data.drop('is_criminal', axis=1)
    
    # Perform analysis
    print("   🔬 Running comprehensive analysis...")
    analysis_results = analyst.analyze_transactions(investigation_data)
    
    simulate_pause(2)
    
    # ==========================================
    # PHASE 4: BATTLE RESULTS
    # ==========================================
    print_step("4️⃣", "⚔️ BATTLE RESULTS")
    
    # Evaluate Blue Team performance
    print("📊 Evaluating Blue Team detection performance...")
    
    # Get suspicious entities identified by Blue Team
    detected_entities = [entity['entity_id'] for entity in analysis_results.get('suspicious_entities', [])]
    
    # Get actual criminal entities from Red Team
    actual_criminal_entities = set()
    for entity_type, entities in execution_result.get('entities_created', {}).items():
        for entity in entities:
            actual_criminal_entities.add(entity['entity_id'])
    
    # Also add entities involved in criminal transactions
    criminal_senders = set(criminal_transactions['sender_id'].unique())
    criminal_receivers = set(criminal_transactions['receiver_id'].unique())
    actual_criminal_entities.update(criminal_senders)
    actual_criminal_entities.update(criminal_receivers)
    
    # Calculate performance metrics
    detected_set = set(detected_entities)
    actual_set = actual_criminal_entities
    
    true_positives = len(detected_set & actual_set)
    false_positives = len(detected_set - actual_set)
    false_negatives = len(actual_set - detected_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   🎯 Detection Performance:")
    print(f"      • Precision: {precision:.2%} ({true_positives}/{true_positives + false_positives})")
    print(f"      • Recall: {recall:.2%} ({true_positives}/{true_positives + false_negatives})")
    print(f"      • F1-Score: {f1_score:.2%}")
    print(f"      • Criminal entities detected: {true_positives}/{len(actual_set)}")
    
    # Show analysis summary
    print("\n📋 Blue Team Analysis Summary:")
    print(analyst.get_analysis_summary())
    
    # ==========================================
    # PHASE 5: SAVE RESULTS
    # ==========================================
    print_step("5️⃣", "💾 SAVING SIMULATION RESULTS")
    
    # Save all results
    results_summary = {
        'simulation_timestamp': datetime.now().isoformat(),
        'red_team': {
            'plan_id': criminal_plan.get('plan_id'),
            'target_amount': target_amount,
            'techniques_used': criminal_plan.get('techniques_used', []),
            'criminal_transactions_count': len(criminal_transactions),
            'entities_created': len(actual_criminal_entities),
            'execution_success': execution_result['success']
        },
        'blue_team': {
            'analysis_methods': analysis_results.get('analysis_methods', []),
            'suspicious_entities_detected': len(detected_entities),
            'overall_risk_level': analysis_results.get('risk_assessment', {}).get('overall_risk_level', 'unknown'),
            'anomalies_detected': analysis_results.get('anomalies_detected', {}).get('count', 0)
        },
        'battle_results': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'dataset_stats': {
            'total_transactions': len(combined_data),
            'normal_transactions': len(normal_transactions),
            'criminal_transactions': len(criminal_transactions),
            'criminal_percentage': len(criminal_transactions)/len(combined_data)*100
        }
    }
    
    # Save files
    with open(output_dir / 'simulation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    with open(output_dir / 'criminal_plan.json', 'w') as f:
        json.dump(criminal_plan, f, indent=2, default=str)
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Save datasets
    combined_data.to_csv(output_dir / 'combined_transactions.csv', index=False)
    criminal_transactions.to_csv(output_dir / 'criminal_transactions.csv', index=False)
    normal_transactions.to_csv(output_dir / 'normal_transactions.csv', index=False)
    
    print(f"   ✅ All results saved to '{output_dir}' directory")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print_header("🏆 SIMULATION COMPLETE")
    
    # Determine winner
    if f1_score > 0.8:
        winner = "🔵 BLUE TEAM WINS!"
        outcome = "Excellent detection - Criminal network exposed!"
    elif f1_score > 0.6:
        winner = "🔵 BLUE TEAM WINS!"
        outcome = "Good detection - Most criminals caught!"
    elif f1_score > 0.4:
        winner = "⚖️ DRAW!"
        outcome = "Partial detection - Some criminals escaped!"
    else:
        winner = "🔴 RED TEAM WINS!"
        outcome = "Poor detection - Criminals mostly undetected!"
    
    print(f"\n{winner}")
    print(f"{outcome}")
    
    print(f"\n📈 SIMULATION METRICS:")
    print(f"   • Red Team Sophistication: {criminal_plan.get('complexity_level', 'Unknown')}")
    print(f"   • Blue Team Detection Rate: {recall:.1%}")
    print(f"   • Investigation Accuracy: {precision:.1%}")
    print(f"   • Overall Performance: {f1_score:.1%}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    
    # Red Team insights
    techniques_used = criminal_plan.get('techniques_used', [])
    print(f"   • Red Team used {len(techniques_used)} techniques: {', '.join(techniques_used)}")
    print(f"   • Created {len(actual_criminal_entities)} criminal entities")
    print(f"   • Injected {len(criminal_transactions)} criminal transactions")
    
    # Blue Team insights
    methods_used = analysis_results.get('analysis_methods', [])
    print(f"   • Blue Team used {len(methods_used)} analysis methods")
    print(f"   • Detected {len(detected_entities)} suspicious entities")
    print(f"   • Found {analysis_results.get('anomalies_detected', {}).get('count', 0)} anomalous transactions")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review detailed results in '{output_dir}' directory")
    print(f"   2. Analyze false positives and false negatives")
    print(f"   3. Improve detection algorithms based on findings")
    print(f"   4. Run simulation with different parameters")
    
    print(f"\n🚀 ADVANCED FEATURES TO EXPLORE:")
    print(f"   • Run with different complexity levels")
    print(f"   • Try different laundering techniques")
    print(f"   • Implement adaptive learning between rounds")
    print(f"   • Add OSINT and report generation agents")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Simulation completed successfully!")
            print("This demonstrates the core adversarial dynamics of the AML-FT system.")
        else:
            print("\n💥 Simulation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1) 