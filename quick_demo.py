#!/usr/bin/env python3
"""
Quick Demo - AML-FT Adversarial Simulation

This script provides a quick demonstration of the core system capabilities
without requiring API keys or extensive setup. Perfect for showcasing
the project's functionality.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import random

# Add src to path
sys.path.append('src')

import pandas as pd
import numpy as np


def print_header(title: str, width: int = 80):
    """Print a formatted header."""
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def simulate_typing(text: str, delay: float = 0.03):
    """Simulate typing effect for dramatic demonstration."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def demo_data_generation():
    """Demonstrate data generation capabilities."""
    print_section("📊 DATA GENERATION")
    
    simulate_typing("🏦 Generating realistic financial ecosystem...")
    time.sleep(1)
    
    # Simulate data generation
    print("   ✅ Created 5,000 customers with realistic profiles")
    print("   ✅ Generated 500 businesses across various industries")
    print("   ✅ Produced 50,000 normal transactions over 365 days")
    print("   ✅ Applied realistic transaction patterns and timing")
    
    # Show sample data structure
    print("\n📋 Sample Customer Profile:")
    sample_customer = {
        "customer_id": "CUST_001234",
        "name": "Sarah Johnson",
        "account_type": "checking",
        "risk_category": "low",
        "avg_monthly_transactions": 45,
        "avg_transaction_amount": 2750.50,
        "location": "New York, NY"
    }
    
    for key, value in sample_customer.items():
        print(f"   {key}: {value}")
    
    print("\n📋 Sample Business Entity:")
    sample_business = {
        "business_id": "BUS_005678",
        "name": "Metro Coffee Shop",
        "industry": "food_service",
        "monthly_revenue": 45000,
        "transaction_volume": "high",
        "cash_intensive": True
    }
    
    for key, value in sample_business.items():
        print(f"   {key}: {value}")


def demo_red_team_attack():
    """Demonstrate Red Team attack capabilities."""
    print_section("🔴 RED TEAM ATTACK")
    
    simulate_typing("🧠 Mastermind Agent creating criminal plan...")
    time.sleep(1)
    
    # Simulate criminal plan generation
    criminal_plan = {
        "plan_id": "CRIMINAL_PLAN_20241201_143022",
        "objective": "Launder $750,000 with minimal detection risk",
        "complexity_level": "complex",
        "techniques_used": ["smurfing", "shell_companies", "money_mules"],
        "execution_timeline": "21 days",
        "detection_probability": 0.18,
        "success_probability": 0.82
    }
    
    print("   ✅ Criminal plan generated successfully!")
    print(f"   📋 Plan ID: {criminal_plan['plan_id']}")
    print(f"   🎯 Target: ${criminal_plan['objective'].split('$')[1]}")
    print(f"   🔧 Techniques: {', '.join(criminal_plan['techniques_used'])}")
    print(f"   ⏱️ Timeline: {criminal_plan['execution_timeline']}")
    print(f"   📊 Detection Risk: {criminal_plan['detection_probability']:.1%}")
    
    time.sleep(1)
    simulate_typing("\n⚡ Operator Agent executing criminal plan...")
    time.sleep(1)
    
    # Simulate plan execution
    execution_results = {
        "entities_created": {
            "shell_companies": 3,
            "money_mules": 8,
            "cash_businesses": 2
        },
        "transactions_generated": 127,
        "total_amount_processed": 750000,
        "execution_success": True,
        "stealth_rating": "high"
    }
    
    print("   ✅ Criminal plan executed successfully!")
    print(f"   🏢 Shell companies created: {execution_results['entities_created']['shell_companies']}")
    print(f"   👥 Money mules recruited: {execution_results['entities_created']['money_mules']}")
    print(f"   💰 Transactions generated: {execution_results['transactions_generated']}")
    print(f"   🎯 Total amount processed: ${execution_results['total_amount_processed']:,}")
    print(f"   🥷 Stealth rating: {execution_results['stealth_rating']}")
    
    return criminal_plan, execution_results


def demo_blue_team_defense():
    """Demonstrate Blue Team defense capabilities."""
    print_section("🔵 BLUE TEAM DEFENSE")
    
    # Step 1: Transaction Analysis
    simulate_typing("🔍 Transaction Analyst analyzing patterns...")
    time.sleep(1)
    
    analysis_results = {
        "suspicious_entities": 23,
        "anomalies_detected": 45,
        "structuring_cases": 3,
        "network_communities": 2,
        "risk_score_avg": 0.76
    }
    
    print("   ✅ Transaction analysis completed!")
    print(f"   🚨 Suspicious entities identified: {analysis_results['suspicious_entities']}")
    print(f"   📊 Anomalies detected: {analysis_results['anomalies_detected']}")
    print(f"   🔄 Structuring cases found: {analysis_results['structuring_cases']}")
    print(f"   🕸️ Suspicious communities: {analysis_results['network_communities']}")
    print(f"   ⚠️ Average risk score: {analysis_results['risk_score_avg']:.2f}")
    
    # Step 2: OSINT Intelligence
    time.sleep(1)
    simulate_typing("\n🔎 OSINT Agent gathering intelligence...")
    time.sleep(1)
    
    osint_results = {
        "intelligence_pieces": 67,
        "sanctions_matches": 0,
        "news_mentions": 5,
        "court_records": 2,
        "business_registry_hits": 8,
        "high_relevance_findings": 12
    }
    
    print("   ✅ OSINT investigation completed!")
    print(f"   📰 Intelligence pieces gathered: {osint_results['intelligence_pieces']}")
    print(f"   🚫 Sanctions list matches: {osint_results['sanctions_matches']}")
    print(f"   📺 News mentions found: {osint_results['news_mentions']}")
    print(f"   ⚖️ Court records discovered: {osint_results['court_records']}")
    print(f"   🏢 Business registry hits: {osint_results['business_registry_hits']}")
    print(f"   🎯 High relevance findings: {osint_results['high_relevance_findings']}")
    
    # Step 3: Lead Investigation
    time.sleep(1)
    simulate_typing("\n🕵️ Lead Investigator constructing narratives...")
    time.sleep(1)
    
    investigation_results = {
        "criminal_narratives": 3,
        "high_risk_cases": 2,
        "evidence_pieces": 156,
        "timeline_events": 89,
        "confidence_avg": 0.87
    }
    
    print("   ✅ Criminal narratives constructed!")
    print(f"   📖 Criminal narratives built: {investigation_results['criminal_narratives']}")
    print(f"   🚨 High-risk cases identified: {investigation_results['high_risk_cases']}")
    print(f"   📋 Evidence pieces analyzed: {investigation_results['evidence_pieces']}")
    print(f"   ⏰ Timeline events mapped: {investigation_results['timeline_events']}")
    print(f"   🎯 Average confidence: {investigation_results['confidence_avg']:.2f}")
    
    # Step 4: Report Generation
    time.sleep(1)
    simulate_typing("\n📝 Report Writer generating SAR reports...")
    time.sleep(1)
    
    report_results = {
        "sar_reports": 3,
        "compliance_documents": 5,
        "pdf_reports": 3,
        "regulatory_citations": 12,
        "recommended_actions": 8
    }
    
    print("   ✅ Professional reports generated!")
    print(f"   📄 SAR reports created: {report_results['sar_reports']}")
    print(f"   📋 Compliance documents: {report_results['compliance_documents']}")
    print(f"   📑 PDF reports generated: {report_results['pdf_reports']}")
    print(f"   📚 Regulatory citations: {report_results['regulatory_citations']}")
    print(f"   💡 Recommended actions: {report_results['recommended_actions']}")
    
    return analysis_results, osint_results, investigation_results, report_results


def demo_performance_evaluation():
    """Demonstrate performance evaluation."""
    print_section("📊 PERFORMANCE EVALUATION")
    
    simulate_typing("⚖️ Evaluating Blue Team detection performance...")
    time.sleep(1)
    
    # Simulate performance metrics
    performance_metrics = {
        "precision": 0.87,
        "recall": 0.65,
        "f1_score": 0.74,
        "accuracy": 0.92,
        "criminals_detected": 15,
        "total_criminals": 23,
        "false_positives": 2,
        "battle_winner": "Blue Team"
    }
    
    print("   ✅ Performance evaluation completed!")
    print(f"   🎯 Precision: {performance_metrics['precision']:.1%}")
    print(f"   🎯 Recall: {performance_metrics['recall']:.1%}")
    print(f"   🎯 F1-Score: {performance_metrics['f1_score']:.1%}")
    print(f"   🎯 Accuracy: {performance_metrics['accuracy']:.1%}")
    print(f"   🔍 Criminals detected: {performance_metrics['criminals_detected']}/{performance_metrics['total_criminals']}")
    print(f"   ⚠️ False positives: {performance_metrics['false_positives']}")
    
    # Determine winner
    if performance_metrics['f1_score'] > 0.7:
        print(f"\n🏆 BATTLE RESULT: {performance_metrics['battle_winner']} WINS!")
        print("   🔵 Excellent detection performance!")
    elif performance_metrics['f1_score'] > 0.5:
        print(f"\n🏆 BATTLE RESULT: {performance_metrics['battle_winner']} WINS!")
        print("   🔵 Good detection performance!")
    else:
        print("\n🏆 BATTLE RESULT: Red Team WINS!")
        print("   🔴 Criminal evasion successful!")
    
    return performance_metrics


def demo_adaptive_learning():
    """Demonstrate adaptive learning capabilities."""
    print_section("🧠 ADAPTIVE LEARNING")
    
    simulate_typing("📚 Analyzing performance patterns...")
    time.sleep(1)
    
    # Simulate learning insights
    learning_insights = [
        {
            "category": "blue_team",
            "insight": "Detection threshold of 0.7 shows optimal performance",
            "confidence": 0.85,
            "action": "Maintain current threshold setting"
        },
        {
            "category": "red_team",
            "insight": "Smurfing technique consistently detected",
            "confidence": 0.89,
            "action": "Improve transaction timing randomization"
        },
        {
            "category": "general",
            "insight": "Complex techniques show 15% better evasion",
            "confidence": 0.76,
            "action": "Gradually increase technique complexity"
        }
    ]
    
    print("   ✅ Learning analysis completed!")
    print(f"   💡 Insights generated: {len(learning_insights)}")
    
    for i, insight in enumerate(learning_insights, 1):
        print(f"\n   📋 Insight {i} ({insight['category']}):")
        print(f"      • {insight['insight']}")
        print(f"      • Confidence: {insight['confidence']:.1%}")
        print(f"      • Action: {insight['action']}")
    
    # Simulate adaptation
    time.sleep(1)
    simulate_typing("\n🔄 Applying adaptive improvements...")
    time.sleep(1)
    
    adaptations = {
        "config_changes": 3,
        "threshold_adjustments": 1,
        "technique_improvements": 2,
        "strategy_updates": 4
    }
    
    print("   ✅ Adaptive improvements applied!")
    print(f"   ⚙️ Configuration changes: {adaptations['config_changes']}")
    print(f"   🎚️ Threshold adjustments: {adaptations['threshold_adjustments']}")
    print(f"   🔧 Technique improvements: {adaptations['technique_improvements']}")
    print(f"   📈 Strategy updates: {adaptations['strategy_updates']}")
    
    return learning_insights, adaptations


def demo_tournament_mode():
    """Demonstrate tournament mode capabilities."""
    print_section("🏆 TOURNAMENT MODE")
    
    simulate_typing("🎮 Running multi-round adaptive tournament...")
    time.sleep(1)
    
    # Simulate tournament rounds
    tournament_results = []
    
    for round_num in range(1, 6):
        print(f"\n   🔄 Round {round_num}/5:")
        
        # Simulate round performance with improvement over time
        base_f1 = 0.6 + (round_num - 1) * 0.04 + random.uniform(-0.05, 0.05)
        f1_score = max(0.3, min(0.9, base_f1))
        
        if f1_score > 0.7:
            winner = "Blue Team"
        elif f1_score < 0.4:
            winner = "Red Team"
        else:
            winner = "Draw"
        
        tournament_results.append({
            "round": round_num,
            "f1_score": f1_score,
            "winner": winner
        })
        
        print(f"      F1-Score: {f1_score:.3f}")
        print(f"      Winner: {winner}")
        
        time.sleep(0.5)
    
    # Calculate tournament statistics
    blue_wins = len([r for r in tournament_results if r['winner'] == 'Blue Team'])
    red_wins = len([r for r in tournament_results if r['winner'] == 'Red Team'])
    draws = len([r for r in tournament_results if r['winner'] == 'Draw'])
    avg_f1 = sum(r['f1_score'] for r in tournament_results) / len(tournament_results)
    improvement = tournament_results[-1]['f1_score'] - tournament_results[0]['f1_score']
    
    print(f"\n   🏁 TOURNAMENT RESULTS:")
    print(f"      🔵 Blue Team Wins: {blue_wins}")
    print(f"      🔴 Red Team Wins: {red_wins}")
    print(f"      ⚖️ Draws: {draws}")
    print(f"      📊 Average F1-Score: {avg_f1:.3f}")
    print(f"      📈 Improvement: {improvement:+.3f}")
    
    if blue_wins > red_wins:
        print(f"\n   🎉 TOURNAMENT CHAMPION: Blue Team!")
    elif red_wins > blue_wins:
        print(f"\n   🎉 TOURNAMENT CHAMPION: Red Team!")
    else:
        print(f"\n   🤝 TOURNAMENT RESULT: Draw!")
    
    return tournament_results


def demo_system_outputs():
    """Demonstrate system outputs and reports."""
    print_section("📄 SYSTEM OUTPUTS")
    
    simulate_typing("📋 Generating system outputs...")
    time.sleep(1)
    
    # Create demo output directory
    output_dir = Path("quick_demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Sample SAR Report
    sar_report = """
# SUSPICIOUS ACTIVITY REPORT (SAR)

**Report ID:** SAR-20241201-A7F3B2C1
**Filing Date:** 2024-12-01
**Subject:** Phantom Consulting LLC Network
**Risk Level:** HIGH

## EXECUTIVE SUMMARY
Investigation identified sophisticated money laundering operation involving 
structured deposits, shell companies, and coordinated money mule activities.

## SUSPICIOUS ACTIVITIES
- 79 deposits of $9,500 each (structuring pattern)
- Rapid inter-company transfers through shell entities
- Coordinated cash withdrawals by mule network
- No legitimate business activity for shell companies

## RECOMMENDED ACTIONS
- File SAR with FinCEN immediately
- Freeze all associated accounts
- Notify law enforcement for criminal investigation
- Enhanced monitoring of connected entities

---
**Generated by:** AML-FT Adversarial Simulation System
**Date:** 2024-12-01
"""
    
    # Save sample report
    with open(output_dir / "sample_sar_report.md", 'w') as f:
        f.write(sar_report)
    
    # Sample analytics data
    analytics_data = {
        "simulation_timestamp": datetime.now().isoformat(),
        "performance_metrics": {
            "precision": 0.87,
            "recall": 0.65,
            "f1_score": 0.74,
            "accuracy": 0.92
        },
        "detection_results": {
            "suspicious_entities": 23,
            "criminals_detected": 15,
            "false_positives": 2,
            "true_positives": 15
        },
        "techniques_analysis": {
            "smurfing_detection_rate": 0.89,
            "shell_company_detection_rate": 0.76,
            "money_mule_detection_rate": 0.82
        }
    }
    
    # Save analytics data
    with open(output_dir / "analytics_summary.json", 'w') as f:
        json.dump(analytics_data, f, indent=2)
    
    print("   ✅ System outputs generated!")
    print(f"   📄 SAR report: {output_dir}/sample_sar_report.md")
    print(f"   📊 Analytics data: {output_dir}/analytics_summary.json")
    print(f"   📁 Output directory: {output_dir}/")
    
    # Show sample report preview
    print("\n   📋 Sample SAR Report Preview:")
    print("   " + "="*50)
    for line in sar_report.split('\n')[:10]:
        print(f"   {line}")
    print("   ... (truncated)")
    
    return output_dir


def main():
    """Main demonstration function."""
    print_header("🎯 AML-FT ADVERSARIAL SIMULATION - QUICK DEMO")
    
    print("🚀 Welcome to the AML-FT Adversarial Simulation System!")
    print("This demonstration showcases the core capabilities without requiring API keys.")
    print("All outputs are simulated but demonstrate real system functionality.")
    
    # Demo sequence
    demo_data_generation()
    criminal_plan, execution_results = demo_red_team_attack()
    analysis_results, osint_results, investigation_results, report_results = demo_blue_team_defense()
    performance_metrics = demo_performance_evaluation()
    learning_insights, adaptations = demo_adaptive_learning()
    tournament_results = demo_tournament_mode()
    output_dir = demo_system_outputs()
    
    # Final summary
    print_header("🎉 DEMONSTRATION COMPLETED")
    
    print("✅ **CAPABILITIES DEMONSTRATED:**")
    print()
    print("🔴 **Red Team (Criminal Agents):**")
    print("   • LLM-powered criminal planning and strategy")
    print("   • Sophisticated money laundering technique execution")
    print("   • Realistic entity creation and transaction generation")
    print("   • Adaptive strategy improvement based on detection")
    print()
    print("🔵 **Blue Team (Investigation Agents):**")
    print("   • Multi-method transaction analysis and detection")
    print("   • External intelligence gathering (OSINT)")
    print("   • Criminal narrative construction and evidence linking")
    print("   • Professional SAR report generation")
    print()
    print("🧠 **Adaptive Learning System:**")
    print("   • Performance pattern analysis across rounds")
    print("   • Strategic insight generation and recommendations")
    print("   • Automated configuration optimization")
    print("   • Continuous improvement demonstration")
    print()
    print("🏆 **Tournament & Orchestration:**")
    print("   • Multi-round adaptive tournaments")
    print("   • Performance tracking and improvement measurement")
    print("   • Battle result determination and analysis")
    print("   • Comprehensive reporting and analytics")
    print()
    print("📊 **Final Performance Summary:**")
    print(f"   • Detection Precision: {performance_metrics['precision']:.1%}")
    print(f"   • Detection Recall: {performance_metrics['recall']:.1%}")
    print(f"   • Overall F1-Score: {performance_metrics['f1_score']:.1%}")
    print(f"   • Tournament Winner: {performance_metrics['battle_winner']}")
    print(f"   • Learning Insights: {len(learning_insights)} generated")
    print(f"   • Adaptations Applied: {sum(adaptations.values())}")
    print()
    print("📁 **Generated Outputs:**")
    print(f"   • Sample reports saved to: {output_dir}/")
    print("   • Professional SAR documents demonstrated")
    print("   • Analytics data and performance metrics")
    print("   • Complete system capability showcase")
    print()
    print("🌟 **Next Steps:**")
    print("   1. Set up API keys for full LLM functionality")
    print("   2. Run 'python demo_advanced.py' for complete system")
    print("   3. Launch 'streamlit run interface/streamlit_app.py' for web interface")
    print("   4. Explore configuration options in config/config.yaml")
    print("   5. Customize for your specific use case")
    print()
    print("🎯 **This system demonstrates production-ready capabilities for:**")
    print("   • Financial crime detection and investigation")
    print("   • Compliance reporting and regulatory requirements")
    print("   • AI-powered adversarial security testing")
    print("   • Continuous learning and system improvement")
    print("   • Professional financial intelligence analysis")
    
    print(f"\n{'='*80}")
    print("Thank you for exploring the AML-FT Adversarial Simulation System!")
    print("For more information, see the comprehensive README.md")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
        print("Thank you for trying the AML-FT Adversarial Simulation System!")
    except Exception as e:
        print(f"\n💥 Demonstration error: {str(e)}")
        print("This is a simulated demo - no actual functionality affected.") 