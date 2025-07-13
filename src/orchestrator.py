"""
AML-FT Adversarial Simulation Orchestrator

This module orchestrates the complete adversarial simulation system,
managing multiple rounds of Red Team vs Blue Team battles with
adaptive learning and continuous improvement.
"""

import json
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

# Import our agents and systems
from data.transaction_generator import TransactionGenerator
from agents.red_team.mastermind_agent import MastermindAgent
from agents.red_team.operator_agent import OperatorAgent
from agents.blue_team.transaction_analyst import TransactionAnalyst
from agents.blue_team.osint_agent import OSINTAgent
from agents.blue_team.lead_investigator import LeadInvestigator
from agents.blue_team.report_writer import ReportWriter
from adaptive_learning import AdaptiveLearningSystem

import pandas as pd
import numpy as np


@dataclass
class BattleResult:
    """Data class for battle results."""
    round_id: str
    timestamp: datetime
    winner: str  # 'red_team', 'blue_team', 'draw'
    score: float  # F1-score
    red_team_score: float
    blue_team_score: float
    performance_metrics: Dict[str, Any]
    battle_summary: str


@dataclass
class TournamentResult:
    """Data class for tournament results."""
    tournament_id: str
    start_time: datetime
    end_time: datetime
    total_rounds: int
    red_team_wins: int
    blue_team_wins: int
    draws: int
    average_f1_score: float
    improvement_rate: float
    final_adaptations: Dict[str, Any]


class SimulationOrchestrator:
    """
    Orchestrates the complete AML-FT adversarial simulation system.
    
    This class manages multiple rounds of simulation, tracks performance,
    applies adaptive learning, and provides comprehensive analysis of
    the Red Team vs Blue Team dynamics.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the simulation orchestrator."""
        self.config = self._load_config(config_path)
        self.adaptive_learning = AdaptiveLearningSystem(config_path)
        self.battle_history = []
        self.current_config = self.config.copy()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.transaction_generator = None
        self.red_team_agents = {}
        self.blue_team_agents = {}
        
        # Performance tracking
        self.performance_tracker = {
            'rounds_completed': 0,
            'total_improvement': 0.0,
            'best_f1_score': 0.0,
            'adaptation_count': 0
        }
        
        self.logger.info("Simulation Orchestrator initialized")
    
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
            'orchestrator': {
                'max_rounds': 10,
                'adaptation_frequency': 3,
                'early_stopping_threshold': 0.95,
                'min_improvement_threshold': 0.05,
                'battle_timeout_minutes': 30
            },
            'tournament': {
                'enable_tournaments': True,
                'tournament_size': 5,
                'championship_threshold': 0.8
            }
        }
    
    def _setup_logging(self):
        """Setup logging for the orchestrator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "orchestrator.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("SimulationOrchestrator")
    
    def run_single_battle(self, round_config: Optional[Dict] = None) -> BattleResult:
        """
        Run a single Red Team vs Blue Team battle.
        
        Args:
            round_config: Optional configuration override for this round
            
        Returns:
            BattleResult containing the outcome of the battle
        """
        round_id = f"BATTLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting battle: {round_id}")
        
        # Use provided config or current adaptive config
        battle_config = round_config or self.current_config
        
        start_time = datetime.now()
        
        try:
            # Phase 1: Setup Battlefield
            self.logger.info("Phase 1: Setting up battlefield...")
            normal_data = self._setup_battlefield(battle_config)
            
            # Phase 2: Red Team Attack
            self.logger.info("Phase 2: Red Team launching attack...")
            red_team_results = self._execute_red_team_attack(battle_config, normal_data)
            
            if not red_team_results['success']:
                raise Exception("Red Team attack failed")
            
            # Phase 3: Blue Team Defense
            self.logger.info("Phase 3: Blue Team launching defense...")
            blue_team_results = self._execute_blue_team_defense(
                battle_config, red_team_results, normal_data
            )
            
            # Phase 4: Evaluate Battle
            self.logger.info("Phase 4: Evaluating battle results...")
            battle_result = self._evaluate_battle(
                round_id, battle_config, red_team_results, blue_team_results, start_time
            )
            
            # Record for adaptive learning
            self.adaptive_learning.record_simulation_round(
                battle_config,
                red_team_results,
                blue_team_results,
                battle_result.performance_metrics
            )
            
            # Store battle result
            self.battle_history.append(battle_result)
            self.performance_tracker['rounds_completed'] += 1
            
            # Update best score
            if battle_result.score > self.performance_tracker['best_f1_score']:
                self.performance_tracker['best_f1_score'] = battle_result.score
            
            self.logger.info(f"Battle completed: {battle_result.winner} wins with F1={battle_result.score:.3f}")
            
            return battle_result
            
        except Exception as e:
            self.logger.error(f"Battle failed: {str(e)}")
            
            # Create failed battle result
            return BattleResult(
                round_id=round_id,
                timestamp=start_time,
                winner='error',
                score=0.0,
                red_team_score=0.0,
                blue_team_score=0.0,
                performance_metrics={'error': str(e)},
                battle_summary=f"Battle failed: {str(e)}"
            )
    
    def _setup_battlefield(self, config: Dict) -> Dict[str, Any]:
        """Setup the battlefield with normal transaction data."""
        if not self.transaction_generator:
            self.transaction_generator = TransactionGenerator(config_path="config/config.yaml")
        
        # Generate entities
        customers = self.transaction_generator.generate_customers()
        businesses = self.transaction_generator.generate_businesses()
        
        # Generate normal transactions
        normal_transactions = self.transaction_generator.generate_transactions()
        
        return {
            'customers': customers,
            'businesses': businesses,
            'normal_transactions': normal_transactions
        }
    
    def _execute_red_team_attack(self, config: Dict, normal_data: Dict) -> Dict[str, Any]:
        """Execute Red Team attack phase."""
        # Initialize Red Team agents
        mastermind = MastermindAgent(config_path="config/config.yaml")
        operator = OperatorAgent(
            customers=normal_data['customers'],
            businesses=normal_data['businesses']
        )
        
        # Get attack parameters
        red_team_config = config.get('simulation', {}).get('red_team', {})
        target_amount = red_team_config.get('target_amount', 500000)
        complexity = red_team_config.get('complexity_level', 'medium')
        
        # Create criminal plan
        criminal_plan = mastermind.create_laundering_plan(
            target_amount=target_amount,
            complexity_level=complexity,
            time_constraint=21
        )
        
        # Execute plan
        execution_result = operator.execute_plan(criminal_plan, normal_data['normal_transactions'])
        
        return {
            'success': execution_result['success'],
            'criminal_plan': criminal_plan,
            'execution_result': execution_result,
            'criminal_transactions': execution_result.get('criminal_transactions', pd.DataFrame()),
            'entities_created': execution_result.get('entities_created', {})
        }
    
    def _execute_blue_team_defense(self, config: Dict, red_team_results: Dict, normal_data: Dict) -> Dict[str, Any]:
        """Execute Blue Team defense phase."""
        # Combine normal and criminal transactions
        normal_transactions = normal_data['normal_transactions'].copy()
        criminal_transactions = red_team_results['criminal_transactions']
        
        if criminal_transactions.empty:
            raise Exception("No criminal transactions to analyze")
        
        # Add flags
        normal_transactions['is_criminal'] = False
        criminal_transactions['is_criminal'] = True
        
        # Find common columns
        common_columns = list(set(normal_transactions.columns) & set(criminal_transactions.columns))
        
        # Combine datasets
        combined_data = pd.concat([
            normal_transactions[common_columns],
            criminal_transactions[common_columns]
        ], ignore_index=True)
        
        # Shuffle data
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Remove criminal flag for analysis (simulate real-world scenario)
        investigation_data = combined_data.drop('is_criminal', axis=1)
        
        # Initialize Blue Team agents
        analyst = TransactionAnalyst()
        
        # Transaction analysis
        analysis_results = analyst.analyze_transactions(investigation_data)
        
        # OSINT analysis (if enabled)
        osint_results = {}
        blue_team_config = config.get('simulation', {}).get('blue_team', {})
        
        if blue_team_config.get('enable_osint', True):
            osint_agent = OSINTAgent()
            osint_results = osint_agent.investigate_entities(
                analysis_results.get('suspicious_entities', [])
            )
        
        # Lead investigation
        narratives = []
        if blue_team_config.get('investigation_depth', 'thorough') in ['standard', 'thorough']:
            investigator = LeadInvestigator()
            narratives = investigator.investigate_case(analysis_results, osint_results)
        
        # Report generation (if enabled)
        reports = []
        if blue_team_config.get('enable_reports', True):
            report_writer = ReportWriter()
            reports = report_writer.generate_sar_reports(narratives)
        
        return {
            'combined_data': combined_data,
            'analysis_results': analysis_results,
            'osint_results': osint_results,
            'narratives': narratives,
            'reports': reports
        }
    
    def _evaluate_battle(self, round_id: str, config: Dict, red_team_results: Dict, 
                        blue_team_results: Dict, start_time: datetime) -> BattleResult:
        """Evaluate the battle results and determine winner."""
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            blue_team_results['combined_data'],
            blue_team_results['analysis_results'],
            red_team_results
        )
        
        f1_score = performance_metrics['f1_score']
        precision = performance_metrics['precision']
        recall = performance_metrics['recall']
        
        # Determine winner
        if f1_score > 0.8:
            winner = 'blue_team'
            blue_team_score = f1_score
            red_team_score = 1.0 - f1_score
        elif f1_score > 0.6:
            winner = 'blue_team'
            blue_team_score = f1_score
            red_team_score = 1.0 - f1_score
        elif f1_score > 0.4:
            winner = 'draw'
            blue_team_score = f1_score
            red_team_score = 1.0 - f1_score
        else:
            winner = 'red_team'
            red_team_score = 1.0 - f1_score
            blue_team_score = f1_score
        
        # Generate battle summary
        battle_summary = self._generate_battle_summary(
            winner, performance_metrics, red_team_results, blue_team_results
        )
        
        return BattleResult(
            round_id=round_id,
            timestamp=start_time,
            winner=winner,
            score=f1_score,
            red_team_score=red_team_score,
            blue_team_score=blue_team_score,
            performance_metrics=performance_metrics,
            battle_summary=battle_summary
        )
    
    def _calculate_performance_metrics(self, combined_data: pd.DataFrame, 
                                     analysis_results: Dict, red_team_results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        # Get detected entities
        detected_entities = set(
            entity['entity_id'] for entity in analysis_results.get('suspicious_entities', [])
        )
        
        # Get actual criminal entities
        actual_criminal_entities = set()
        
        # From created entities
        for entity_type, entities in red_team_results.get('entities_created', {}).items():
            for entity in entities:
                actual_criminal_entities.add(entity['entity_id'])
        
        # From criminal transactions
        criminal_data = combined_data[combined_data['is_criminal'] == True]
        actual_criminal_entities.update(criminal_data['sender_id'].unique())
        actual_criminal_entities.update(criminal_data['receiver_id'].unique())
        
        # Calculate confusion matrix
        true_positives = len(detected_entities & actual_criminal_entities)
        false_positives = len(detected_entities - actual_criminal_entities)
        false_negatives = len(actual_criminal_entities - detected_entities)
        true_negatives = len(combined_data) - true_positives - false_positives - false_negatives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        accuracy = (true_positives + true_negatives) / len(combined_data) if len(combined_data) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'detected_entities': len(detected_entities),
            'actual_criminal_entities': len(actual_criminal_entities),
            'total_transactions': len(combined_data),
            'criminal_transactions': len(criminal_data),
            'criminal_percentage': len(criminal_data) / len(combined_data) * 100
        }
    
    def _generate_battle_summary(self, winner: str, metrics: Dict, 
                               red_team_results: Dict, blue_team_results: Dict) -> str:
        """Generate a human-readable battle summary."""
        summary = f"🏆 **{winner.replace('_', ' ').title()} Victory!**\n\n"
        
        # Performance overview
        summary += f"**Performance Metrics:**\n"
        summary += f"- Precision: {metrics['precision']:.2%}\n"
        summary += f"- Recall: {metrics['recall']:.2%}\n"
        summary += f"- F1-Score: {metrics['f1_score']:.2%}\n"
        summary += f"- Accuracy: {metrics['accuracy']:.2%}\n\n"
        
        # Red Team summary
        criminal_plan = red_team_results.get('criminal_plan', {})
        summary += f"**Red Team Performance:**\n"
        summary += f"- Techniques Used: {', '.join(criminal_plan.get('techniques_used', []))}\n"
        summary += f"- Complexity Level: {criminal_plan.get('complexity_level', 'Unknown')}\n"
        summary += f"- Entities Created: {len(red_team_results.get('entities_created', {}))}\n"
        summary += f"- Criminal Transactions: {metrics['criminal_transactions']}\n\n"
        
        # Blue Team summary
        analysis_results = blue_team_results.get('analysis_results', {})
        summary += f"**Blue Team Performance:**\n"
        summary += f"- Entities Detected: {metrics['detected_entities']}\n"
        summary += f"- Detection Accuracy: {metrics['true_positives']}/{metrics['actual_criminal_entities']} criminals found\n"
        summary += f"- Analysis Methods: {len(analysis_results.get('analysis_methods', []))}\n"
        summary += f"- Reports Generated: {len(blue_team_results.get('reports', []))}\n"
        
        return summary
    
    def run_adaptive_tournament(self, num_rounds: int = 5) -> TournamentResult:
        """
        Run an adaptive tournament with multiple rounds.
        
        Args:
            num_rounds: Number of rounds to run
            
        Returns:
            TournamentResult containing tournament statistics
        """
        tournament_id = f"TOURNAMENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting adaptive tournament: {tournament_id} ({num_rounds} rounds)")
        
        start_time = datetime.now()
        
        # Tournament tracking
        red_team_wins = 0
        blue_team_wins = 0
        draws = 0
        f1_scores = []
        
        initial_f1 = 0.0
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"Tournament Round {round_num}/{num_rounds}")
            
            # Apply adaptive learning every few rounds
            if round_num > 1 and round_num % self.config['orchestrator']['adaptation_frequency'] == 0:
                self.logger.info("Applying adaptive learning...")
                self._apply_adaptive_learning()
            
            # Run battle
            battle_result = self.run_single_battle()
            
            # Track results
            if battle_result.winner == 'red_team':
                red_team_wins += 1
            elif battle_result.winner == 'blue_team':
                blue_team_wins += 1
            else:
                draws += 1
            
            f1_scores.append(battle_result.score)
            
            if round_num == 1:
                initial_f1 = battle_result.score
            
            # Check for early stopping
            if battle_result.score > self.config['orchestrator']['early_stopping_threshold']:
                self.logger.info(f"Early stopping triggered at round {round_num}")
                break
            
            # Brief pause between rounds
            time.sleep(2)
        
        end_time = datetime.now()
        
        # Calculate improvement
        final_f1 = f1_scores[-1] if f1_scores else 0.0
        improvement_rate = (final_f1 - initial_f1) / initial_f1 if initial_f1 > 0 else 0.0
        
        # Get final adaptations
        final_adaptations = self.adaptive_learning.get_adaptive_recommendations()
        
        tournament_result = TournamentResult(
            tournament_id=tournament_id,
            start_time=start_time,
            end_time=end_time,
            total_rounds=len(f1_scores),
            red_team_wins=red_team_wins,
            blue_team_wins=blue_team_wins,
            draws=draws,
            average_f1_score=np.mean(f1_scores) if f1_scores else 0.0,
            improvement_rate=improvement_rate,
            final_adaptations=final_adaptations
        )
        
        self.logger.info(f"Tournament completed: {tournament_result.blue_team_wins} Blue wins, "
                        f"{tournament_result.red_team_wins} Red wins, {tournament_result.draws} draws")
        
        return tournament_result
    
    def _apply_adaptive_learning(self):
        """Apply adaptive learning to improve the simulation."""
        recommendations = self.adaptive_learning.get_adaptive_recommendations()
        
        if 'adaptations' in recommendations:
            # Apply configuration adaptations
            self.current_config = self.adaptive_learning.apply_adaptations(self.current_config)
            self.performance_tracker['adaptation_count'] += 1
            
            self.logger.info("Applied adaptive learning recommendations")
        else:
            self.logger.info("No adaptations available yet")
    
    def get_tournament_report(self, tournament_result: TournamentResult) -> str:
        """Generate comprehensive tournament report."""
        report = f"""
ADAPTIVE TOURNAMENT REPORT
==========================

Tournament ID: {tournament_result.tournament_id}
Duration: {tournament_result.start_time.strftime('%Y-%m-%d %H:%M')} - {tournament_result.end_time.strftime('%Y-%m-%d %H:%M')}
Total Time: {(tournament_result.end_time - tournament_result.start_time).total_seconds() / 60:.1f} minutes

RESULTS SUMMARY
===============
Total Rounds: {tournament_result.total_rounds}
🔵 Blue Team Wins: {tournament_result.blue_team_wins}
🔴 Red Team Wins: {tournament_result.red_team_wins}
⚖️ Draws: {tournament_result.draws}

PERFORMANCE METRICS
==================
Average F1-Score: {tournament_result.average_f1_score:.3f}
Improvement Rate: {tournament_result.improvement_rate:.1%}
Best F1-Score: {self.performance_tracker['best_f1_score']:.3f}

ADAPTIVE LEARNING
=================
Adaptations Applied: {self.performance_tracker['adaptation_count']}
Learning Insights: {len(self.adaptive_learning.learning_insights)}
"""
        
        # Add recent battle summaries
        if self.battle_history:
            report += f"\nRECENT BATTLES\n==============\n"
            for battle in self.battle_history[-3:]:  # Last 3 battles
                report += f"\n{battle.round_id}: {battle.winner} wins (F1: {battle.score:.3f})\n"
                report += f"  {battle.battle_summary.split('**Performance Metrics:**')[0].strip()}\n"
        
        # Add adaptation recommendations
        if tournament_result.final_adaptations:
            report += f"\nNEXT STEPS\n==========\n"
            adaptations = tournament_result.final_adaptations.get('adaptations', {})
            
            for category, category_adaptations in adaptations.items():
                if category_adaptations:
                    report += f"\n{category.replace('_', ' ').title()}:\n"
                    for adaptation in category_adaptations[:2]:  # Top 2 per category
                        report += f"  • {adaptation['strategy']}\n"
        
        return report
    
    def export_results(self, output_dir: str = "tournament_results/"):
        """Export all tournament results and data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export battle history
        battle_data = [asdict(battle) for battle in self.battle_history]
        with open(output_path / f"battle_history_{timestamp}.json", 'w') as f:
            json.dump(battle_data, f, indent=2, default=str)
        
        # Export performance tracker
        with open(output_path / f"performance_tracker_{timestamp}.json", 'w') as f:
            json.dump(self.performance_tracker, f, indent=2)
        
        # Export adaptive learning report
        learning_report = self.adaptive_learning.get_learning_report()
        with open(output_path / f"learning_report_{timestamp}.txt", 'w') as f:
            f.write(learning_report)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and statistics."""
        return {
            'rounds_completed': self.performance_tracker['rounds_completed'],
            'best_f1_score': self.performance_tracker['best_f1_score'],
            'adaptations_applied': self.performance_tracker['adaptation_count'],
            'battle_history_length': len(self.battle_history),
            'learning_insights': len(self.adaptive_learning.learning_insights),
            'current_config_hash': hash(str(self.current_config)),
            'last_battle': self.battle_history[-1].round_id if self.battle_history else None
        }


def main():
    """Main function for testing the Simulation Orchestrator."""
    print("🎯 Testing AML-FT Simulation Orchestrator")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = SimulationOrchestrator()
    
    # Run a small tournament
    print("\n🏆 Running Adaptive Tournament (3 rounds)...")
    tournament_result = orchestrator.run_adaptive_tournament(num_rounds=3)
    
    # Display results
    print("\n📊 Tournament Report:")
    print("=" * 60)
    report = orchestrator.get_tournament_report(tournament_result)
    print(report)
    
    # Export results
    print("\n💾 Exporting results...")
    orchestrator.export_results()
    
    # Show final status
    print("\n📈 Final Status:")
    print("=" * 60)
    status = orchestrator.get_orchestrator_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n🎉 Orchestrator test completed!")


if __name__ == "__main__":
    main() 