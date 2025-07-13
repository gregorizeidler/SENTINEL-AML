"""
AML-FT Adversarial Simulation - Streamlit Interface

Interactive web interface for the AML-FT adversarial simulation system.
Allows users to configure, run, and analyze simulations through a web browser.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import networkx as nx
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our agents
from data.transaction_generator import TransactionGenerator
from agents.red_team.mastermind_agent import MastermindAgent
from agents.red_team.operator_agent import OperatorAgent
from agents.blue_team.transaction_analyst import TransactionAnalyst
from agents.blue_team.osint_agent import OSINTAgent
from agents.blue_team.lead_investigator import LeadInvestigator
from agents.blue_team.report_writer import ReportWriter

# Configure Streamlit page
st.set_page_config(
    page_title="AML-FT Adversarial Simulation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .stProgress .st-bo {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {}
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'results_available' not in st.session_state:
    st.session_state.results_available = False


def main():
    """Main Streamlit application."""
    st.title("🎯 AML-FT Adversarial Simulation")
    st.markdown("**Advanced Anti-Money Laundering Detection using AI Multi-Agents**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "⚙️ Configuration", "🚀 Run Simulation", "📊 Results", "📋 Reports", "📈 Analytics"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "⚙️ Configuration":
        show_configuration_page()
    elif page == "🚀 Run Simulation":
        show_simulation_page()
    elif page == "📊 Results":
        show_results_page()
    elif page == "📋 Reports":
        show_reports_page()
    elif page == "📈 Analytics":
        show_analytics_page()


def show_home_page():
    """Display the home page with project overview."""
    st.header("Welcome to the AML-FT Adversarial Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Red Team (Criminals)")
        st.markdown("""
        - **Mastermind Agent**: Creates sophisticated money laundering plans
        - **Operator Agent**: Executes criminal strategies
        - **Techniques**: Smurfing, Shell Companies, Money Mules, etc.
        """)
        
        st.subheader("🔵 Blue Team (Investigators)")
        st.markdown("""
        - **Transaction Analyst**: Detects suspicious patterns
        - **OSINT Agent**: Gathers external intelligence
        - **Lead Investigator**: Constructs criminal narratives
        - **Report Writer**: Generates compliance reports
        """)
    
    with col2:
        st.subheader("🎮 How It Works")
        st.markdown("""
        1. **Setup**: Generate realistic financial ecosystem
        2. **Attack**: Red Team creates and executes criminal plans
        3. **Defense**: Blue Team analyzes and detects suspicious activities
        4. **Evaluation**: Measure detection performance and accuracy
        """)
        
        st.subheader("📈 Key Metrics")
        st.markdown("""
        - **Precision**: Accuracy of suspicious entity detection
        - **Recall**: Percentage of actual criminals detected
        - **F1-Score**: Overall detection performance
        - **Risk Assessment**: Comprehensive threat evaluation
        """)
    
    # Quick stats (if available)
    if st.session_state.results_available:
        st.subheader("📊 Latest Simulation Results")
        display_quick_stats()
    
    # Getting started
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Configure** your simulation parameters in the Configuration page
    2. **Run** the simulation to see Red Team vs Blue Team in action
    3. **Analyze** the results to understand detection performance
    4. **Generate** professional compliance reports
    """)


def show_configuration_page():
    """Display the configuration page."""
    st.header("⚙️ Simulation Configuration")
    
    # LLM Configuration
    st.subheader("🤖 LLM Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "anthropic", "google"],
            help="Choose your preferred LLM provider"
        )
        
        llm_model = st.text_input(
            "Model Name",
            value="gpt-4-turbo-preview" if llm_provider == "openai" else "claude-3-sonnet-20240229",
            help="Specify the model to use"
        )
    
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Controls randomness in LLM responses"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1000,
            max_value=8000,
            value=4000,
            help="Maximum tokens for LLM responses"
        )
    
    # Simulation Parameters
    st.subheader("🎯 Simulation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Red Team Configuration**")
        target_amount = st.number_input(
            "Target Amount to Launder ($)",
            min_value=100000,
            max_value=10000000,
            value=500000,
            step=50000
        )
        
        complexity_level = st.selectbox(
            "Complexity Level",
            ["simple", "medium", "complex"],
            index=1
        )
        
        techniques_enabled = st.multiselect(
            "Enabled Techniques",
            ["smurfing", "shell_companies", "money_mules", "cash_intensive_businesses", "cryptocurrency"],
            default=["smurfing", "shell_companies", "money_mules"]
        )
    
    with col2:
        st.markdown("**Blue Team Configuration**")
        detection_threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            help="Minimum confidence for flagging suspicious activities"
        )
        
        investigation_depth = st.selectbox(
            "Investigation Depth",
            ["basic", "standard", "thorough"],
            index=2
        )
        
        enable_osint = st.checkbox("Enable OSINT Analysis", value=True)
        enable_reports = st.checkbox("Generate SAR Reports", value=True)
    
    # Data Generation Settings
    st.subheader("📊 Data Generation")
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_count = st.number_input(
            "Normal Transactions",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=5000
        )
        
        customer_count = st.number_input(
            "Customer Count",
            min_value=100,
            max_value=10000,
            value=5000,
            step=500
        )
    
    with col2:
        business_count = st.number_input(
            "Business Count",
            min_value=10,
            max_value=1000,
            value=500,
            step=50
        )
        
        time_period = st.number_input(
            "Time Period (days)",
            min_value=30,
            max_value=730,
            value=365,
            step=30
        )
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        config = {
            'llm': {
                'provider': llm_provider,
                'model': llm_model,
                'temperature': temperature,
                'max_tokens': max_tokens
            },
            'simulation': {
                'red_team': {
                    'target_amount': target_amount,
                    'complexity_level': complexity_level,
                    'techniques_enabled': techniques_enabled
                },
                'blue_team': {
                    'detection_threshold': detection_threshold,
                    'investigation_depth': investigation_depth,
                    'enable_osint': enable_osint,
                    'enable_reports': enable_reports
                }
            },
            'data': {
                'transaction_count': transaction_count,
                'customer_count': customer_count,
                'business_count': business_count,
                'time_period': time_period
            }
        }
        
        st.session_state.simulation_config = config
        st.success("✅ Configuration saved successfully!")


def show_simulation_page():
    """Display the simulation execution page."""
    st.header("🚀 Run Adversarial Simulation")
    
    # Check if configuration exists
    if 'simulation_config' not in st.session_state:
        st.warning("⚠️ Please configure simulation parameters first!")
        st.stop()
    
    config = st.session_state.simulation_config
    
    # Display current configuration
    st.subheader("📋 Current Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Amount", f"${config['simulation']['red_team']['target_amount']:,}")
        st.metric("Complexity", config['simulation']['red_team']['complexity_level'].title())
    
    with col2:
        st.metric("Transactions", f"{config['data']['transaction_count']:,}")
        st.metric("Detection Threshold", f"{config['simulation']['blue_team']['detection_threshold']:.1%}")
    
    with col3:
        st.metric("LLM Provider", config['llm']['provider'].title())
        st.metric("Investigation Depth", config['simulation']['blue_team']['investigation_depth'].title())
    
    # Simulation controls
    st.subheader("🎮 Simulation Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Simulation", disabled=st.session_state.simulation_running):
            run_simulation()
    
    with col2:
        if st.button("⏹️ Stop Simulation", disabled=not st.session_state.simulation_running):
            st.session_state.simulation_running = False
            st.warning("Simulation stopped by user")
    
    with col3:
        if st.button("🔄 Reset Simulation"):
            reset_simulation()
    
    # Display simulation progress
    if st.session_state.simulation_running:
        display_simulation_progress()


def run_simulation():
    """Execute the adversarial simulation."""
    st.session_state.simulation_running = True
    config = st.session_state.simulation_config
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Phase 1: Data Generation
        status_text.text("🏦 Phase 1: Generating financial ecosystem...")
        progress_bar.progress(10)
        
        generator = TransactionGenerator()
        customers = generator.generate_customers()
        businesses = generator.generate_businesses()
        normal_transactions = generator.generate_transactions()
        
        progress_bar.progress(20)
        
        # Phase 2: Red Team Attack
        status_text.text("🔴 Phase 2: Red Team planning attack...")
        progress_bar.progress(30)
        
        mastermind = MastermindAgent()
        criminal_plan = mastermind.create_laundering_plan(
            target_amount=config['simulation']['red_team']['target_amount'],
            complexity_level=config['simulation']['red_team']['complexity_level']
        )
        
        progress_bar.progress(40)
        
        status_text.text("⚡ Phase 2: Red Team executing plan...")
        operator = OperatorAgent(customers=customers, businesses=businesses)
        execution_result = operator.execute_plan(criminal_plan, normal_transactions)
        
        if not execution_result['success']:
            st.error("❌ Red Team execution failed!")
            st.session_state.simulation_running = False
            return
        
        progress_bar.progress(50)
        
        # Combine datasets
        criminal_transactions = execution_result['criminal_transactions']
        normal_transactions['is_criminal'] = False
        criminal_transactions['is_criminal'] = True
        
        common_columns = list(set(normal_transactions.columns) & set(criminal_transactions.columns))
        combined_data = pd.concat([
            normal_transactions[common_columns],
            criminal_transactions[common_columns]
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        progress_bar.progress(60)
        
        # Phase 3: Blue Team Defense
        status_text.text("🔵 Phase 3: Blue Team analyzing transactions...")
        
        # Transaction Analysis
        analyst = TransactionAnalyst()
        investigation_data = combined_data.drop('is_criminal', axis=1)
        analysis_results = analyst.analyze_transactions(investigation_data)
        
        progress_bar.progress(70)
        
        # OSINT Analysis (if enabled)
        osint_results = {}
        if config['simulation']['blue_team']['enable_osint']:
            status_text.text("🔍 Phase 3: OSINT intelligence gathering...")
            osint_agent = OSINTAgent()
            osint_results = osint_agent.investigate_entities(analysis_results.get('suspicious_entities', []))
        
        progress_bar.progress(80)
        
        # Lead Investigation
        status_text.text("🕵️ Phase 3: Lead investigator constructing narratives...")
        investigator = LeadInvestigator()
        narratives = investigator.investigate_case(analysis_results, osint_results)
        
        progress_bar.progress(90)
        
        # Report Generation (if enabled)
        reports = []
        if config['simulation']['blue_team']['enable_reports']:
            status_text.text("📝 Phase 3: Generating compliance reports...")
            report_writer = ReportWriter()
            reports = report_writer.generate_sar_reports(narratives)
        
        progress_bar.progress(100)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(
            combined_data, analysis_results, execution_result
        )
        
        # Store results
        st.session_state.simulation_data = {
            'config': config,
            'normal_transactions': normal_transactions,
            'criminal_transactions': criminal_transactions,
            'combined_data': combined_data,
            'criminal_plan': criminal_plan,
            'execution_result': execution_result,
            'analysis_results': analysis_results,
            'osint_results': osint_results,
            'narratives': narratives,
            'reports': reports,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now()
        }
        
        st.session_state.results_available = True
        st.session_state.simulation_running = False
        
        status_text.text("✅ Simulation completed successfully!")
        st.success("🎉 Adversarial simulation completed! Check the Results page for detailed analysis.")
        
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        st.session_state.simulation_running = False


def calculate_performance_metrics(combined_data, analysis_results, execution_result):
    """Calculate performance metrics for the simulation."""
    # Get detected entities
    detected_entities = set(entity['entity_id'] for entity in analysis_results.get('suspicious_entities', []))
    
    # Get actual criminal entities
    actual_criminal_entities = set()
    for entity_type, entities in execution_result.get('entities_created', {}).items():
        for entity in entities:
            actual_criminal_entities.add(entity['entity_id'])
    
    # Add entities from criminal transactions
    criminal_data = combined_data[combined_data['is_criminal'] == True]
    actual_criminal_entities.update(criminal_data['sender_id'].unique())
    actual_criminal_entities.update(criminal_data['receiver_id'].unique())
    
    # Calculate metrics
    true_positives = len(detected_entities & actual_criminal_entities)
    false_positives = len(detected_entities - actual_criminal_entities)
    false_negatives = len(actual_criminal_entities - detected_entities)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'detected_entities': len(detected_entities),
        'actual_criminal_entities': len(actual_criminal_entities),
        'total_transactions': len(combined_data),
        'criminal_transactions': len(criminal_data)
    }


def display_simulation_progress():
    """Display real-time simulation progress."""
    st.subheader("⏳ Simulation in Progress...")
    
    # This would be enhanced with real-time updates in a production version
    with st.spinner("Running adversarial simulation..."):
        time.sleep(1)  # Simulate processing time


def reset_simulation():
    """Reset simulation state."""
    st.session_state.simulation_running = False
    st.session_state.results_available = False
    st.session_state.simulation_data = {}
    st.success("🔄 Simulation reset successfully!")


def show_results_page():
    """Display simulation results."""
    st.header("📊 Simulation Results")
    
    if not st.session_state.results_available:
        st.warning("⚠️ No simulation results available. Please run a simulation first!")
        return
    
    data = st.session_state.simulation_data
    metrics = data['performance_metrics']
    
    # Performance Overview
    st.subheader("🎯 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Precision",
            f"{metrics['precision']:.2%}",
            delta=f"{metrics['true_positives']}/{metrics['detected_entities']} correct"
        )
    
    with col2:
        st.metric(
            "Recall",
            f"{metrics['recall']:.2%}",
            delta=f"{metrics['true_positives']}/{metrics['actual_criminal_entities']} detected"
        )
    
    with col3:
        st.metric(
            "F1-Score",
            f"{metrics['f1_score']:.2%}",
            delta="Overall Performance"
        )
    
    with col4:
        criminal_pct = metrics['criminal_transactions'] / metrics['total_transactions'] * 100
        st.metric(
            "Criminal %",
            f"{criminal_pct:.2%}",
            delta=f"{metrics['criminal_transactions']:,} transactions"
        )
    
    # Battle Result
    st.subheader("⚔️ Battle Result")
    
    if metrics['f1_score'] > 0.8:
        st.success("🔵 **BLUE TEAM WINS!** - Excellent detection performance")
        result_color = "success"
    elif metrics['f1_score'] > 0.6:
        st.success("🔵 **BLUE TEAM WINS!** - Good detection performance")
        result_color = "success"
    elif metrics['f1_score'] > 0.4:
        st.warning("⚖️ **DRAW!** - Partial detection")
        result_color = "warning"
    else:
        st.error("🔴 **RED TEAM WINS!** - Poor detection performance")
        result_color = "danger"
    
    # Detailed Analysis
    st.subheader("🔍 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "🕸️ Network", "📊 Transactions", "🎯 Entities"])
    
    with tab1:
        display_metrics_analysis(data)
    
    with tab2:
        display_network_analysis(data)
    
    with tab3:
        display_transaction_analysis(data)
    
    with tab4:
        display_entity_analysis(data)


def display_metrics_analysis(data):
    """Display detailed metrics analysis."""
    metrics = data['performance_metrics']
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    confusion_data = pd.DataFrame({
        'Predicted': ['Criminal', 'Criminal', 'Normal', 'Normal'],
        'Actual': ['Criminal', 'Normal', 'Criminal', 'Normal'],
        'Count': [
            metrics['true_positives'],
            metrics['false_positives'],
            metrics['false_negatives'],
            metrics['total_transactions'] - metrics['true_positives'] - metrics['false_positives'] - metrics['false_negatives']
        ]
    })
    
    fig = px.bar(
        confusion_data,
        x='Predicted',
        y='Count',
        color='Actual',
        title="Detection Confusion Matrix",
        text='Count'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Trends (if multiple runs were available)
    st.subheader("Performance Metrics Breakdown")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Value': [metrics['precision'], metrics['recall'], metrics['f1_score']],
        'Color': ['#1f77b4', '#ff7f0e', '#2ca02c']
    })
    
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        color='Color',
        title="Performance Metrics Summary",
        text='Value'
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def display_network_analysis(data):
    """Display network analysis visualization."""
    st.subheader("🕸️ Transaction Network Analysis")
    
    # Network statistics
    network_analysis = data['analysis_results'].get('network_analysis', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Network Nodes", network_analysis.get('nodes', 0))
    
    with col2:
        st.metric("Network Edges", network_analysis.get('edges', 0))
    
    with col3:
        st.metric("Network Density", f"{network_analysis.get('density', 0):.3f}")
    
    # Suspicious communities
    suspicious_subgraphs = network_analysis.get('suspicious_subgraphs', [])
    if suspicious_subgraphs:
        st.subheader("🚨 Suspicious Communities")
        
        for i, subgraph in enumerate(suspicious_subgraphs):
            with st.expander(f"Community {i+1}: {subgraph.get('suspicion_reason', 'Unknown')}"):
                st.write(f"**Nodes:** {', '.join(subgraph.get('nodes', []))}")
                st.write(f"**Internal Connections:** {subgraph.get('internal_connections', 0)}")
                st.write(f"**Total Volume:** ${subgraph.get('total_volume', 0):,.2f}")
    
    # Centrality Analysis
    centrality_analysis = network_analysis.get('centrality_analysis', {})
    if centrality_analysis:
        st.subheader("🎯 Most Central Entities")
        
        # Top betweenness centrality
        top_betweenness = centrality_analysis.get('top_betweenness', [])[:5]
        if top_betweenness:
            st.write("**Top Betweenness Centrality:**")
            for entity, score in top_betweenness:
                st.write(f"- {entity}: {score:.3f}")


def display_transaction_analysis(data):
    """Display transaction analysis."""
    st.subheader("📊 Transaction Analysis")
    
    combined_data = data['combined_data']
    
    # Transaction volume over time
    combined_data['date'] = pd.to_datetime(combined_data['timestamp']).dt.date
    
    daily_volume = combined_data.groupby(['date', 'is_criminal'])['amount'].sum().reset_index()
    
    fig = px.line(
        daily_volume,
        x='date',
        y='amount',
        color='is_criminal',
        title="Daily Transaction Volume",
        labels={'amount': 'Volume ($)', 'is_criminal': 'Transaction Type'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Amount distribution
    fig = px.histogram(
        combined_data,
        x='amount',
        color='is_criminal',
        nbins=50,
        title="Transaction Amount Distribution",
        labels={'amount': 'Amount ($)', 'is_criminal': 'Transaction Type'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Transaction patterns
    st.subheader("🔄 Transaction Patterns")
    
    # Anomalies detected
    anomalies = data['analysis_results'].get('anomalies_detected', {})
    st.metric("Anomalies Detected", f"{anomalies.get('count', 0)} ({anomalies.get('percentage', 0):.1f}%)")
    
    # Structuring cases
    structuring = data['analysis_results'].get('structuring_analysis', {})
    st.metric("Structuring Cases", structuring.get('cases_detected', 0))


def display_entity_analysis(data):
    """Display entity analysis."""
    st.subheader("🎯 Entity Analysis")
    
    # Suspicious entities
    suspicious_entities = data['analysis_results'].get('suspicious_entities', [])
    
    if suspicious_entities:
        st.subheader("🚨 Suspicious Entities Detected")
        
        # Convert to DataFrame for display
        entities_df = pd.DataFrame(suspicious_entities)
        
        # Display top entities
        st.dataframe(
            entities_df[['entity_id', 'reason', 'risk_score', 'detection_method']].head(10),
            use_container_width=True
        )
        
        # Risk score distribution
        fig = px.histogram(
            entities_df,
            x='risk_score',
            nbins=20,
            title="Risk Score Distribution",
            labels={'risk_score': 'Risk Score', 'count': 'Number of Entities'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detection methods
        method_counts = entities_df['detection_method'].value_counts()
        fig = px.pie(
            values=method_counts.values,
            names=method_counts.index,
            title="Detection Methods Used"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_reports_page():
    """Display generated reports."""
    st.header("📋 Compliance Reports")
    
    if not st.session_state.results_available:
        st.warning("⚠️ No reports available. Please run a simulation first!")
        return
    
    data = st.session_state.simulation_data
    reports = data.get('reports', [])
    narratives = data.get('narratives', [])
    
    if not reports:
        st.warning("⚠️ No SAR reports were generated. Enable report generation in configuration.")
        return
    
    # Reports Overview
    st.subheader("📊 Reports Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reports", len(reports))
    
    with col2:
        high_risk_reports = len([r for r in reports if r.risk_level == 'high'])
        st.metric("High Risk Reports", high_risk_reports)
    
    with col3:
        st.metric("Narratives Generated", len(narratives))
    
    # Individual Reports
    st.subheader("📄 Individual Reports")
    
    for i, report in enumerate(reports):
        with st.expander(f"SAR Report {i+1}: {report.sar_id}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Subject Entity:** {report.subject_entity}")
                st.write(f"**Risk Level:** {report.risk_level.upper()}")
                st.write(f"**Confidence Score:** {report.confidence_score:.2f}")
                st.write(f"**Filing Date:** {report.filing_date.strftime('%Y-%m-%d')}")
            
            with col2:
                st.write(f"**Status:** {report.status}")
                st.write(f"**Created By:** {report.created_by}")
                st.write(f"**Evidence Pieces:** {len(report.supporting_evidence)}")
                st.write(f"**Recommendations:** {len(report.recommendations)}")
            
            st.write("**Narrative Summary:**")
            st.write(report.narrative_summary)
            
            if report.suspicious_activities:
                st.write("**Suspicious Activities:**")
                for activity in report.suspicious_activities[:5]:
                    st.write(f"- {activity}")
            
            if report.recommendations:
                st.write("**Recommendations:**")
                for rec in report.recommendations:
                    st.write(f"- {rec}")


def show_analytics_page():
    """Display advanced analytics."""
    st.header("📈 Advanced Analytics")
    
    if not st.session_state.results_available:
        st.warning("⚠️ No analytics available. Please run a simulation first!")
        return
    
    data = st.session_state.simulation_data
    
    # Performance Analytics
    st.subheader("📊 Performance Analytics")
    
    metrics = data['performance_metrics']
    
    # ROC-like curve (simplified)
    thresholds = np.linspace(0, 1, 11)
    precision_curve = []
    recall_curve = []
    
    for threshold in thresholds:
        # Simulate different threshold effects
        adjusted_tp = max(0, metrics['true_positives'] - int((1-threshold) * metrics['true_positives']))
        adjusted_fp = max(0, metrics['false_positives'] - int(threshold * metrics['false_positives']))
        adjusted_fn = metrics['false_negatives'] + (metrics['true_positives'] - adjusted_tp)
        
        prec = adjusted_tp / (adjusted_tp + adjusted_fp) if (adjusted_tp + adjusted_fp) > 0 else 0
        rec = adjusted_tp / (adjusted_tp + adjusted_fn) if (adjusted_tp + adjusted_fn) > 0 else 0
        
        precision_curve.append(prec)
        recall_curve.append(rec)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision_curve, mode='lines+markers', name='Precision'))
    fig.add_trace(go.Scatter(x=thresholds, y=recall_curve, mode='lines+markers', name='Recall'))
    fig.update_layout(
        title="Precision-Recall vs Threshold",
        xaxis_title="Detection Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Technique Effectiveness
    st.subheader("🎯 Technique Effectiveness")
    
    narratives = data.get('narratives', [])
    if narratives:
        technique_scores = {}
        for narrative in narratives:
            technique = narrative.criminal_technique
            if technique not in technique_scores:
                technique_scores[technique] = []
            technique_scores[technique].append(narrative.confidence_score)
        
        # Average confidence by technique
        avg_scores = {tech: np.mean(scores) for tech, scores in technique_scores.items()}
        
        fig = px.bar(
            x=list(avg_scores.keys()),
            y=list(avg_scores.values()),
            title="Average Confidence Score by Technique",
            labels={'x': 'Technique', 'y': 'Average Confidence Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline Analysis
    st.subheader("⏰ Timeline Analysis")
    
    combined_data = data['combined_data']
    combined_data['hour'] = pd.to_datetime(combined_data['timestamp']).dt.hour
    
    hourly_activity = combined_data.groupby(['hour', 'is_criminal']).size().reset_index(name='count')
    
    fig = px.bar(
        hourly_activity,
        x='hour',
        y='count',
        color='is_criminal',
        title="Transaction Activity by Hour",
        labels={'hour': 'Hour of Day', 'count': 'Transaction Count', 'is_criminal': 'Transaction Type'}
    )
    st.plotly_chart(fig, use_container_width=True)


def display_quick_stats():
    """Display quick statistics on the home page."""
    if not st.session_state.results_available:
        return
    
    data = st.session_state.simulation_data
    metrics = data['performance_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    
    with col4:
        st.metric("Entities Detected", f"{metrics['detected_entities']}")


if __name__ == "__main__":
    main() 