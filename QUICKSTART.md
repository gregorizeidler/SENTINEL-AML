# 🚀 Quick Start Guide

## AML-FT Adversarial Simulation

This guide will get you up and running with the AML-FT adversarial simulation in minutes!

## 📋 Prerequisites

- Python 3.9 or higher
- Git
- API key for at least one LLM provider (OpenAI, Anthropic, or Google)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aml-adversarial-simulation.git
cd aml-adversarial-simulation
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Copy the example environment file:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```bash
# Example for OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Or for Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Or for Google
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Update Configuration

Edit `config/config.yaml` to match your preferred LLM provider:

```yaml
llm:
  provider: "openai"  # or "anthropic" or "google"
  model: "gpt-4-turbo-preview"
  api_key: "${OPENAI_API_KEY}"
```

## 🎮 Running the Simulation

### Option 1: Full Demo (Recommended)

Run the complete adversarial simulation:

```bash
python demo.py
```

This will:
- Generate realistic financial data
- Create and execute criminal plans (Red Team)
- Detect suspicious activities (Blue Team)
- Show performance metrics and results

### Option 2: Test Red Team Only

Test just the criminal agents:

```bash
python test_red_team.py
```

### Option 3: Individual Components

Test individual components:

```bash
# Test data generation
python src/data/transaction_generator.py

# Test Mastermind Agent
python src/agents/red_team/mastermind_agent.py

# Test Transaction Analyst
python src/agents/blue_team/transaction_analyst.py
```

## 📊 Understanding the Output

### Console Output

The simulation provides real-time feedback:

```
🎯 AML-FT ADVERSARIAL SIMULATION
🔴 Red Team (Criminals) vs 🔵 Blue Team (Investigators)

1️⃣ SETTING UP THE BATTLEFIELD
🏦 Generating realistic financial ecosystem...
   ✅ Created 5,000 customers
   ✅ Created 500 businesses
   ✅ Generated 50,000 normal transactions

2️⃣ 🔴 RED TEAM LAUNCHES ATTACK
🧠 Mastermind Agent planning criminal operation...
   ✅ Criminal plan created: PLAN_20240101_123456
   ✅ Techniques: smurfing, money_mules
   ✅ Risk level: medium

3️⃣ 🔵 BLUE TEAM LAUNCHES INVESTIGATION
🔍 Transaction Analyst beginning investigation...
   ✅ Analysis completed successfully!

4️⃣ ⚔️ BATTLE RESULTS
📊 Evaluating Blue Team detection performance...
   🎯 Detection Performance:
      • Precision: 75.00%
      • Recall: 60.00%
      • F1-Score: 66.67%
```

### Generated Files

Results are saved in the `demo_results/` directory:

- `simulation_results.json` - Overall performance metrics
- `criminal_plan.json` - Detailed criminal plan
- `analysis_results.json` - Blue Team analysis results
- `combined_transactions.csv` - Full transaction dataset
- `criminal_transactions.csv` - Only criminal transactions
- `normal_transactions.csv` - Only normal transactions

## 🔍 Key Metrics Explained

### Red Team Metrics
- **Plan Complexity**: Simple, Medium, or Complex
- **Techniques Used**: Number and types of laundering methods
- **Entities Created**: Criminal entities (mules, shell companies, etc.)
- **Transactions Generated**: Number of criminal transactions

### Blue Team Metrics
- **Precision**: Percentage of detected entities that are actually criminal
- **Recall**: Percentage of actual criminals that were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Analysis Methods**: Number of detection techniques used

### Battle Results
- **🔵 Blue Team Wins**: F1-Score > 60%
- **⚖️ Draw**: F1-Score 40-60%
- **🔴 Red Team Wins**: F1-Score < 40%

## 🛠️ Customization

### Adjust Simulation Parameters

Edit `config/config.yaml`:

```yaml
simulation:
  red_team:
    target_amount_min: 100000
    target_amount_max: 1000000
    complexity_level: "medium"  # simple, medium, complex
    techniques_enabled:
      - "smurfing"
      - "shell_companies"
      - "money_mules"
  
  blue_team:
    detection_threshold: 0.7
    investigation_depth: "thorough"
```

### Change Data Volume

```yaml
data:
  normal_transactions:
    count: 50000  # Increase for more data
    customer_count: 5000
    business_count: 500
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: Invalid API key
   ```
   - Check your `.env` file
   - Verify API key is correct
   - Ensure you have credits/quota

2. **Import Error**
   ```
   ModuleNotFoundError: No module named 'openai'
   ```
   - Run `pip install -r requirements.txt`
   - Activate virtual environment

3. **Memory Error**
   ```
   MemoryError: Unable to allocate array
   ```
   - Reduce transaction count in config
   - Use smaller dataset for testing

### Performance Tips

1. **Faster Testing**: Reduce data volume in config
2. **Better Results**: Use more sophisticated LLM models
3. **Lower Costs**: Use fallback plans when LLM unavailable

## 📈 Next Steps

### Explore Advanced Features

1. **Add More Agents**: Implement OSINT and Report Writer agents
2. **Interactive Interface**: Build Streamlit dashboard
3. **Adaptive Learning**: Implement feedback loops
4. **Real Data**: Integrate with actual financial datasets

### Extend the System

1. **New Techniques**: Add cryptocurrency laundering
2. **Better Detection**: Implement graph neural networks
3. **Compliance**: Add regulatory reporting features
4. **Visualization**: Create network analysis charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This project is for educational and research purposes only. Do not use for actual criminal activities. All simulated scenarios are fictional and designed to improve financial crime detection capabilities.

---

**Need help?** Check the [full documentation](README.md) or open an issue! 