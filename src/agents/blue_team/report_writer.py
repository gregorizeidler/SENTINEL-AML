"""
Blue Team Report Writer Agent

This agent generates professional Suspicious Activity Reports (SAR) and other
compliance documents based on the investigation findings from the Lead Investigator.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path
import uuid

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# LLM imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None


@dataclass
class SARReport:
    """Data class for Suspicious Activity Report."""
    sar_id: str
    filing_date: datetime
    subject_entity: str
    narrative_summary: str
    suspicious_activities: List[str]
    transaction_details: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str
    confidence_score: float
    supporting_evidence: List[str]
    regulatory_citations: List[str]
    follow_up_actions: List[str]
    created_by: str
    reviewed_by: Optional[str] = None
    status: str = "draft"


class ReportWriter:
    """
    Generates professional compliance reports from investigation findings.
    
    This agent creates SAR-compliant reports and other regulatory documents
    based on the criminal narratives constructed by the Lead Investigator.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Report Writer."""
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        self.generated_reports = []
        self.templates = self._load_report_templates()
        
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
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.2,
                'max_tokens': 4000
            },
            'reporting': {
                'output_format': 'markdown',
                'include_visualizations': True,
                'compliance_standard': 'FinCEN_SAR',
                'auto_generate_citations': True,
                'quality_threshold': 0.7
            },
            'organization': {
                'name': 'Financial Institution',
                'compliance_officer': 'Chief Compliance Officer',
                'reporting_contact': 'compliance@institution.com',
                'license_number': 'FI-2024-001'
            }
        }
    
    def _initialize_llm(self):
        """Initialize LLM client for report generation."""
        if not openai:
            return None
            
        provider = self.config['llm']['provider']
        
        if provider == 'openai':
            return OpenAI(api_key=self.config['llm'].get('api_key', 'your-api-key'))
        
        return None
    
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates."""
        return {
            'sar_template': """
# SUSPICIOUS ACTIVITY REPORT (SAR)

**Report ID:** {sar_id}
**Filing Date:** {filing_date}
**Subject:** {subject_entity}
**Risk Level:** {risk_level}

## EXECUTIVE SUMMARY

{narrative_summary}

## SUSPICIOUS ACTIVITIES IDENTIFIED

{suspicious_activities}

## TRANSACTION DETAILS

{transaction_details}

## SUPPORTING EVIDENCE

{supporting_evidence}

## REGULATORY COMPLIANCE

{regulatory_citations}

## RECOMMENDATIONS

{recommendations}

## FOLLOW-UP ACTIONS

{follow_up_actions}

---
**Prepared by:** {created_by}
**Date:** {creation_date}
**Institution:** {institution_name}
""",
            'investigation_summary': """
# INVESTIGATION SUMMARY REPORT

**Case ID:** {case_id}
**Investigation Date:** {investigation_date}
**Lead Investigator:** {lead_investigator}

## CASE OVERVIEW

{case_overview}

## FINDINGS

{findings}

## EVIDENCE ANALYSIS

{evidence_analysis}

## CONCLUSIONS

{conclusions}

## NEXT STEPS

{next_steps}
""",
            'compliance_memo': """
# COMPLIANCE MEMORANDUM

**To:** {recipient}
**From:** {sender}
**Date:** {date}
**Re:** {subject}

## BACKGROUND

{background}

## ANALYSIS

{analysis}

## RECOMMENDATIONS

{recommendations}

## CONCLUSION

{conclusion}
"""
        }
    
    def generate_sar_reports(self, narratives: List[Any]) -> List[SARReport]:
        """
        Generate SAR reports from criminal narratives.
        
        Args:
            narratives: List of criminal narratives from Lead Investigator
            
        Returns:
            List of generated SAR reports
        """
        print(f"📝 Generating SAR reports for {len(narratives)} narratives...")
        
        reports = []
        
        for narrative in narratives:
            # Only generate SARs for high and medium risk narratives
            if narrative.risk_level in ['high', 'medium']:
                print(f"   📋 Generating SAR for {narrative.title}...")
                
                sar_report = self._generate_single_sar(narrative)
                if sar_report:
                    reports.append(sar_report)
        
        self.generated_reports.extend(reports)
        
        print(f"✅ Generated {len(reports)} SAR reports")
        return reports
    
    def _generate_single_sar(self, narrative: Any) -> Optional[SARReport]:
        """Generate a single SAR report from a narrative."""
        try:
            # Generate unique SAR ID
            sar_id = f"SAR-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            
            # Extract primary entity
            primary_entity = narrative.entities_involved[0] if narrative.entities_involved else "Unknown Entity"
            
            # Generate narrative summary
            narrative_summary = self._generate_narrative_summary(narrative)
            
            # Extract suspicious activities
            suspicious_activities = self._extract_suspicious_activities(narrative)
            
            # Extract transaction details
            transaction_details = self._extract_transaction_details(narrative)
            
            # Generate recommendations
            recommendations = narrative.recommended_actions or self._generate_default_recommendations(narrative)
            
            # Generate supporting evidence
            supporting_evidence = self._format_supporting_evidence(narrative)
            
            # Generate regulatory citations
            regulatory_citations = self._generate_regulatory_citations(narrative)
            
            # Generate follow-up actions
            follow_up_actions = self._generate_follow_up_actions(narrative)
            
            sar_report = SARReport(
                sar_id=sar_id,
                filing_date=datetime.now(),
                subject_entity=primary_entity,
                narrative_summary=narrative_summary,
                suspicious_activities=suspicious_activities,
                transaction_details=transaction_details,
                recommendations=recommendations,
                risk_level=narrative.risk_level,
                confidence_score=narrative.confidence_score,
                supporting_evidence=supporting_evidence,
                regulatory_citations=regulatory_citations,
                follow_up_actions=follow_up_actions,
                created_by=self.config['organization']['compliance_officer'],
                status="draft"
            )
            
            return sar_report
            
        except Exception as e:
            print(f"   ❌ Failed to generate SAR for {narrative.title}: {str(e)}")
            return None
    
    def _generate_narrative_summary(self, narrative: Any) -> str:
        """Generate enhanced narrative summary using LLM."""
        if not self.llm_client:
            return narrative.summary
        
        # Prepare context for LLM
        context = f"""
Criminal Narrative: {narrative.title}
Technique: {narrative.criminal_technique}
Entities: {', '.join(narrative.entities_involved)}
Risk Level: {narrative.risk_level}
Confidence: {narrative.confidence_score:.2f}

Original Summary: {narrative.summary}

Evidence Timeline:
"""
        
        for event in narrative.timeline[:5]:  # Limit to first 5 events
            context += f"- {event['timestamp'].strftime('%Y-%m-%d')}: {event['event']}\n"
        
        prompt = f"""
As a compliance officer, rewrite the following money laundering investigation summary for a Suspicious Activity Report (SAR). 
The summary should be:
1. Professional and regulatory-compliant
2. Clear and factual
3. Suitable for law enforcement review
4. Include specific suspicious indicators
5. Avoid speculation, stick to facts

{context}

Generate a professional SAR narrative summary:
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are a compliance officer writing regulatory reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )
            
            enhanced_summary = response.choices[0].message.content.strip()
            return enhanced_summary
            
        except Exception as e:
            print(f"   ⚠️ LLM enhancement failed, using original summary: {str(e)}")
            return narrative.summary
    
    def _extract_suspicious_activities(self, narrative: Any) -> List[str]:
        """Extract suspicious activities from narrative."""
        activities = []
        
        # Based on criminal technique
        technique_activities = {
            'structuring': [
                'Multiple transactions just below $10,000 reporting threshold',
                'Systematic breaking of large amounts into smaller transactions',
                'Pattern of deposits designed to avoid currency transaction reporting'
            ],
            'shell_companies': [
                'Use of entities with no apparent legitimate business purpose',
                'Transactions involving companies with minimal operational activity',
                'Rapid movement of funds through business accounts'
            ],
            'money_mules': [
                'Use of third-party accounts for fund transfers',
                'Individuals receiving and immediately transferring funds',
                'Pattern of funds flowing through multiple personal accounts'
            ],
            'cash_businesses': [
                'Cash-intensive business with inconsistent revenue patterns',
                'Deposits not commensurate with business type',
                'Mixing of potentially illicit funds with legitimate business revenue'
            ],
            'layering': [
                'Complex series of transactions to obscure fund origins',
                'Rapid movement of funds through multiple accounts',
                'Use of multiple financial institutions simultaneously'
            ]
        }
        
        activities.extend(technique_activities.get(narrative.criminal_technique, []))
        
        # Add evidence-based activities
        for evidence in narrative.evidence_pieces:
            if evidence.confidence > 0.7:
                activities.append(f"Evidence of {evidence.evidence_type}: {evidence.description}")
        
        return activities[:10]  # Limit to top 10 activities
    
    def _extract_transaction_details(self, narrative: Any) -> List[Dict[str, Any]]:
        """Extract transaction details from narrative."""
        details = []
        
        # Extract from timeline
        for event in narrative.timeline:
            if 'transaction' in event['event'].lower():
                details.append({
                    'date': event['timestamp'].strftime('%Y-%m-%d'),
                    'entity': event['entity'],
                    'description': event['event'],
                    'source': event['source']
                })
        
        # Extract from evidence
        for evidence in narrative.evidence_pieces:
            if evidence.evidence_type in ['suspicious_pattern', 'statistical_anomaly']:
                if hasattr(evidence, 'supporting_data') and evidence.supporting_data:
                    details.append({
                        'date': evidence.timestamp.strftime('%Y-%m-%d'),
                        'entity': evidence.entity_id,
                        'description': evidence.description,
                        'confidence': evidence.confidence,
                        'source': evidence.source
                    })
        
        return details[:20]  # Limit to top 20 details
    
    def _format_supporting_evidence(self, narrative: Any) -> List[str]:
        """Format supporting evidence for the report."""
        evidence_list = []
        
        for evidence in narrative.evidence_pieces:
            evidence_str = f"{evidence.source}: {evidence.description}"
            if evidence.confidence:
                evidence_str += f" (Confidence: {evidence.confidence:.2f})"
            evidence_list.append(evidence_str)
        
        return evidence_list
    
    def _generate_regulatory_citations(self, narrative: Any) -> List[str]:
        """Generate relevant regulatory citations."""
        citations = []
        
        # Base citations for SAR filing
        citations.extend([
            "31 CFR 1020.320 - Reports by banks of suspicious transactions",
            "31 CFR 1010.311 - Filing obligations for reports of transactions in currency",
            "Bank Secrecy Act (BSA) - 31 U.S.C. 5311 et seq."
        ])
        
        # Technique-specific citations
        if narrative.criminal_technique == 'structuring':
            citations.append("31 U.S.C. 5324 - Structuring transactions to evade reporting requirements")
        
        if narrative.criminal_technique == 'shell_companies':
            citations.append("31 CFR 1010.230 - Beneficial ownership requirements")
        
        # Risk-based citations
        if narrative.risk_level == 'high':
            citations.append("31 CFR 1020.220 - Customer due diligence requirements")
        
        return citations
    
    def _generate_follow_up_actions(self, narrative: Any) -> List[str]:
        """Generate follow-up actions based on narrative."""
        actions = []
        
        # Risk-based actions
        if narrative.risk_level == 'high':
            actions.extend([
                "File SAR with FinCEN within 30 days",
                "Notify law enforcement if criminal activity suspected",
                "Consider account closure or restrictions",
                "Enhanced monitoring of related entities"
            ])
        elif narrative.risk_level == 'medium':
            actions.extend([
                "Enhanced due diligence on subject entities",
                "Increased transaction monitoring",
                "Quarterly review of account activity",
                "Consider additional information gathering"
            ])
        
        # Technique-specific actions
        if narrative.criminal_technique == 'structuring':
            actions.append("Review all transactions below $10,000 from subject entities")
        
        if narrative.criminal_technique == 'shell_companies':
            actions.append("Investigate beneficial ownership and business purpose")
        
        return actions
    
    def _generate_default_recommendations(self, narrative: Any) -> List[str]:
        """Generate default recommendations if none provided."""
        recommendations = []
        
        if narrative.risk_level == 'high':
            recommendations.extend([
                "Immediate SAR filing recommended",
                "Account monitoring and potential restrictions",
                "Law enforcement notification if warranted"
            ])
        elif narrative.risk_level == 'medium':
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Additional due diligence required",
                "Periodic review of account activity"
            ])
        
        return recommendations
    
    def generate_report_documents(self, reports: List[SARReport], output_dir: str = "reports/"):
        """Generate formatted report documents."""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"📄 Generating report documents in {output_dir}...")
        
        for report in reports:
            # Generate Markdown report
            markdown_content = self._generate_markdown_report(report)
            markdown_file = Path(output_dir) / f"{report.sar_id}.md"
            
            with open(markdown_file, 'w') as f:
                f.write(markdown_content)
            
            # Generate PDF if ReportLab is available
            if HAS_REPORTLAB:
                pdf_file = Path(output_dir) / f"{report.sar_id}.pdf"
                self._generate_pdf_report(report, str(pdf_file))
            
            print(f"   ✅ Generated report: {report.sar_id}")
        
        # Generate summary report
        summary_file = Path(output_dir) / "SAR_Summary.md"
        summary_content = self._generate_summary_report(reports)
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        print(f"✅ All report documents generated in {output_dir}")
    
    def _generate_markdown_report(self, report: SARReport) -> str:
        """Generate Markdown formatted report."""
        template = self.templates['sar_template']
        
        # Format suspicious activities
        activities_text = "\n".join(f"- {activity}" for activity in report.suspicious_activities)
        
        # Format transaction details
        details_text = ""
        for detail in report.transaction_details:
            details_text += f"- **{detail['date']}**: {detail['description']}\n"
        
        # Format supporting evidence
        evidence_text = "\n".join(f"- {evidence}" for evidence in report.supporting_evidence)
        
        # Format regulatory citations
        citations_text = "\n".join(f"- {citation}" for citation in report.regulatory_citations)
        
        # Format recommendations
        recommendations_text = "\n".join(f"- {rec}" for rec in report.recommendations)
        
        # Format follow-up actions
        actions_text = "\n".join(f"- {action}" for action in report.follow_up_actions)
        
        return template.format(
            sar_id=report.sar_id,
            filing_date=report.filing_date.strftime('%Y-%m-%d'),
            subject_entity=report.subject_entity,
            risk_level=report.risk_level.upper(),
            narrative_summary=report.narrative_summary,
            suspicious_activities=activities_text,
            transaction_details=details_text,
            supporting_evidence=evidence_text,
            regulatory_citations=citations_text,
            recommendations=recommendations_text,
            follow_up_actions=actions_text,
            created_by=report.created_by,
            creation_date=datetime.now().strftime('%Y-%m-%d'),
            institution_name=self.config['organization']['name']
        )
    
    def _generate_pdf_report(self, report: SARReport, output_file: str):
        """Generate PDF formatted report."""
        try:
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("SUSPICIOUS ACTIVITY REPORT", title_style))
            story.append(Spacer(1, 12))
            
            # Report details table
            report_data = [
                ['Report ID:', report.sar_id],
                ['Filing Date:', report.filing_date.strftime('%Y-%m-%d')],
                ['Subject Entity:', report.subject_entity],
                ['Risk Level:', report.risk_level.upper()],
                ['Confidence Score:', f"{report.confidence_score:.2f}"]
            ]
            
            report_table = Table(report_data, colWidths=[2*inch, 4*inch])
            report_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(report_table)
            story.append(Spacer(1, 12))
            
            # Narrative summary
            story.append(Paragraph("EXECUTIVE SUMMARY", styles['Heading2']))
            story.append(Paragraph(report.narrative_summary, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Suspicious activities
            story.append(Paragraph("SUSPICIOUS ACTIVITIES", styles['Heading2']))
            for activity in report.suspicious_activities:
                story.append(Paragraph(f"• {activity}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Recommendations
            story.append(Paragraph("RECOMMENDATIONS", styles['Heading2']))
            for rec in report.recommendations:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            
            doc.build(story)
            
        except Exception as e:
            print(f"   ⚠️ PDF generation failed: {str(e)}")
    
    def _generate_summary_report(self, reports: List[SARReport]) -> str:
        """Generate summary report of all SARs."""
        summary = f"""# SAR FILING SUMMARY

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Total SARs Generated:** {len(reports)}
**Institution:** {self.config['organization']['name']}

## OVERVIEW

"""
        
        # Risk level breakdown
        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        for report in reports:
            risk_counts[report.risk_level] += 1
        
        summary += f"- **High Risk SARs:** {risk_counts['high']}\n"
        summary += f"- **Medium Risk SARs:** {risk_counts['medium']}\n"
        summary += f"- **Low Risk SARs:** {risk_counts['low']}\n\n"
        
        # Technique breakdown
        techniques = {}
        for report in reports:
            # Extract technique from narrative (simplified)
            for activity in report.suspicious_activities:
                if 'structuring' in activity.lower():
                    techniques['Structuring'] = techniques.get('Structuring', 0) + 1
                elif 'shell' in activity.lower():
                    techniques['Shell Companies'] = techniques.get('Shell Companies', 0) + 1
                elif 'mule' in activity.lower():
                    techniques['Money Mules'] = techniques.get('Money Mules', 0) + 1
        
        summary += f"## TECHNIQUES IDENTIFIED\n\n"
        for technique, count in techniques.items():
            summary += f"- **{technique}:** {count} cases\n"
        
        summary += f"\n## INDIVIDUAL REPORTS\n\n"
        
        # List all reports
        for i, report in enumerate(reports, 1):
            summary += f"{i}. **{report.sar_id}** - {report.subject_entity} ({report.risk_level.upper()})\n"
            summary += f"   - Filing Date: {report.filing_date.strftime('%Y-%m-%d')}\n"
            summary += f"   - Confidence: {report.confidence_score:.2f}\n"
            summary += f"   - Status: {report.status}\n\n"
        
        summary += f"## NEXT STEPS\n\n"
        summary += f"1. Review all high-risk SARs for immediate filing\n"
        summary += f"2. Coordinate with law enforcement as appropriate\n"
        summary += f"3. Implement enhanced monitoring for flagged entities\n"
        summary += f"4. Schedule follow-up reviews as recommended\n"
        
        return summary
    
    def get_reports_summary(self) -> str:
        """Get summary of all generated reports."""
        if not self.generated_reports:
            return "No reports have been generated yet."
        
        summary = f"""
Report Writer Summary
====================

Total Reports Generated: {len(self.generated_reports)}
Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Risk Level Breakdown:
"""
        
        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        for report in self.generated_reports:
            risk_counts[report.risk_level] += 1
        
        for level, count in risk_counts.items():
            summary += f"  {level.upper()}: {count} reports\n"
        
        summary += f"\nRecent Reports:\n"
        for report in self.generated_reports[-5:]:  # Last 5 reports
            summary += f"  • {report.sar_id} - {report.subject_entity} ({report.risk_level.upper()})\n"
        
        return summary


def main():
    """Main function for testing the Report Writer."""
    print("Testing Blue Team Report Writer...")
    
    # Create sample narrative (simplified)
    sample_narrative = type('Narrative', (), {
        'narrative_id': 'NARR_TEST_001',
        'title': 'Suspicious Structuring Activity',
        'summary': 'Investigation reveals systematic structuring of transactions to avoid reporting requirements.',
        'criminal_technique': 'structuring',
        'entities_involved': ['ENTITY_001', 'ENTITY_002'],
        'evidence_pieces': [
            type('Evidence', (), {
                'evidence_id': 'EV_001',
                'source': 'Transaction Analysis',
                'evidence_type': 'suspicious_pattern',
                'entity_id': 'ENTITY_001',
                'description': 'Multiple transactions just below $10,000 threshold',
                'confidence': 0.85,
                'timestamp': datetime.now()
            })()
        ],
        'timeline': [
            {
                'timestamp': datetime.now(),
                'event': 'Suspicious transaction pattern detected',
                'entity': 'ENTITY_001',
                'source': 'Transaction Analysis'
            }
        ],
        'confidence_score': 0.82,
        'risk_level': 'high',
        'recommended_actions': ['File SAR immediately', 'Enhanced monitoring'],
        'created_at': datetime.now()
    })()
    
    # Initialize Report Writer
    writer = ReportWriter()
    
    # Generate SAR reports
    print("\nGenerating SAR reports...")
    reports = writer.generate_sar_reports([sample_narrative])
    
    # Generate documents
    print("\nGenerating report documents...")
    writer.generate_report_documents(reports)
    
    # Display summary
    print("\nReport Writer Summary:")
    print("=" * 50)
    print(writer.get_reports_summary())


if __name__ == "__main__":
    main() 