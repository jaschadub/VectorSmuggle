# Sample Documents for VectorSmuggle Testing

This directory contains sample documents in various formats to demonstrate the multi-format document support capabilities of VectorSmuggle. These documents simulate realistic corporate data that might be targeted for exfiltration.

## Document Types

### Financial Documents
- `financial_report.csv` - Quarterly financial data with revenue and profit information
- `budget_analysis.json` - Budget allocation and spending analysis
- `payroll_data.xlsx` - Employee salary and compensation data (simulated)

### HR Documents
- `employee_handbook.md` - Company policies and procedures
- `performance_reviews.txt` - Employee performance evaluation data
- `org_chart.xml` - Organizational structure and reporting relationships

### Technical Documents
- `api_documentation.html` - Internal API documentation with endpoints
- `database_schema.yaml` - Database structure and sensitive table information
- `system_config.json` - System configuration with potential credentials

### Communication
- `executive_emails.eml` - Sample executive email communications
- `project_discussions.mbox` - Project team email discussions

## Security Considerations

These sample documents contain:
- Simulated PII (Social Security Numbers, email addresses)
- Financial data (salaries, revenue figures)
- Technical information (API keys, database schemas)
- Organizational data (employee hierarchies, contact information)

The documents are designed to trigger VectorSmuggle's sensitive data detection capabilities and demonstrate various attack vectors for data exfiltration.

## Usage

Process all sample documents:
```bash
python scripts/embed.py --directory sample_docs --recursive --show-stats
```

Process specific document types:
```bash
python scripts/embed.py --files sample_docs/financial_report.csv sample_docs/payroll_data.xlsx
```

Enable content sanitization:
```bash
python scripts/embed.py --directory sample_docs --sanitize --show-stats