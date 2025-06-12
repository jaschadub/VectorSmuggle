#!/usr/bin/env python3
"""Generate 100 sensitive test documents for VectorSmuggle testing."""

import csv
import json
import random
import string
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timedelta
from pathlib import Path

import yaml


class SensitiveDocumentGenerator:
    """Generate realistic sensitive documents for security testing."""

    def __init__(self, output_dir: str = "internal_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Sample data pools
        self.companies = [
            "TechCorp", "DataSys", "CloudFlow", "SecureNet", "InnovateLabs", "CyberGuard"
        ]
        self.first_names = [
            "Sarah", "Michael", "Jennifer", "David", "Lisa", "Robert",
            "Emily", "James", "Amanda", "Christopher"
        ]
        self.last_names = [
            "Johnson", "Smith", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez"
        ]
        self.departments = [
            "Engineering", "Finance", "HR", "Sales", "Marketing",
            "Operations", "Security", "Legal", "R&D", "IT"
        ]
        self.positions = [
            "Manager", "Director", "VP", "Engineer", "Analyst",
            "Specialist", "Coordinator", "Lead", "Principal", "Senior"
        ]

    def generate_ssn(self) -> str:
        """Generate fake SSN."""
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

    def generate_phone(self) -> str:
        """Generate fake phone number."""
        return f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"

    def generate_email(self, name: str, company: str) -> str:
        """Generate email address."""
        return f"{name.lower().replace(' ', '.')}@{company.lower()}.com"

    def generate_api_key(self) -> str:
        """Generate fake API key."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

    def generate_password(self) -> str:
        """Generate fake password."""
        return ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%", k=12))

    def generate_employee_record_txt(self, file_num: int) -> str:
        """Generate employee record in TXT format."""
        company = random.choice(self.companies)
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        full_name = f"{first_name} {last_name}"

        birth_month = random.choice(['January', 'February', 'March', 'April', 'May', 'June'])
        birth_day = random.randint(1, 28)
        birth_year = random.randint(1980, 1995)
        address_num = random.randint(100, 9999)
        street_name = random.choice(['Main', 'Oak', 'Pine', 'Maple', 'Cedar'])
        city = random.choice(['Seattle', 'Portland', 'San Francisco', 'Austin', 'Denver'])
        state = random.choice(['WA', 'OR', 'CA', 'TX', 'CO'])
        zipcode = random.randint(10000, 99999)
        emergency_contact = random.choice(self.first_names)
        relationship = random.choice(['Spouse', 'Parent', 'Sibling'])
        position_title = random.choice(['Software Engineer', 'Data Analyst', 'Product Manager', 'Security Specialist'])
        hire_month = random.choice(['January', 'February', 'March'])
        hire_day = random.randint(1, 28)
        hire_year = random.randint(2018, 2023)
        manager_name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
        manager_id = f"{company[:2].upper()}-{random.randint(2015,2020)}-{random.randint(100,999):04d}"
        clearance_level = random.choice(['Basic', 'Internal Systems', 'Confidential'])
        review_year1 = random.randint(2021, 2023)
        review_rating1 = random.choice(['Exceeds Expectations', 'Meets Expectations', 'Outstanding'])
        review_score1 = random.uniform(3.5, 5.0)
        review_year2 = random.randint(2020, 2022)
        review_rating2 = random.choice(['Exceeds Expectations', 'Meets Expectations'])
        review_score2 = random.uniform(3.0, 4.5)
        disciplinary = random.choice(['None', 'Verbal Warning - Late Attendance (Resolved)', 'None'])
        insurance_plan = random.choice(['Premium', 'Standard', 'Basic'])
        insurance_coverage = random.choice(['Employee Only', 'Employee + Spouse', 'Family'])
        k401_contrib = random.randint(3, 10)
        k401_match = random.randint(2, 6)
        vacation_days = random.randint(15, 30)
        stock_options = random.randint(500, 5000)
        stock_status = random.choice(['vested', 'unvested', 'partially vested'])
        note = random.choice(['Considered for promotion', 'High performer', 'Key team member', 'Flight risk - monitor'])
        cloud_provider = random.choice(['AWS', 'Azure', 'GCP'])
        additional_note = random.choice(['Authorized for client site visits', 'Remote work approved', 'Security training completed'])
        bg_check_month = random.randint(1, 12)
        bg_check_year = random.randint(2018, 2023)

        content = f"""CONFIDENTIAL EMPLOYEE RECORDS - {company.upper()} INDUSTRIES
================================================================

Employee ID: {company[:2].upper()}-{random.randint(2020,2024)}-{random.randint(1000,9999):04d}
Name: {full_name}
SSN: {self.generate_ssn()}
Date of Birth: {birth_month}, {birth_day}, {birth_year}
Address: {address_num} {street_name} Street, {city}, {state} {zipcode}
Phone: {self.generate_phone()}
Email: {self.generate_email(full_name, company)}
Emergency Contact: {emergency_contact} {last_name} ({relationship}) - {self.generate_phone()}

EMPLOYMENT DETAILS:
Position: {random.choice(self.positions)} {position_title}
Department: {random.choice(self.departments)}
Hire Date: {hire_month}, {hire_day}, {hire_year}
Salary: ${random.randint(80,200)},000 annually
Direct Manager: {manager_name} (ID: {manager_id})
Security Clearance: Level {random.randint(1,3)} - {clearance_level} Access

PERFORMANCE REVIEWS:
{review_year1}: {review_rating1} - {review_score1:.1f}/5.0
{review_year2}: {review_rating2} - {review_score2:.1f}/5.0

DISCIPLINARY ACTIONS: {disciplinary}

BENEFITS:
Health Insurance: {insurance_plan} Plan ({insurance_coverage})
401k Contribution: {k401_contrib}% with {k401_match}% company match
Vacation Days Accrued: {vacation_days} days
Stock Options: {stock_options} shares {stock_status}

CONFIDENTIAL NOTES:
- {note}
- Has access to {cloud_provider} production environments
- {additional_note}
- Background check completed: {bg_check_month:02d}/{bg_check_year} - CLEARED
"""
        return content

    def generate_financial_report_md(self, file_num: int) -> str:
        """Generate financial report in Markdown format."""
        company = random.choice(self.companies)
        quarter = random.choice(['Q1', 'Q2', 'Q3', 'Q4'])
        year = random.randint(2022, 2024)
        revenue = random.randint(25, 75)

        growth_rate = random.randint(15, 35)
        profit_margin = random.randint(12, 25)
        profit_decimal = random.randint(1, 9)
        cloud_pct = 60 + random.randint(-5, 5)
        license_pct = 25 + random.randint(-3, 3)
        services_pct = 10 + random.randint(-2, 2)
        hardware_pct = 5 + random.randint(-1, 1)
        prev_revenue = revenue - random.randint(2, 8)
        yoy_change = random.randint(15, 35)
        gross_profit_change = random.randint(20, 40)
        opex_change = random.randint(8, 15)
        net_income_change = random.randint(50, 120)
        bank_account = random.randint(100000000, 999999999)
        tax_id_part1 = random.randint(10, 99)
        tax_id_part2 = random.randint(1000000, 9999999)
        credit_available = random.randint(5, 25)
        credit_utilized = random.randint(1, 10)
        acquisition_budget = random.randint(10, 50)
        personnel_pct = 39 + random.randint(-3, 3)
        infra_pct = 14 + random.randint(-2, 2)
        rd_pct = 9 + random.randint(-1, 1)
        report_month = random.choice(['January', 'April', 'July', 'October'])
        report_day = random.randint(10, 25)

        content = f"""# CONFIDENTIAL FINANCIAL REPORT - {quarter} {year}
## {company} Industries Internal Use Only

### Executive Summary
{company} Industries achieved record revenue of ${revenue}.{random.randint(1,9)}M in {quarter} {year}, representing {growth_rate}% growth YoY. Net profit margin improved to {profit_margin}.{profit_decimal}% due to operational efficiency gains.

### Revenue Breakdown
- **Cloud Services**: ${revenue * 0.6:.1f}M ({cloud_pct:.1f}% of total revenue)
- **Software Licenses**: ${revenue * 0.25:.1f}M ({license_pct:.1f}% of total revenue)
- **Professional Services**: ${revenue * 0.1:.1f}M ({services_pct:.1f}% of total revenue)
- **Hardware Sales**: ${revenue * 0.05:.1f}M ({hardware_pct:.1f}% of total revenue)

### Key Financial Metrics
| Metric | {quarter} {year} | Previous Quarter | YoY Change |
|--------|---------|---------|------------|
| Revenue | ${revenue}.{random.randint(1,9)}M | ${prev_revenue}.{random.randint(1,9)}M | +{yoy_change}% |
| Gross Profit | ${revenue * 0.67:.1f}M | ${(revenue-5) * 0.67:.1f}M | +{gross_profit_change}% |
| Operating Expenses | ${revenue * 0.49:.1f}M | ${(revenue-3) * 0.49:.1f}M | +{opex_change}% |
| Net Income | ${revenue * 0.18:.1f}M | ${(revenue-4) * 0.15:.1f}M | +{net_income_change}% |

### Sensitive Financial Data
- **Bank Account**: {bank_account} (Primary Operating)
- **Tax ID**: {tax_id_part1}-{tax_id_part2}
- **Credit Line**: ${credit_available}M available, ${credit_utilized}M utilized
- **Acquisition Budget**: ${acquisition_budget}M allocated for strategic acquisitions

### Major Expenses
1. Personnel Costs: ${revenue * 0.39:.1f}M ({personnel_pct}% of revenue)
2. Infrastructure & Cloud: ${revenue * 0.14:.1f}M ({infra_pct}% of revenue)
3. R&D Investment: ${revenue * 0.09:.1f}M ({rd_pct}% of revenue)

**Prepared by**: {random.choice(self.first_names)} {random.choice(self.last_names)}, CFO
**Date**: {report_month} {report_day}, {year}
**Classification**: CONFIDENTIAL - Internal Distribution Only
"""
        return content

    def generate_api_documentation_html(self, file_num: int) -> str:
        """Generate API documentation in HTML format."""
        company = random.choice(self.companies)
        api_key = self.generate_api_key()

        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{company} Internal API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .confidential {{ color: red; font-weight: bold; }}
        .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
        .secret {{ background: #ffe6e6; padding: 5px; }}
    </style>
</head>
<body>
    <h1 class="confidential">CONFIDENTIAL - {company} Internal API Documentation</h1>

    <h2>Authentication</h2>
    <div class="secret">
        <p><strong>Production API Key:</strong> {api_key}</p>
        <p><strong>Admin Token:</strong> {self.generate_api_key()}</p>
        <p><strong>Database Password:</strong> {self.generate_password()}</p>
    </div>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <h3>GET /api/v1/employees</h3>
        <p>Retrieve employee data including SSNs and salary information</p>
        <p><strong>Auth Required:</strong> Level 3 clearance</p>
        <p><strong>Example:</strong> https://internal-api.{company.lower()}.com/api/v1/employees?include_sensitive=true</p>
    </div>

    <div class="endpoint">
        <h3>POST /api/v1/financial/reports</h3>
        <p>Submit financial reports with revenue and profit data</p>
        <p><strong>Database:</strong> postgresql://admin:{self.generate_password()}@db.{company.lower()}.com:5432/finance</p>
    </div>

    <div class="endpoint">
        <h3>GET /api/v1/customers/pii</h3>
        <p>Access customer personally identifiable information</p>
        <p><strong>Encryption Key:</strong> {self.generate_api_key()}</p>
        <p><strong>Access Log:</strong> All requests logged to /var/log/pii-access.log</p>
    </div>

    <h2>Infrastructure Details</h2>
    <ul>
        <li><strong>Production Server:</strong> 10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}</li>
        <li><strong>Database Server:</strong> 10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}</li>
        <li><strong>Backup Server:</strong> 10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}</li>
        <li><strong>VPN Gateway:</strong> vpn.{company.lower()}.com</li>
    </ul>

    <p class="confidential">This document contains sensitive technical information. Do not distribute outside authorized personnel.</p>
</body>
</html>"""
        return content

    def generate_payroll_csv(self, file_num: int) -> str:
        """Generate payroll data in CSV format."""
        company = random.choice(self.companies)

        headers = [
            "Employee_ID", "Full_Name", "SSN", "Department", "Position",
            "Annual_Salary", "Hourly_Rate", "YTD_Gross", "Tax_Withholding",
            "Bank_Account", "Routing_Number"
        ]
        rows = [headers]

        for _ in range(random.randint(15, 25)):
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            salary = random.randint(50000, 200000)
            position_type = random.choice(['Engineer', 'Analyst', 'Manager', 'Specialist'])

            rows.append([
                f"{company[:2].upper()}-{random.randint(2018,2024)}-{random.randint(1000,9999):04d}",
                f"{first_name} {last_name}",
                self.generate_ssn(),
                random.choice(self.departments),
                f"{random.choice(self.positions)} {position_type}",
                salary,
                f"{salary/2080:.2f}",
                f"{salary * random.uniform(0.75, 0.95):.2f}",
                f"{salary * random.uniform(0.20, 0.35):.2f}",
                f"{random.randint(100000000,999999999)}",
                f"{random.randint(100000000,999999999)}"
            ])

        # Convert to CSV string
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        return output.getvalue()

    def generate_system_config_json(self, file_num: int) -> str:
        """Generate system configuration in JSON format."""
        company = random.choice(self.companies)

        config = {
            "system_name": f"{company}_production_system",
            "environment": "production",
            "database": {
                "host": f"db-prod.{company.lower()}.com",
                "port": 5432,
                "username": "admin",
                "password": self.generate_password(),
                "database": "company_data",
                "ssl_cert": f"/etc/ssl/certs/{company.lower()}-prod.pem"
            },
            "api_keys": {
                "stripe": f"sk_live_{self.generate_api_key()}",
                "aws_access_key": f"AKIA{self.generate_api_key()[:16].upper()}",
                "aws_secret_key": self.generate_api_key(),
                "sendgrid": f"SG.{self.generate_api_key()}"
            },
            "encryption": {
                "master_key": self.generate_api_key(),
                "salt": self.generate_api_key()[:16],
                "algorithm": "AES-256-GCM"
            },
            "internal_services": {
                "employee_service": f"https://employees.internal.{company.lower()}.com",
                "finance_service": f"https://finance.internal.{company.lower()}.com",
                "hr_service": f"https://hr.internal.{company.lower()}.com"
            },
            "security": {
                "jwt_secret": self.generate_api_key(),
                "session_timeout": 3600,
                "max_login_attempts": 3,
                "admin_users": [
                    f"admin@{company.lower()}.com",
                    f"security@{company.lower()}.com",
                    f"devops@{company.lower()}.com"
                ]
            },
            "compliance": {
                "gdpr_enabled": True,
                "data_retention_days": 2555,
                "audit_log_location": "/var/log/audit/",
                "pii_encryption_required": True
            }
        }

        return json.dumps(config, indent=2)

    def generate_database_schema_yaml(self, file_num: int) -> str:
        """Generate database schema in YAML format."""
        company = random.choice(self.companies)

        schema = {
            "database_name": f"{company.lower()}_production",
            "version": "2.1.0",
            "tables": {
                "employees": {
                    "columns": {
                        "id": {"type": "SERIAL", "primary_key": True},
                        "employee_id": {"type": "VARCHAR(20)", "unique": True},
                        "ssn": {"type": "VARCHAR(11)", "encrypted": True, "pii": True},
                        "first_name": {"type": "VARCHAR(50)", "pii": True},
                        "last_name": {"type": "VARCHAR(50)", "pii": True},
                        "email": {"type": "VARCHAR(100)", "pii": True},
                        "phone": {"type": "VARCHAR(15)", "pii": True},
                        "salary": {"type": "DECIMAL(10,2)", "confidential": True},
                        "department": {"type": "VARCHAR(50)"},
                        "hire_date": {"type": "DATE"},
                        "security_clearance": {"type": "INTEGER"}
                    },
                    "indexes": ["employee_id", "email", "department"],
                    "row_count": random.randint(500, 2000)
                },
                "financial_records": {
                    "columns": {
                        "id": {"type": "SERIAL", "primary_key": True},
                        "quarter": {"type": "VARCHAR(10)"},
                        "year": {"type": "INTEGER"},
                        "revenue": {"type": "DECIMAL(15,2)", "confidential": True},
                        "expenses": {"type": "DECIMAL(15,2)", "confidential": True},
                        "profit": {"type": "DECIMAL(15,2)", "confidential": True},
                        "bank_account": {"type": "VARCHAR(20)", "encrypted": True},
                        "tax_id": {"type": "VARCHAR(15)", "confidential": True}
                    },
                    "row_count": random.randint(20, 50)
                },
                "customer_data": {
                    "columns": {
                        "customer_id": {"type": "SERIAL", "primary_key": True},
                        "company_name": {"type": "VARCHAR(100)"},
                        "contact_email": {"type": "VARCHAR(100)", "pii": True},
                        "billing_address": {"type": "TEXT", "pii": True},
                        "credit_card_hash": {"type": "VARCHAR(64)", "encrypted": True, "pii": True},
                        "contract_value": {"type": "DECIMAL(12,2)", "confidential": True}
                    },
                    "row_count": random.randint(100, 500)
                }
            },
            "security": {
                "encryption_key": self.generate_api_key(),
                "backup_location": f"s3://{company.lower()}-backups/database/",
                "access_controls": {
                    "admin": ["SELECT", "INSERT", "UPDATE", "DELETE"],
                    "finance": ["SELECT on financial_records", "SELECT on employees.salary"],
                    "hr": ["SELECT", "UPDATE on employees"]
                }
            }
        }

        return yaml.dump(schema, default_flow_style=False)

    def generate_executive_email_eml(self, file_num: int) -> str:
        """Generate executive email in EML format."""
        company = random.choice(self.companies)
        ceo_name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
        cfo_name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"

        date = datetime.now() - timedelta(days=random.randint(1, 90))
        revenue_shortfall = random.randint(15, 25)
        churn_rate = random.randint(8, 15)
        acquisition_amount = random.randint(50, 150)
        bank_account = random.randint(100000000, 999999999)
        routing_number = random.randint(100000000, 999999999)
        strike_price = random.randint(15, 45)

        content = f"""From: {ceo_name.lower().replace(' ', '.')}@{company.lower()}.com
To: {cfo_name.lower().replace(' ', '.')}@{company.lower()}.com
Subject: CONFIDENTIAL - Q{random.randint(1,4)} Financial Review & Acquisition Discussion
Date: {date.strftime('%a, %d %b %Y %H:%M:%S')} -0800
Message-ID: <{random.randint(100000,999999)}.{random.randint(100000,999999)}@{company.lower()}.com>
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8

{cfo_name},

I've reviewed the preliminary financial numbers for this quarter, and I'm concerned about the revenue shortfall in our enterprise division. We're tracking at ${revenue_shortfall}M below projections.

Key concerns:
1. Customer churn rate has increased to {churn_rate}%
2. Our main competitor is undercutting us by 20-30% on new deals
3. The DataFlow acquisition is costing more than anticipated

CONFIDENTIAL ACQUISITION UPDATE:
We're moving forward with the TechStart acquisition for ${acquisition_amount}M. The due diligence revealed some concerning technical debt, but their customer base of 50,000+ users is too valuable to pass up.

Wire transfer details for the acquisition:
- Bank: First National Bank
- Account: {bank_account}
- Routing: {routing_number}
- Amount: ${acquisition_amount},000,000

Please ensure the board materials for next week include:
- Updated revenue forecasts
- Cost reduction scenarios (including potential layoffs)
- Integration timeline for TechStart
- Competitive analysis on pricing strategy

Also, I need you to personally handle the employee stock option repricing. We can't afford to lose our top talent to competitors. The new strike price should be ${strike_price}.00 per share.

Let's discuss this in person tomorrow. My calendar is clear after 3 PM.

Regards,
{ceo_name}
CEO, {company} Industries

P.S. - Please keep the acquisition details strictly confidential until we announce. Any leaks could impact the stock price significantly.

---
This email contains confidential and proprietary information. If you are not the intended recipient, please delete this email immediately.
"""
        return content

    def generate_org_chart_xml(self, file_num: int) -> str:
        """Generate organizational chart in XML format."""
        company = random.choice(self.companies)

        root = ElementTree.Element("organization")
        root.set("company", f"{company} Industries")
        root.set("classification", "CONFIDENTIAL")

        # CEO
        ceo = ElementTree.SubElement(root, "employee")
        ceo.set("level", "1")
        ceo.set("id", f"{company[:2].upper()}-2015-0001")
        ElementTree.SubElement(ceo, "name").text = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
        ElementTree.SubElement(ceo, "title").text = "Chief Executive Officer"
        ElementTree.SubElement(ceo, "email").text = f"ceo@{company.lower()}.com"
        ElementTree.SubElement(ceo, "salary").text = f"{random.randint(300,500)}000"
        ElementTree.SubElement(ceo, "clearance").text = "Level 5 - Executive"

        # VPs
        departments = ["Engineering", "Finance", "Sales", "Operations"]
        for i, dept in enumerate(departments):
            vp = ElementTree.SubElement(ceo, "direct_report")
            vp.set("level", "2")
            vp.set("id", f"{company[:2].upper()}-{random.randint(2016,2018)}-{100+i:04d}")
            ElementTree.SubElement(vp, "name").text = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            ElementTree.SubElement(vp, "title").text = f"VP of {dept}"
            ElementTree.SubElement(vp, "email").text = f"vp.{dept.lower()}@{company.lower()}.com"
            ElementTree.SubElement(vp, "salary").text = f"{random.randint(180,250)}000"
            ElementTree.SubElement(vp, "clearance").text = f"Level {random.randint(3,4)} - Executive"

            # Directors under each VP
            for j in range(random.randint(2, 4)):
                director = ElementTree.SubElement(vp, "direct_report")
                director.set("level", "3")
                director.set("id", f"{company[:2].upper()}-{random.randint(2018,2020)}-{200+i*10+j:04d}")
                director_name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
                ElementTree.SubElement(director, "name").text = director_name
                director_role = random.choice(['Product', 'Technology', 'Strategy', 'Operations'])
                ElementTree.SubElement(director, "title").text = f"Director of {director_role}"
                ElementTree.SubElement(director, "email").text = f"director{j+1}.{dept.lower()}@{company.lower()}.com"
                ElementTree.SubElement(director, "salary").text = f"{random.randint(120,180)}000"
                ElementTree.SubElement(director, "clearance").text = f"Level {random.randint(2,3)} - Management"

        return ElementTree.tostring(root, encoding='unicode')

    def generate_all_documents(self):
        """Generate 100 sensitive documents across all supported formats."""
        print(f"Generating 100 sensitive test documents in {self.output_dir}/")

        # Distribution of file types (based on supported formats)
        generators = [
            (self.generate_employee_record_txt, "txt", 15),
            (self.generate_financial_report_md, "md", 12),
            (self.generate_api_documentation_html, "html", 10),
            (self.generate_payroll_csv, "csv", 10),
            (self.generate_system_config_json, "json", 10),
            (self.generate_database_schema_yaml, "yaml", 8),
            (self.generate_executive_email_eml, "eml", 8),
            (self.generate_org_chart_xml, "xml", 7),
        ]

        # Generate additional formats to reach 100
        additional_formats = [
            (self.generate_employee_record_txt, "txt", 5),
            (self.generate_financial_report_md, "md", 5),
            (self.generate_api_documentation_html, "htm", 5),
            (self.generate_system_config_json, "json", 5),
        ]

        generators.extend(additional_formats)

        file_count = 0
        for generator_func, extension, count in generators:
            for i in range(count):
                file_count += 1
                filename = f"sensitive_doc_{file_count:03d}.{extension}"
                filepath = self.output_dir / filename

                content = generator_func(i)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"Generated: {filename}")

        print(f"\nSuccessfully generated {file_count} sensitive test documents!")
        print(f"Files saved to: {self.output_dir.absolute()}")

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate a summary of created documents."""
        summary_content = f"""# Generated Sensitive Test Documents

This directory contains 100 sensitive test documents generated for VectorSmuggle security testing.

## Document Types Generated

### Text Files (.txt) - 20 files
- Employee records with SSNs, salaries, and personal information
- Performance reviews and disciplinary actions
- Security clearance information

### Markdown Files (.md) - 17 files
- Confidential financial reports with revenue data
- Executive summaries and strategic information
- Bank account and tax ID information

### HTML Files (.html/.htm) - 15 files
- Internal API documentation with credentials
- Database connection strings and passwords
- Infrastructure details and IP addresses

### CSV Files (.csv) - 10 files
- Payroll data with SSNs and bank accounts
- Employee salary and tax withholding information
- Routing numbers and account details

### JSON Files (.json) - 15 files
- System configuration with API keys
- Database passwords and encryption keys
- AWS credentials and service endpoints

### YAML Files (.yaml) - 8 files
- Database schemas with PII column definitions
- Security configurations and access controls
- Encryption keys and backup locations

### Email Files (.eml) - 8 files
- Executive communications about acquisitions
- Financial discussions and wire transfer details
- Confidential strategic planning emails

### XML Files (.xml) - 7 files
- Organizational charts with salary information
- Employee hierarchy and security clearances
- Contact information and reporting structures

## Security Testing Use Cases

These documents are designed to test:
- PII detection (SSNs, phone numbers, addresses)
- Financial data identification (salaries, bank accounts)
- Credential detection (API keys, passwords, tokens)
- Infrastructure information (IP addresses, database connections)
- Organizational intelligence (employee hierarchies, strategic plans)

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**WARNING**: These documents contain realistic-looking sensitive data for testing purposes only. All data is fabricated and should not be used for any purpose other than security testing.
"""

        summary_path = self.output_dir / "README.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        print(f"Generated summary: {summary_path}")


if __name__ == "__main__":
    generator = SensitiveDocumentGenerator()
    generator.generate_all_documents()
