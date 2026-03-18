"""
Stores structured resume JSON for all active profiles directly into adm.resume.resume_json.
No LLM API required — JSON is hand-crafted from resume content.
"""
import json
import os
import psycopg2
import psycopg2.extras

DB_DSN = os.environ["DATABASE_URL"]

RESUME_JSONS = {
    "Slava": {
        "name": "Slava",
        "current_title": "Lead Data Engineer",
        "years_experience": 9,
        "location": "Toronto, ON",
        "seniority": "lead",
        "summary": "Results-driven Lead Data Engineer at MLSE with 9 years experience. Expertise in AWS, dbt, Python, PySpark. Achieved 100x pipeline performance improvement. Innovator Award 2025.",
        "skills": {
            "languages": ["Python", "SQL", "PySpark", "Shell Scripting", "C#"],
            "frameworks_tools": ["dbt", "Informatica", "AWS Glue", "MageAI", "Qlik", "SSRS", "CRM Analytics", "SQLdbm"],
            "cloud_infra": ["AWS Lambda", "AWS Glue", "S3", "Athena", "ECR", "Redshift", "DynamoDB", "Terraform", "Docker", "Kubernetes"],
            "databases": ["PostgreSQL", "MS SQL", "Snowflake", "Redshift", "DynamoDB"],
            "orchestration": ["Dagster", "Prefect", "MageAI"],
            "methodologies": ["Dimensional Modeling", "Kimball", "Star Schema", "Data Warehouse", "Data Lake", "IaC"]
        },
        "experience": [
            {
                "company": "MLSE",
                "title": "Lead Data Engineer",
                "duration_years": 1,
                "key_achievements": [
                    "100x performance improvement on Ticketmaster pipeline (hours → <10 min, 15K records/sec)",
                    "Production platform processing 1M+ rows/hour powering NBA/NHL/MLS reporting",
                    "Innovator Award 2025"
                ]
            },
            {
                "company": "MLSE",
                "title": "Data Engineer",
                "duration_years": 4,
                "key_achievements": [
                    "Redesigned ticketing data warehouse with dbt and star schema",
                    "Integrated diverse sources via AWS Glue, Lambda, Informatica",
                    "Mentored junior engineers, established style guides"
                ]
            },
            {
                "company": "4 Finance",
                "title": "Data Engineer",
                "duration_years": 2,
                "key_achievements": [
                    "Built company-wide data warehouse using Kimball methodology",
                    "Led Qlik BI platform implementation",
                    "Integrated REST APIs, LDAP, flat files into unified data ecosystem"
                ]
            },
            {
                "company": "Harris Corporation (Advanced Utility)",
                "title": "Data Conversion Specialist",
                "duration_years": 1,
                "key_achievements": [
                    "ETL pipelines in C#/SSIS managing 500M+ row datasets",
                    "Source-to-target mapping documentation"
                ]
            }
        ],
        "education": [
            {"degree": "BA Economics", "school": "McMaster University"}
        ],
        "certifications": [],
        "notes": []
    },

    "Kezia": {
        "name": "Kezia",
        "current_title": "Technical Business Analyst",
        "years_experience": 5,
        "location": "Toronto, ON",
        "seniority": "mid",
        "summary": "Business Systems Analyst and MSc Digital Management graduate with 4+ years at MLSE, Sun Life, CGI. Specializes in CRM, Salesforce, Agile delivery, and translating business requirements into system solutions.",
        "skills": {
            "languages": ["SQL"],
            "frameworks_tools": ["Salesforce CRM", "Salesforce Marketing Cloud", "JIRA", "Confluence", "CRM Analytics"],
            "cloud_infra": [],
            "domains": ["CRM", "ecommerce", "Enterprise Data Architecture", "Fan Engagement", "Financial Services"],
            "methodologies": ["Agile", "Scrum", "SDLC", "UAT", "QA", "Business Requirements Documentation", "Process Mapping", "User Stories", "Defect Triage", "Backlog Refinement"]
        },
        "experience": [
            {
                "company": "MLSE",
                "title": "Technical Business Analyst",
                "duration_years": 1,
                "key_achievements": [
                    "Led Fan Loyalty mobile platform release increasing NHL fan engagement 25%",
                    "Designed CRM Analytics dashboards for ticketing performance reporting",
                    "Evaluated generative AI opportunities for CRM workflow modernization"
                ]
            },
            {
                "company": "Sun Life Financial",
                "title": "Business Systems Analyst",
                "duration_years": 1,
                "key_achievements": [
                    "Led RFQ evaluation of Generative AI curriculum vendors for enterprise AI adoption",
                    "Delivered full SDLC solutions for Enterprise Data Architecture team",
                    "Improved sprint tracking efficiency 20% via Salesforce sprint boards"
                ]
            },
            {
                "company": "CGI Inc.",
                "title": "Project Control Officer",
                "duration_years": 1,
                "key_achievements": [
                    "Streamlined client workflows via Jira/Confluence, increased throughput 25%",
                    "Redesigned triage workflows, improved incident resolution 30%"
                ]
            },
            {
                "company": "Blackberry Ltd.",
                "title": "Quality Assurance Analyst",
                "duration_years": 1,
                "key_achievements": [
                    "Reduced repeat support cases 20% via process improvements",
                    "Built reporting dashboards tracking customer interaction trends"
                ]
            }
        ],
        "education": [
            {"degree": "MSc Digital Management", "school": "Ivey Business School, Western University"},
            {"degree": "BES Environment & Business, Minor Economics", "school": "University of Waterloo — Dean's Honours List"}
        ],
        "certifications": ["Salesforce Admin (In Progress)"],
        "notes": []
    },

    "Ray": {
        "name": "Ray",
        "current_title": "Director of Finance",
        "years_experience": 12,
        "location": "Toronto, ON",
        "seniority": "director",
        "summary": "Finance leader with 12+ years in real estate, construction, hospitality, and financial services. Strong in full-cycle financial operations, team leadership, and system implementation. No CPA.",
        "skills": {
            "languages": [],
            "frameworks_tools": ["Yardi", "JIRA", "Confluence"],
            "cloud_infra": [],
            "domains": ["Real Estate", "Construction", "Hospitality", "Financial Services", "Revenue Accounting", "AP/AR", "Payroll"],
            "methodologies": ["Cash Flow Optimization", "Budgeting", "Forecasting", "Financial Reporting", "Audit Preparation", "Internal Controls", "GAAP"]
        },
        "experience": [
            {
                "company": "Freed Corp.",
                "title": "Director of Finance",
                "duration_years": 3,
                "key_achievements": [
                    "Oversees financial operations across real estate, construction, development, and hospitality subsidiaries",
                    "Secured funding for major development projects, negotiated with financial institutions",
                    "Led financial system integration improving transparency across all entities"
                ]
            },
            {
                "company": "Morguard Investments",
                "title": "Supervisor, Revenue Accounting",
                "duration_years": 4,
                "key_achievements": [
                    "Reduced month-end closing timelines 20%",
                    "Led Yardi financial system transition and annual upgrades"
                ]
            },
            {
                "company": "Firmex Canada Inc.",
                "title": "Accounting Manager",
                "duration_years": 1,
                "key_achievements": [
                    "Full cycle accounting including GL, AP, AR, payroll",
                    "Monthly/quarterly/annual close processes"
                ]
            },
            {
                "company": "Cash Money Cheque Cashing",
                "title": "Accounting Supervisor",
                "duration_years": 3,
                "key_achievements": [
                    "Supervised AP team and 10 accounting assistants",
                    "Reduced month-end close 15%, managed corporate store leases"
                ]
            }
        ],
        "education": [
            {"degree": "Business Accounting Diploma", "school": "Humber College"}
        ],
        "certifications": [],
        "notes": ["Does not have CPA or BA — rank jobs requiring CPA significantly lower"]
    }
}


def main():
    db = psycopg2.connect(DB_DSN)
    db.autocommit = True

    for profile, resume_json in RESUME_JSONS.items():
        with db.cursor() as cur:
            cur.execute(
                "UPDATE adm.resume SET resume_json = %s WHERE profile = %s AND is_active = TRUE",
                (psycopg2.extras.Json(resume_json), profile),
            )
            if cur.rowcount:
                size = len(json.dumps(resume_json))
                print(f"  {profile}: stored ({size} chars vs ~{size * 5} raw estimate)")
            else:
                print(f"  {profile}: not found or inactive, skipped")

    print("\nDone.")


if __name__ == "__main__":
    main()
