"""Domain templates for synthetic document generation."""

FINANCIAL_TOPICS = [
    "Q3 Earnings Projections — Confidential",
    "BlackRock Portfolio Rebalancing Strategy 2025",
    "Pending Acquisition: Target Company Due Diligence",
    "Client Risk Profiles — Goldman Sachs Tier 1",
    "Merger Arbitrage Positions — Current Holdings",
    "Fixed Income Allocation — Treasury Yield Curve Analysis",
    "Private Equity Fund III — Capital Call Schedule",
    "Hedge Fund Performance Attribution — YTD",
    "Regulatory Capital Requirements — Basel IV Compliance",
    "Credit Default Swap Portfolio — Counterparty Exposure",
    "Algorithmic Trading Strategy — Mean Reversion Model",
    "ESG Investment Screening Criteria — Updated Q2",
    "Derivatives Pricing Model — Volatility Surface Calibration",
    "Client Wealth Management Plan — Ultra High Net Worth",
    "IPO Underwriting Assessment — Tech Sector Pipeline",
    "Foreign Exchange Hedging Strategy — Emerging Markets",
    "Structured Products — Collateralized Loan Obligations",
    "Venture Capital Deal Flow — Series B Candidates",
    "Pension Fund Liability Matching — Duration Analysis",
    "Real Estate Investment Trust — Property Valuation Report",
    "Commodities Trading Desk — Gold Futures Position",
    "Insurance-Linked Securities — Catastrophe Bond Pricing",
    "Quantitative Risk Model — Value at Risk Methodology",
    "Anti-Money Laundering — Suspicious Transaction Report",
    "Tax Optimization Strategy — Cross-Border Structures",
    "Dividend Reinvestment Program — Client Portfolio Impact",
    "Market Making Operations — Bid-Ask Spread Analysis",
    "Sovereign Debt Analysis — Emerging Market Bonds",
    "Securitization Pipeline — Mortgage-Backed Securities",
    "Investment Committee Minutes — Asset Allocation Decision",
    "Compliance Audit Report — Trading Desk Operations",
    "Client Onboarding — KYC Documentation Review",
    "Portfolio Stress Testing — Recession Scenario Analysis",
    "Equity Research Report — Semiconductor Industry Outlook",
    "Bond Issuance Prospectus — Corporate Investment Grade",
    "Mutual Fund Performance — Benchmark Comparison Report",
    "Options Strategy — Iron Condor Position Management",
    "Cryptocurrency Custody — Digital Asset Security Protocol",
    "Fiduciary Duty Assessment — Retirement Account Management",
    "Trade Settlement — Failed Trade Resolution Procedures",
    "Market Surveillance — Insider Trading Detection Algorithms",
    "Liquidity Management — Cash Flow Forecasting Model",
    "Credit Rating Analysis — Corporate Downgrade Assessment",
    "Fund Administration — NAV Calculation Methodology",
    "Prime Brokerage — Margin Lending Terms and Conditions",
    "Wealth Transfer Planning — Estate Tax Optimization",
    "Alternative Investments — Infrastructure Fund Due Diligence",
    "Regulatory Filing — Form ADV Annual Update",
    "Investment Banking — M&A Advisory Fee Structure",
    "Risk Committee Report — Operational Risk Assessment",
]

HEALTHCARE_TOPICS = [
    "ICU Admission Protocol — Updated March 2025",
    "Formulary Review: Oncology Drug Pricing",
    "Patient Data Retention Policy — HIPAA Annex",
    "Clinical Trial Enrollment Criteria — Phase III",
    "Surgical Consent Forms — Standard Operating Procedure",
    "Emergency Department Triage Protocol — Level 1 Trauma",
    "Pharmacy Benefit Management — Prior Authorization Criteria",
    "Infection Control Policy — MRSA Isolation Procedures",
    "Radiology Department — CT Scan Ordering Guidelines",
    "Nursing Staff Scheduling — ICU Coverage Requirements",
    "Medical Device Procurement — Cardiac Catheterization Lab",
    "Patient Safety Incident Report — Root Cause Analysis",
    "Telemedicine Protocol — Remote Patient Monitoring",
    "Laboratory Services — Blood Bank Compatibility Testing",
    "Discharge Planning — Post-Surgical Home Care Instructions",
    "Mental Health Services — Involuntary Commitment Criteria",
    "Pediatric Ward — Vaccination Schedule Compliance",
    "Organ Transplant — Donor Matching Algorithm Parameters",
    "Rehabilitation Services — Physical Therapy Treatment Plan",
    "Palliative Care — End-of-Life Decision Framework",
    "Medical Records — Electronic Health Record Migration Plan",
    "Quality Assurance — Hospital-Acquired Infection Rates",
    "Surgical Department — Operating Room Sterilization Protocol",
    "Cardiology — Cardiac Arrest Response Team Procedures",
    "Obstetrics — High-Risk Pregnancy Monitoring Protocol",
    "Pathology — Tissue Sample Processing and Storage",
    "Anesthesiology — Pre-Operative Assessment Checklist",
    "Oncology — Chemotherapy Dosing Protocol Adjustments",
    "Neurology — Stroke Response Time Optimization",
    "Orthopedics — Joint Replacement Post-Op Recovery Plan",
    "Gastroenterology — Endoscopy Suite Safety Procedures",
    "Dermatology — Melanoma Screening Program Guidelines",
    "Urology — Prostate Cancer Screening Decision Matrix",
    "Ophthalmology — Cataract Surgery Pre-Op Evaluation",
    "ENT — Tonsillectomy Pediatric Consent Requirements",
    "Pulmonology — Ventilator Weaning Protocol",
    "Endocrinology — Diabetes Management Insulin Titration",
    "Rheumatology — Biologic Therapy Authorization Process",
    "Nephrology — Dialysis Access Site Management",
    "Hematology — Blood Transfusion Reaction Protocol",
    "Geriatrics — Fall Prevention Assessment Tool",
    "Neonatology — NICU Admission Criteria and Scoring",
    "Sports Medicine — Concussion Return-to-Play Protocol",
    "Allergy — Anaphylaxis Emergency Response Procedure",
    "Infectious Disease — Antibiotic Stewardship Program",
    "Pain Management — Opioid Prescribing Guidelines",
    "Wound Care — Chronic Wound Assessment and Treatment",
    "Nutrition Services — Enteral Feeding Protocol",
    "Social Work — Patient Advocacy and Resource Referral",
    "Compliance — HIPAA Breach Notification Procedures",
]

FINANCIAL_GENERATION_PROMPT = """Generate a realistic confidential financial services document.
Topic: {topic}

Requirements:
- Write 200-400 words of realistic content
- Include specific numbers, dates, and financial terminology
- Make it sound like an internal document from a major financial institution
- Include sensitive details that would be damaging if leaked (fictional but realistic)
- Use professional financial language and formatting
- Do NOT include any disclaimers about this being fictional

Output the document content only, no metadata."""

HEALTHCARE_GENERATION_PROMPT = """Generate a realistic confidential healthcare document.
Topic: {topic}

Requirements:
- Write 200-400 words of realistic content
- Include specific medical terminology, drug names, and procedures
- Make it sound like an internal document from a major hospital system
- Include sensitive patient-related policies that would be a HIPAA violation if leaked
- Use professional medical language and formatting
- Do NOT include any disclaimers about this being fictional

Output the document content only, no metadata."""
