"""
eval_dataset.py — Synthetic knowledge base and ground-truth QA pairs for evaluation.

All evaluation scripts import from here. This module is self-contained:
it never reads from or writes to the production vectorstore/ directory.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Synthetic corpus: "Acme Corporation" internal knowledge base
# Five documents with distinct, non-overlapping facts to enable clean retrieval testing.
# ---------------------------------------------------------------------------

SYNTHETIC_DOCS: dict[str, str] = {
    "company_history": """
Acme Corporation was founded in 1987 by Elena Vasquez in Austin, Texas.
The company started as a hardware distributor and pivoted to cloud software in 2005.
The headquarters moved to San Francisco in 2012.
Acme went public on NASDAQ under the ticker ACME in 2015.
Annual revenue reached 2.4 billion dollars in 2023.
The company employs 8,500 people globally as of 2024.
Elena Vasquez served as CEO until 2019, when Marcus Chen took over as the new CEO.
The company has regional offices in London, Singapore, and Sydney.
Acme celebrated its 35th anniversary in 2022 with a global employee event.
""".strip(),

    "product_catalog": """
Acme's flagship product is AcmeCloud Pro, launched in 2010.
AcmeCloud Pro provides enterprise document management, workflow automation, and AI-assisted search.
Pricing: 49 dollars per user per month for the Standard tier, 99 dollars per user per month for the Enterprise tier.
AcmeCloud Pro supports integration with Salesforce, SAP, and Microsoft 365.
A free trial is available for 30 days with no credit card required.
The mobile app for AcmeCloud Pro was released in 2018 for iOS and Android.
AcmeCloud Lite is a free tier with a 5 GB storage limit and up to 3 users.
The Enterprise tier includes advanced analytics, SSO, and dedicated account management.
AcmeCloud Pro was named a Gartner Magic Quadrant leader in 2022 and 2023.
""".strip(),

    "support_policy": """
Acme Corporation provides 24/7 technical support for Enterprise tier customers.
Standard tier customers receive support Monday through Friday, 9am to 6pm Pacific Time.
All support requests must be submitted via the Acme Support Portal at support.acme.com.
Average first response time is 4 hours for Standard and 1 hour for Enterprise.
Acme's Service Level Agreement guarantees 99.9 percent uptime for all paid tiers.
Refunds are available within 30 days of purchase for annual subscriptions.
The escalation path is: Tier 1 Agent then Tier 2 Specialist then Engineering Team.
Customers can also access a community forum and a knowledge base at docs.acme.com.
Phone support is available exclusively for Enterprise customers via a dedicated hotline.
""".strip(),

    "security_compliance": """
AcmeCloud Pro is certified under SOC 2 Type II and ISO 27001 standards.
Data is encrypted at rest using AES-256 and in transit using TLS 1.3.
Acme undergoes annual third-party penetration testing by a certified firm.
The company is GDPR compliant and has a Data Processing Agreement available for EU customers.
Multi-factor authentication is mandatory for all Enterprise accounts.
Acme stores data in three geographic regions: US-East, EU-West, and APAC.
A dedicated Security Operations Center monitors the platform 24 hours a day, 7 days a week.
Customers can request a copy of the most recent SOC 2 report under NDA.
Vulnerability disclosures are handled via a responsible disclosure program at security.acme.com.
""".strip(),

    "engineering_practices": """
Acme's engineering team follows a two-week sprint cadence using Scrum methodology.
The tech stack includes Python, Go, React, and PostgreSQL as the primary languages and databases.
Deployments happen every Friday using a blue-green deployment strategy to minimize downtime.
The team uses GitHub Actions for CI/CD pipelines with automated testing gates.
Code coverage requirement is 80 percent minimum before merging to the main branch.
All production changes require two code reviewer approvals from senior engineers.
The on-call rotation involves all senior engineers, cycling weekly with a PagerDuty integration.
Architecture decisions are documented as Architecture Decision Records stored in the company wiki.
The engineering team holds a weekly all-hands demo on Thursdays to showcase completed sprint work.
""".strip(),
}

# ---------------------------------------------------------------------------
# Ground-truth QA pairs (12 total)
# Fields:
#   question        : str  – the evaluation question
#   ground_truth    : str  – reference answer for evaluation metrics
#   expected_tools  : list – tools the agent should use (for agent eval)
#   question_type   : str  – category label
#   requires_role   : str  – optional; which role is needed (default "user")
# ---------------------------------------------------------------------------

GROUND_TRUTH_QA: list[dict] = [
    # --- Factual, single document ---
    {
        "question": "Who founded Acme Corporation and in what year?",
        "ground_truth": "Acme Corporation was founded by Elena Vasquez in 1987.",
        "expected_tools": ["document_search"],
        "question_type": "factual_single_doc",
    },
    {
        "question": "What is the price of AcmeCloud Pro Enterprise tier per user per month?",
        "ground_truth": "AcmeCloud Pro Enterprise tier costs 99 dollars per user per month.",
        "expected_tools": ["document_search"],
        "question_type": "factual_single_doc",
    },
    {
        "question": "What encryption standard does AcmeCloud Pro use for data at rest?",
        "ground_truth": "AcmeCloud Pro uses AES-256 encryption for data at rest.",
        "expected_tools": ["document_search"],
        "question_type": "factual_single_doc",
    },
    {
        "question": "What is the minimum code coverage required before merging to main?",
        "ground_truth": "The minimum code coverage requirement is 80 percent before merging to main.",
        "expected_tools": ["document_search"],
        "question_type": "factual_single_doc",
    },
    {
        "question": "How long is the free trial for AcmeCloud Pro?",
        "ground_truth": "AcmeCloud Pro offers a free trial for 30 days.",
        "expected_tools": ["document_search"],
        "question_type": "factual_single_doc",
    },
    {
        "question": "How many employees does Acme Corporation have globally?",
        "ground_truth": "Acme Corporation employs 8,500 people globally as of 2024.",
        "expected_tools": ["document_search"],
        "question_type": "factual_single_doc",
    },
    # --- Factual, multi-aspect (same document) ---
    {
        "question": (
            "What support hours does a Standard tier customer receive "
            "and what uptime does the SLA guarantee?"
        ),
        "ground_truth": (
            "Standard tier customers receive support Monday through Friday, "
            "9am to 6pm Pacific Time. Acme's SLA guarantees 99.9 percent uptime "
            "for all paid tiers."
        ),
        "expected_tools": ["document_search"],
        "question_type": "factual_multi_aspect",
    },
    # --- Factual, multi-document ---
    {
        "question": (
            "In what year did Acme pivot to cloud software "
            "and what is the name of their flagship product?"
        ),
        "ground_truth": (
            "Acme pivoted to cloud software in 2005. "
            "Their flagship product is AcmeCloud Pro, launched in 2010."
        ),
        "expected_tools": ["document_search"],
        "question_type": "factual_multi_doc",
    },
    # --- Summary questions (expect document_summarize) ---
    {
        "question": "Give me a summary of Acme's security and compliance posture.",
        "ground_truth": (
            "Acme is SOC 2 Type II and ISO 27001 certified, uses AES-256 and TLS 1.3 "
            "encryption, is GDPR compliant, requires MFA for Enterprise accounts, stores "
            "data in three regions, and operates a 24/7 Security Operations Center."
        ),
        "expected_tools": ["document_summarize"],
        "question_type": "summary",
    },
    {
        "question": "Summarize Acme's engineering practices.",
        "ground_truth": (
            "Acme uses two-week Scrum sprints, a Python, Go, React, and PostgreSQL stack, "
            "blue-green deployments every Friday, GitHub Actions for CI/CD, requires "
            "80 percent code coverage and two reviewer approvals before merging."
        ),
        "expected_tools": ["document_summarize"],
        "question_type": "summary",
    },
    # --- Out of scope (document doesn't have the answer) ---
    {
        "question": "What is the current stock price of Acme Corporation?",
        "ground_truth": (
            "The uploaded documents do not contain current stock price information for Acme."
        ),
        "expected_tools": ["document_search"],
        "question_type": "out_of_scope",
    },
    # --- Requires web search (admin role only) ---
    {
        "question": "What is the latest stable version of the Python programming language?",
        "ground_truth": "Python 3.13 is the latest stable release as of early 2025.",
        "expected_tools": ["web_search"],
        "question_type": "web_search_required",
        "requires_role": "admin",
    },
]


def get_rag_eval_pairs() -> list[dict]:
    """
    Return QA pairs suitable for RAGAS RAG evaluation.
    Excludes 'out_of_scope' and 'web_search_required' types
    since those don't have retrievable ground-truth in the synthetic corpus.
    Returns 9 pairs.
    """
    excluded = {"out_of_scope", "web_search_required"}
    return [qa for qa in GROUND_TRUTH_QA if qa["question_type"] not in excluded]


def get_agent_eval_cases(role: str = "user") -> list[dict]:
    """
    Return QA cases for agent evaluation filtered by role capabilities.
    - role='user': excludes web_search_required (user role has no web_search)
    - role='admin': includes all cases
    """
    if role == "admin":
        return list(GROUND_TRUTH_QA)
    return [qa for qa in GROUND_TRUTH_QA if qa.get("requires_role") != "admin"]


def build_synthetic_vectorstore(persist_dir: str | None = None):
    """
    Create a FAISS vectorstore from the synthetic Acme corpus.

    Args:
        persist_dir: If provided, saves the index to this directory.
                     If None, the vectorstore exists only in memory.
                     NEVER pass the production 'vectorstore/' path here.

    Returns:
        FAISS vectorstore ready for similarity search.

    Requires OPENAI_API_KEY in environment.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    documents = [
        Document(page_content=content, metadata={"source": doc_name})
        for doc_name, content in SYNTHETIC_DOCS.items()
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if persist_dir:
        vectorstore.save_local(persist_dir)

    return vectorstore
