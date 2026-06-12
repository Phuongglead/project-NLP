import os
os.environ["SA_AQG_USE_STUBS"] = "true"

from src.pipeline.runner import run_pipeline_batch

samples = [
    {
        "cv_text": """Software Engineer with 4 years of experience in backend development and cloud systems. 
Proficient in Python, Java, and Go. Experienced with Kubernetes, Docker, AWS, and CI/CD pipelines.

Worked on microservices architecture for high-scale e-commerce systems, handling millions of requests per day.
Implemented REST APIs using FastAPI and Spring Boot.

Strong knowledge of distributed systems, database design (PostgreSQL, MongoDB), and message queues (Kafka, RabbitMQ).

Familiar with machine learning pipelines for recommendation systems and data processing workflows.""",

        "job_description": """We are looking for a Backend Engineer to join our cloud infrastructure team.

Responsibilities:
- Design and implement scalable backend services using Python or Java
- Build and maintain microservices architecture
- Work with Kubernetes and Docker for container orchestration
- Develop RESTful APIs and integrate with distributed systems
- Collaborate with data engineers to support ML pipelines

Requirements:
- 3+ years experience in backend development
- Strong knowledge of cloud platforms (AWS/GCP/Azure)
- Experience with microservices and distributed systems
- Familiar with CI/CD and DevOps practices
- Good understanding of databases and message queues"""
    }
]

results = run_pipeline_batch(samples)

for r in results:
    print("\n====================")
    print("ID:", r.id)

    print("\n[NER]")
    print(r.skills)

    print("\n[GEN QUESTION]")
    print(r.generated_question)

    print("\n[RAG]")
    print(r.reference_answer)

    print("\n[METRICS - XAI]")
    if r.evaluation:
        print("NLI:", r.evaluation.nli_label)
        print("Citation Precision:", r.evaluation.citation_precision)
        print("SHAP CV ratio:", r.evaluation.shap_cv_ratio)