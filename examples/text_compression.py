"""
Example: Text Prompt Compression — All Modes
==============================================
Shows how LatentGate compresses text prompts locally before
sending to cloud LLMs, saving ~60-80% on token costs.

Prerequisites:
    ollama pull llama3:8b
"""

from latent_gate import LatentGatePipeline, PipelineConfig


def setup():
    """Create a pipeline (fully local for demo)."""
    config = PipelineConfig(
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
        log_level="WARNING",
    )
    return LatentGatePipeline(config)


# ============================================================================
# 1. LONG PROMPT COMPRESSION
# ============================================================================
def example_long_prompt():
    """Compress a verbose prompt before sending to LLM."""
    pipeline = setup()

    long_prompt = """
    I'm working on a Python web application using Flask framework and I need
    help with something. The application is a REST API that handles user
    authentication using JWT tokens. I've been running into issues with token
    refresh logic. Specifically, when a user's access token expires, the
    refresh endpoint should generate a new access token using the refresh
    token, but it seems like the refresh token is also being invalidated
    somehow. I'm using PyJWT library version 2.8.0 and Flask 3.0.2. The
    database is PostgreSQL with SQLAlchemy as the ORM. I've set the access
    token expiry to 15 minutes and refresh token to 7 days. The issue only
    happens in production, not in my local development environment. I think
    it might be related to the SECRET_KEY configuration or possibly a race
    condition when multiple requests hit the refresh endpoint simultaneously.
    Can you help me debug this? I need a solution that handles concurrent
    refresh requests gracefully and doesn't invalidate tokens prematurely.
    Also, I want to implement token rotation where each refresh generates a
    new refresh token and invalidates the old one, but in a way that handles
    the race condition I mentioned. Please provide code examples.
    """

    result = pipeline.query_text(long_prompt)

    print("=" * 60)
    print("LONG PROMPT COMPRESSION")
    print("=" * 60)
    print(f"Original:   ~{result.get('original_tokens', 'N/A')} tokens")
    print(f"Compressed: ~{result['tokens_estimated']} tokens")
    print(f"Savings:    {result.get('compression_ratio', 'N/A')}")
    print(f"\nCompact prompt sent to LLM:")
    print(f"  {result['compact_prompt'][:200]}...")
    print(f"\nAnswer: {result['answer'][:300]}...")


# ============================================================================
# 2. CONVERSATION HISTORY COMPRESSION
# ============================================================================
def example_conversation():
    """Compress growing conversation history."""
    pipeline = setup()

    messages = [
        {"role": "user", "content": "Hi, I need help setting up a Kubernetes cluster on AWS using EKS."},
        {"role": "assistant", "content": "Sure! I can help with that. First, you'll need the AWS CLI configured and eksctl installed. What's your target setup - how many nodes, instance types, and region?"},
        {"role": "user", "content": "I want 3 worker nodes, t3.large instances, in us-east-1. Also need to set up an Application Load Balancer and configure auto-scaling from 3 to 10 nodes based on CPU usage."},
        {"role": "assistant", "content": "Great setup! Here's the plan: 1) Create EKS cluster with eksctl, 2) Configure node group with t3.large, 3) Install AWS Load Balancer Controller, 4) Set up Cluster Autoscaler. Let me start with the cluster creation command..."},
        {"role": "user", "content": "Actually, I also need to set up monitoring with Prometheus and Grafana, and I want to use Helm for package management. Oh, and we need to configure RBAC for three teams: dev, staging, and prod."},
        {"role": "assistant", "content": "Good additions. So the full plan is now: EKS cluster, ALB, autoscaling, Prometheus+Grafana monitoring via Helm, and RBAC for 3 teams. I'll add kube-prometheus-stack Helm chart for monitoring. For RBAC, we'll create namespaces and role bindings per team."},
        {"role": "user", "content": "Perfect. Let's also add Istio service mesh for traffic management between microservices."},
    ]

    result = pipeline.query_conversation(
        messages=messages,
        new_question="Now give me the complete step-by-step setup commands starting from cluster creation."
    )

    print("\n" + "=" * 60)
    print("CONVERSATION COMPRESSION")
    print("=" * 60)
    print(f"Original conversation: ~{result.get('original_tokens', 'N/A')} tokens")
    print(f"Compressed to:         ~{result['tokens_estimated']} tokens")
    print(f"Savings:               {result.get('compression_ratio', 'N/A')}")
    print(f"\nAnswer: {result['answer'][:300]}...")


# ============================================================================
# 3. RAG DOCUMENT COMPRESSION
# ============================================================================
def example_rag_documents():
    """Compress retrieved documents before sending to LLM."""
    pipeline = setup()

    documents = [
        "Flask Documentation - Chapter 12: Authentication. Flask provides several extensions for authentication. Flask-Login manages user sessions and provides a user_loader callback. Flask-JWT-Extended supports JWT tokens with features like token freshness, token blocklisting, and custom claims. Installation: pip install flask-jwt-extended. Basic usage requires setting JWT_SECRET_KEY in app config...",
        "Stack Overflow Answer (Score: 847): JWT Token Refresh Best Practices. When implementing JWT refresh, always use token rotation - issue a new refresh token with each access token refresh. Store refresh tokens in a database table with columns: id, user_id, token_hash, expires_at, is_revoked. On refresh: 1) Verify refresh token exists and is not revoked, 2) Issue new access + refresh tokens, 3) Revoke the old refresh token. For race conditions, use database-level locking...",
        "PyJWT Documentation v2.8.0. PyJWT is a Python library for encoding and decoding JSON Web Tokens. Key functions: jwt.encode(payload, key, algorithm), jwt.decode(token, key, algorithms). Supported algorithms: HS256, HS384, HS512, RS256, RS384, RS512. Common errors: ExpiredSignatureError when token is expired, InvalidTokenError for malformed tokens. The library does NOT handle token storage or refresh logic - that must be implemented by the application...",
    ]

    result = pipeline.query_documents(
        documents=documents,
        question="How do I implement JWT token refresh with rotation in Flask?"
    )

    print("\n" + "=" * 60)
    print("RAG DOCUMENT COMPRESSION")
    print("=" * 60)
    print(f"Original docs:  ~{result.get('original_tokens', 'N/A')} tokens ({len(documents)} docs)")
    print(f"Compressed to:  ~{result['tokens_estimated']} tokens")
    print(f"Savings:        {result.get('compression_ratio', 'N/A')}")
    print(f"\nAnswer: {result['answer'][:300]}...")


# ============================================================================
# 4. CODE PROMPT COMPRESSION
# ============================================================================
def example_code_prompt():
    """Compress a code-heavy prompt."""
    pipeline = setup()

    code_prompt = """
    I have this Python code and it's running very slowly on large datasets.
    The function processes a list of transactions and groups them by category,
    then calculates running totals. It works correctly but takes over 30
    seconds for 1 million records. I need it under 2 seconds.

    ```python
    def process_transactions(transactions):
        result = {}
        for t in transactions:
            cat = t['category']
            if cat not in result:
                result[cat] = {'items': [], 'total': 0}
            result[cat]['items'].append(t)
            result[cat]['total'] += t['amount']

        for cat in result:
            result[cat]['items'].sort(key=lambda x: x['date'])
            running = 0
            for item in result[cat]['items']:
                running += item['amount']
                item['running_total'] = running

        return result
    ```

    The dataset has about 50 unique categories. Each transaction has:
    category (string), amount (float), date (datetime), description (string).
    I'm running Python 3.11 on a machine with 32GB RAM.
    Please optimize this for speed. Can we use pandas or numpy?
    """

    result = pipeline.query_text(code_prompt, mode="code")

    print("\n" + "=" * 60)
    print("CODE PROMPT COMPRESSION")
    print("=" * 60)
    print(f"Original:   ~{result.get('original_tokens', 'N/A')} tokens")
    print(f"Compressed: ~{result['tokens_estimated']} tokens")
    print(f"Savings:    {result.get('compression_ratio', 'N/A')}")
    print(f"\nAnswer: {result['answer'][:300]}...")


if __name__ == "__main__":
    example_long_prompt()
    example_conversation()
    example_rag_documents()
    example_code_prompt()
