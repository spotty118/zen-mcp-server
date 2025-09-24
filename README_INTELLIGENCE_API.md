# Zen Intelligence API - Enterprise-Grade Multi-Model Orchestration

## Overview

The Zen Intelligence API is an advanced OpenAI-compatible REST API that implements sophisticated multi-model orchestration with intelligent decision-making, adaptive routing, and enterprise-grade capabilities. It extends the existing Zen MCP server with a powerful HTTP API layer that enables enterprise applications to leverage multiple AI models intelligently.

## Key Features

### 🧠 Advanced Intelligence Engine
- **Dynamic Model Selection**: Intelligently routes queries to optimal models based on task complexity and domain expertise
- **Adaptive Consensus Building**: Uses weighted confidence scoring and iterative refinement across models
- **Context-Aware Routing**: Analyzes conversation history to determine which models performed best for similar tasks
- **Real-time Performance Optimization**: Tracks model response times and quality, automatically adjusting strategies

### 🚀 Smart Decision Matrix
```
Query Type → Model Assignments:
├── Code Architecture → GPT-4/5 (primary) + Claude (review) + Gemini (validation)
├── Debugging → Gemini (primary) + GPT (optimization) + Claude (synthesis)  
├── Security Analysis → Claude (primary) + GPT (patterns) + Gemini (edge cases)
├── Performance Optimization → GPT (primary) + Claude (best practices) + Gemini (alternatives)
├── Code Review → All models + weighted consensus + confidence scoring
└── Complex Problem Solving → Multi-layer iterative refinement with model specialization
```

### 🔄 Orchestration Strategies
- **Single Model**: Fast responses for simple queries
- **Parallel Processing**: Multiple models work simultaneously
- **Sequential Refinement**: Iterative improvement across models
- **Consensus Building**: Agreement-based decision making

## Architecture

```
zen-intelligence-api/
├── core/
│   ├── orchestrator.py         # Advanced orchestration engine
│   ├── intelligence.py         # Query analysis & model selection
│   ├── consensus.py           # Weighted consensus building
│   └── performance_tracker.py # Model performance analytics
├── models/
│   ├── orchestration_request.py  # Request models
│   └── orchestration_response.py # Response models
├── api/
│   ├── main.py               # FastAPI server with advanced features
│   ├── middleware.py         # Custom middleware & logging
│   └── streaming.py          # Advanced streaming capabilities
├── config/
│   └── orchestration_rules.py # Smart orchestration rules
└── utils/
    └── prompt_engineering.py  # Advanced prompt crafting
```

## Installation

1. **Install additional requirements**:
```bash
pip install -r requirements-intelligence-api.txt
```

2. **Set up API keys** (same as Zen MCP server):
```bash
# Copy and configure environment
cp .env.example .env
# Add your API keys to .env
```

## Running the API Server

### Development Mode
```bash
cd zen_intelligence_api/api
python main.py
```

### Production Mode  
```bash
uvicorn zen_intelligence_api.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### OpenAI-Compatible Endpoints

#### Chat Completions
```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "messages": [
    {"role": "user", "content": "Analyze this code for security vulnerabilities"}
  ],
  "model": "zen-orchestrator",
  "temperature": 0.7,
  "orchestration_preferences": {
    "orchestration_strategy": "consensus",
    "preferred_models": ["claude-sonnet", "gpt-4"],
    "quality_threshold": 0.9
  }
}
```

#### List Models
```http
GET /v1/models
Authorization: Bearer your-api-key
```

### Advanced Zen Endpoints

#### Advanced Orchestration
```http
POST /v1/zen/orchestrate
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "query": "Design a scalable microservices architecture",
  "context": ["Previous analysis context"],
  "strategy": "sequential",
  "models": ["gpt-5", "claude-opus"],
  "max_latency": 30.0,
  "max_cost": 0.50,
  "enable_reasoning": true
}
```

#### Query Analysis
```http
POST /v1/zen/analyze
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "query": "How to optimize database performance?",
  "context": ["Additional context"],
  "include_cost_estimate": true
}
```

#### Performance Metrics
```http
GET /v1/zen/performance?model=gpt-4&domain=security
Authorization: Bearer your-api-key
```

#### Submit Feedback
```http
POST /v1/zen/learn
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "session_id": "session-uuid",
  "satisfaction_score": 0.9,
  "feedback_text": "Excellent analysis of the security issues"
}
```

## Intelligent Features

### Query Intelligence Analysis
The system automatically analyzes queries to determine:
- **Domain**: Architecture, Security, Debugging, Performance, etc.
- **Complexity**: Simple, Moderate, Complex, Expert
- **Urgency**: Research, Implementation, Critical Fix

### Orchestration Rules
Smart rules automatically determine the best approach:

```python
# Security queries use Claude's expertise with GPT validation
Security + Complex + Critical → Claude (primary) + GPT-4 (supporting) → Consensus

# Architecture queries use GPT's design skills with Claude review  
Architecture + Expert + Implementation → GPT-5 (primary) + Claude (review) → Sequential

# Simple urgent tasks prioritize speed
General + Simple + Critical → Gemini-Flash (single) → Fast response
```

### Performance Learning
The system continuously learns from:
- Response quality scores
- User satisfaction feedback
- Latency and cost metrics
- Success/failure rates

## Advanced Configuration

### Custom Orchestration Rules
```python
from zen_intelligence_api.config.orchestration_rules import OrchestrationRule

custom_rule = OrchestrationRule(
    rule_name="custom_security_rule",
    domains=[QueryDomain.SECURITY],
    complexities=[QueryComplexity.EXPERT],
    urgencies=[QueryUrgency.CRITICAL_FIX],
    preferred_models=["claude-opus", "gpt-5"],
    orchestration_strategy="consensus",
    priority=150  # Higher than default rules
)
```

### Environment Variables
```bash
# API Configuration
ZEN_API_HOST=0.0.0.0
ZEN_API_PORT=8000
ZEN_API_WORKERS=4

# Performance Settings
ZEN_MAX_CONCURRENT_REQUESTS=100
ZEN_REQUEST_TIMEOUT=300
ZEN_ENABLE_PERFORMANCE_LEARNING=true

# Security
ZEN_API_KEY_REQUIRED=true
ZEN_RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Logging
ZEN_LOG_LEVEL=INFO
ZEN_ENABLE_REQUEST_LOGGING=true
```

## Example Usage

### Python Client
```python
import httpx

async def analyze_code_security():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Authorization": "Bearer your-api-key"},
            json={
                "messages": [
                    {
                        "role": "user", 
                        "content": "Analyze this Python code for security vulnerabilities:\n\n```python\nuser_input = request.args.get('query')\nresult = db.execute(f'SELECT * FROM users WHERE name = {user_input}')\n```"
                    }
                ],
                "model": "zen-orchestrator",
                "orchestration_preferences": {
                    "orchestration_strategy": "consensus",
                    "quality_threshold": 0.9
                }
            }
        )
        
        result = response.json()
        print(f"Primary model: {result['model']}")
        print(f"Response: {result['choices'][0]['message']['content']}")
        
        # Zen metadata
        zen_data = result.get('zen_metadata', {})
        print(f"Consensus score: {zen_data.get('consensus_score')}")
        print(f"Models used: {zen_data.get('model_responses', [])}")

# Run the analysis
import asyncio
asyncio.run(analyze_code_security())
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/v1/zen/orchestrate" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Design a fault-tolerant distributed system",
    "strategy": "sequential",
    "models": ["gpt-5", "claude-opus"],
    "enable_reasoning": true,
    "enable_consensus_details": true
  }'
```

## Monitoring and Analytics

### Performance Metrics
- Global statistics across all models and domains
- Domain-specific performance analytics
- Model rankings and trends
- Best performer identification

### Health Monitoring
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "version": "1.0.0",
  "services": {
    "orchestrator": "active",
    "query_intelligence": "active", 
    "performance_tracker": "active"
  }
}
```

## Testing

Run the test suite:
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest zen_intelligence_api/tests/

# Run specific test
pytest zen_intelligence_api/tests/test_intelligence.py::TestQueryIntelligence::test_detect_security_domain
```

## Integration with Existing Zen MCP

The Intelligence API seamlessly integrates with the existing Zen MCP server:

1. **Shared Provider System**: Uses the same provider architecture
2. **Conversation Memory**: Leverages existing context revival capabilities  
3. **Tool Integration**: Can invoke existing MCP tools through orchestration
4. **Configuration**: Shares the same model configurations and API keys

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements-intelligence-api.txt ./
RUN pip install -r requirements.txt -r requirements-intelligence-api.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "zen_intelligence_api.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zen-intelligence-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zen-intelligence-api
  template:
    metadata:
      labels:
        app: zen-intelligence-api
    spec:
      containers:
      - name: api
        image: zen-intelligence-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
```

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility with OpenAI API

## License

Same license as the Zen MCP server project.