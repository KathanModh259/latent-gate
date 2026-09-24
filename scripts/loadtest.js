// ============================================================================
// LatentGate Load Test — k6 Script
// ============================================================================
// Usage:
//   # Install k6: https://k6.io/docs/getting-started/installation/
//   k6 run scripts/loadtest.js
//
//   # With custom API URL and duration:
//   k6 run -e API_URL=http://localhost:8000 -e DURATION=30s scripts/loadtest.js
// ============================================================================

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// ---- Custom Metrics ----
const compressionRatio = new Rate('compression_ratio');
const tokenSavings = new Trend('token_savings');
const compressLatency = new Trend('compress_latency');
const errors = new Counter('api_errors');

// ---- Configuration ----
const API_URL = __ENV.API_URL || 'http://localhost:8000';
const DURATION = __ENV.DURATION || '60s';
const VUS = parseInt(__ENV.VUS || '10', 10);

export const options = {
  stages: [
    // Ramp up to target VUs
    { duration: '10s', target: VUS },
    // Stay at target
    { duration: DURATION, target: VUS },
    // Ramp down
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    'http_req_duration': ['p(95)<5000'], // 95% of requests complete in <5s
    'http_req_failed': ['rate<0.05'],    // Less than 5% failure rate
    'compression_ratio': ['rate>0.5'],   // More than 50% of requests get compression
  },
};

// ---- Test Data ----
const SHORT_PROMPT = 'What is the capital of France?';

const LONG_PROMPT = `
I need you to help me build a REST API with FastAPI that handles user
authentication using JWT tokens, includes rate limiting, has PostgreSQL
database integration with SQLAlchemy ORM, supports file uploads to S3,
and generates comprehensive OpenAPI documentation. The API should follow
clean architecture patterns with repository pattern for data access.
Please also include proper error handling, logging, and health check
endpoints. The system should support role-based access control with
admin and regular user roles.

Each endpoint should have proper input validation using Pydantic models.
The database should use Alembic for migrations. Include Docker support
with multi-stage builds and a docker-compose file for local development.
Add comprehensive unit tests with pytest and integration tests.

The API should support both JSON and form-data request bodies.
Include WebSocket support for real-time notifications.
Add Redis caching for frequently accessed endpoints.
`.trim();

const CODE_PROMPT = `
Write a Python function that:
1. Takes a list of file paths as input
2. Reads each file (handle encoding errors gracefully)
3. Extracts all email addresses using regex
4. Deduplicates and sorts the results
5. Returns a dictionary mapping filenames to lists of emails

Include type hints, docstrings, error handling, and a main() function
that reads paths from command-line arguments or stdin.
`.trim();

// ---- Helper ----
function checkResponse(name, response, expectedStatus = 200) {
  const success = check(response, {
    [`${name} status is ${expectedStatus}`]: (r) => r.status === expectedStatus,
    [`${name} response time < 10s`]: (r) => r.timings.duration < 10000,
  });

  if (!success || response.status >= 500) {
    errors.add(1);
  }

  return success;
}

// ---- Test Scenarios ----
export default function () {
  // ---- Health Check ----
  group('health_check', function () {
    const res = http.get(`${API_URL}/health`);
    checkResponse('health', res);
    sleep(0.5);
  });

  // ---- Compress Short Prompt ----
  group('compress_short', function () {
    const payload = JSON.stringify({ text: SHORT_PROMPT });
    const params = { headers: { 'Content-Type': 'application/json' } };

    const res = http.post(`${API_URL}/compress`, payload, params);
    if (checkResponse('compress_short', res)) {
      const data = res.json();
      if (data.compressed_tokens > 0) {
        compressionRatio.add(true);
        tokenSavings.add(data.tokens_saved || 0);
        compressLatency.add(res.timings.duration);
      }
    }
    sleep(0.5);
  });

  // ---- Compress Long Prompt ----
  group('compress_long', function () {
    const payload = JSON.stringify({ text: LONG_PROMPT });
    const params = { headers: { 'Content-Type': 'application/json' } };

    const res = http.post(`${API_URL}/compress`, payload, params);
    if (checkResponse('compress_long', res)) {
      const data = res.json();
      tokenSavings.add(data.tokens_saved || 0);
      compressLatency.add(res.timings.duration);
    }
    sleep(1);
  });

  // ---- Compress Code Prompt ----
  group('compress_code', function () {
    const payload = JSON.stringify({ text: CODE_PROMPT });
    const params = { headers: { 'Content-Type': 'application/json' } };

    const res = http.post(`${API_URL}/compress`, payload, params);
    if (checkResponse('compress_code', res)) {
      const data = res.json();
      tokenSavings.add(data.tokens_saved || 0);
      compressLatency.add(res.timings.duration);
    }
    sleep(0.5);
  });

  // ---- Batch Compression ----
  group('compress_batch', function () {
    const payload = JSON.stringify({
      texts: [SHORT_PROMPT, LONG_PROMPT, CODE_PROMPT],
    });
    const params = { headers: { 'Content-Type': 'application/json' } };

    const res = http.post(`${API_URL}/compress/batch`, payload, params);
    if (checkResponse('compress_batch', res)) {
      const data = res.json();
      if (data.total_tokens_saved > 0) {
        compressionRatio.add(true);
        tokenSavings.add(data.total_tokens_saved);
      }
    }
    sleep(1);
  });
}
