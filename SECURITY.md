# Security Policy

## Supported Versions

Only the latest release receives security fixes.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing the maintainers directly rather than opening a public issue.

**Please include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

We will acknowledge receipt within 48 hours and aim to provide a fix within 7 days for critical issues.

## URL inputs

When `source` is an `http://` or `https://` URL, openextract downloads it with
`httpx`, following redirects, with a 30 second timeout. No host validation is
applied, so a URL can reach private or internal addresses. If URLs come from
untrusted users, validate them yourself, or download the content with your own
client and pass the bytes to `extract()`.

## Model requests

All provider requests use HTTPX. Model credentials are attached to individual
model requests and are not sent when fetching source URLs. Provider redirects
are rejected, including when an injected HTTPX client enables redirects.
Caller-owned clients are used only for model requests and are not closed.

Configure credentials using `Model(api_key=...)`, headers, or the documented
environment variables. Base URLs cannot contain credentials, query strings, or
fragments. Model errors include selected upstream error messages, codes, and finish
reasons. Known configured credentials and standard request authentication headers
are redacted from these messages and error attributes; entire response bodies are
not dumped into messages. Provider messages can contain input data. Validation
retry feedback omits raw input values. Chained exceptions can still contain
request, response, or validation data; handle them appropriately in logs.
