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
