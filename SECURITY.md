# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
**"Report a vulnerability"** button (Security tab → Advisories) to report
privately. Include steps to reproduce and the affected version or commit.

This is a community project maintained on a best-effort basis; there is no
bug bounty and no guaranteed response time.

## Scope

In scope: the `aitesteval` engine and the `portal` web app in this repository,
for example authentication or CSRF bypass, cross-user data access, leaking
stored provider keys, SSRF past `check_outbound_url`, or code execution from
portal input.

Out of scope: deployments you run yourself with unsafe settings (for example
`PORTAL_ALLOW_PRIVATE_URLS=1` on a public host, no TLS, or a weak
`PORTAL_SECRET_KEY`), and YAML scenario files you write yourself. The CLI
deliberately supports `python:` and `pickle:` model loaders, which execute
code, so only run scenarios and model files you trust.

## Running the portal publicly

Follow the production checklist in [docs/PORTAL.md](docs/PORTAL.md): set a
strong `PORTAL_SECRET_KEY`, serve over HTTPS with `PORTAL_SECURE_COOKIES=1`,
and keep `PORTAL_ALLOW_PRIVATE_URLS` off.
