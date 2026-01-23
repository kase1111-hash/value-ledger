# Security Policy

## Supported Versions

The following versions of Value Ledger are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously in Value Ledger, especially given its role in evidentiary accounting and cryptographic proof generation.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, report security vulnerabilities by emailing the maintainers directly or by opening a private security advisory on GitHub:

1. Go to the [Security Advisories](https://github.com/kase1111-hash/value-ledger/security/advisories) page
2. Click "New draft security advisory"
3. Fill in the details of the vulnerability

### What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential mitigations you've identified
- Your assessment of the severity (Critical, High, Medium, Low)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next scheduled release

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
2. **Investigation**: We will investigate and validate the vulnerability
3. **Communication**: We will keep you informed of our progress
4. **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)
5. **Disclosure**: We will coordinate public disclosure timing with you

## Security Features

Value Ledger includes several security features:

### Cryptographic Security
- **Merkle tree proofs** for tamper-evident ledger entries
- **Fernet encryption** for sensitive data (via `cryptography` library)
- **SHA-256 hashing** for content verification

### Input Validation
- **SSRF protection** in NatLangChain export module
- **Path traversal prevention** for file operations
- **Input sanitization** across all public APIs

### Integration Security
- **Boundary-SIEM integration** for security event monitoring
- **Boundary Daemon integration** for trust enforcement
- **Clock drift detection** for timestamp integrity

### Privacy Controls
- **Privacy levels** (PRIVATE, INTERNAL, SHARED, PUBLIC)
- **Consent-based access** via Learning Contracts
- **Content filtering** based on privacy settings

## Security Best Practices

When using Value Ledger:

1. **Keep dependencies updated**: Regularly update `pydantic` and `cryptography`
2. **Secure ledger files**: Protect `.jsonl` ledger files with appropriate file permissions
3. **Use encryption**: Enable Fernet encryption for sensitive entries
4. **Monitor events**: Integrate with Boundary-SIEM for security monitoring
5. **Validate inputs**: Always validate external data before creating ledger entries

## Known Security Considerations

### Alpha Status

This is an alpha release (v0.1.0-alpha.1). While security has been a priority, the APIs are not yet stable and may change. Please:

- Review code before production use
- Report any security concerns you identify
- Monitor releases for security updates

### Memory Vault Integration

The Memory Vault integration is currently stubbed. When the full integration becomes available, additional security considerations will apply.

## Security Audits

Value Ledger has not yet undergone a formal security audit. If you're interested in conducting or sponsoring a security audit, please contact the maintainers.

## Acknowledgments

We appreciate the security research community's efforts to improve the security of open source software. Contributors who report security issues responsibly will be acknowledged here (with their permission).
