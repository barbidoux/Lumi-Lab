# Security Policy

## ⚠️ Disclaimer

**Lumi is a demo/educational project for mini-LLM training.** It is **NOT** intended for production use without proper security review and hardening.

## Security Considerations

### Model Safety
- **No safety filtering**: Generated content is unfiltered and may contain harmful/inappropriate text
- **Training data**: Models inherit biases and potentially harmful content from training datasets
- **Prompt injection**: No protection against adversarial prompts or jailbreaking attempts

### Infrastructure Security
- **API endpoints**: The FastAPI server has no authentication or rate limiting
- **File system access**: Scripts have broad file system access for model loading/saving
- **Network exposure**: API mode can expose models to network requests without security controls

## Reporting Security Issues

If you discover a security vulnerability in Lumi, please report it responsibly:

1. **Do NOT** create a public GitHub issue
2. Send details to: [Your contact email here]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested mitigation (if any)

## Recommended Security Practices

### For Educational Use
- Run in isolated environments (containers, VMs)
- Use test datasets only, avoid sensitive data
- Monitor generated content for inappropriate outputs
- Never expose API endpoints to public internet

### For Development
- Implement proper authentication/authorization for APIs
- Add input validation and sanitization
- Use content filtering for generated text
- Apply rate limiting and monitoring
- Regular security audits of dependencies

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | ⚠️ Educational only |
| < 1.0   | ❌ Pre-release     |

## Disclaimer

This software is provided for educational purposes only. Users are responsible for:
- Ensuring compliance with applicable laws and regulations
- Implementing appropriate security measures
- Monitoring and controlling model outputs
- Understanding risks of language model deployment

**Use at your own risk.**