# Security Policy for Void AI

## Supported Model Versions and Maintenance Status

Void AI is a long-term project with multiple AI models at various stages of development and maintenance. We continuously monitor and maintain **all** released models to ensure security and stability. However, the level of active development and updates varies by model version:

| Model Version | Support Level                                | Description                                                                                      |
|---------------|----------------------------------------------|--------------------------------------------------------------------------------------------------|

<!-- | **Z3**        | ✅ Actively maintained                        | This is our latest and most advanced model. It receives full support including security patches, bug fixes, performance improvements, and new features. Continuous monitoring and active development are ongoing. |-->
<!--| **Z2**        | ✅ Maintenance and critical updates only     | This model is considered stable and mature. It receives critical security patches, important bug fixes, and performance optimizations. However, new features or major changes are no longer planned for Z2. |-->
| **Z1**        | ✅ Actively maintained | This is our latest and most advanced model. It receives full support including security patches, bug fixes, performance improvements, and new features. Continuous monitoring and active development are ongoing. |
<!-- some stuff: ⚠️ Monitored but no longer actively updated -->
<!-- some stuff: ✅ Maintenance and critical updates only -->
<!-- some stuff: ✅ Actively maintained -->
---

## Continuous Security Monitoring

- All models, including deprecated ones, are periodically scanned and reviewed for vulnerabilities and security issues.  
- We leverage automated tools, manual audits, and community reports to identify potential risks.  
- Critical security vulnerabilities affecting any model will be evaluated and patched promptly, regardless of the model’s support status.  

---

## Reporting a Security Vulnerability

We value responsible disclosure and encourage the community and users to report any potential security vulnerabilities. Please follow the guidelines below:

### How to Report

- Send your report via email to: **[axionindustries.official@gmail.com]**).  
- Use the subject line: `[Void AI Security] Vulnerability Report`.  
- Include detailed information such as:  
  - A clear description of the vulnerability.  
  - Step-by-step instructions to reproduce the issue.  
  - Potential impact or severity assessment.  
  - Suggested fixes or mitigations, if applicable.  
- If you prefer, you can encrypt your message using our public PGP key (available on request).  

### What to Expect

- We will acknowledge receipt of your report within **48 hours**.  
- Our security team will assess and verify the issue.  
- We will communicate progress updates and estimated resolution timelines as appropriate.  
- Please keep the details confidential until an official fix or patch is released to avoid exploitation.  

---

## Supported Versions and Upgrade Policy

- **Users should always run supported models Z1 in production environments** to benefit from security updates and active support.  
- Unsupported models (like Z1) may contain unpatched vulnerabilities; continued use is at the user’s own risk.  
- We may backport critical security fixes to maintenance versions when feasible.  
- Official announcements about end-of-life, support changes, or critical updates will be communicated via the project’s channels, including the GitHub repository, discussions, and email lists if applicable.  

---

## Security Best Practices for Users

- Keep your Void AI deployment and dependencies up to date with the latest security patches.  
- Restrict access to sensitive data and model APIs to trusted parties.  
- Regularly audit your environment for unusual behavior or unauthorized access.  
- Follow secure coding and deployment practices when customizing or extending the models.  
- Report any suspicious activity or potential security flaws promptly.  

---

## Thank You for Contributing to Void AI Security

Your vigilance and responsible reporting play a crucial role in maintaining the security and integrity of Void AI. We appreciate the time and effort taken to help protect this project and its users.

---

*Last updated: 2025-06-20*
