# AetherGuard Research

**Offensive security research for AI infrastructure.**

We find vulnerabilities in the systems powering modern AI — vector databases, RAG pipelines, embedding services, and LLM toolchains — before attackers do.

## Mission

AI infrastructure is being deployed at scale with assumptions inherited from traditional software. Those assumptions don't hold. AetherGuard Research produces empirical, reproducible security audits that expose real attack surfaces in production AI systems and validate practical defenses.

## Research Projects

| Project | Focus | Status |
|---------|-------|--------|
| [VectorBleed](./vectorbleed/) | Cross-tenant data leakage in shared vector databases via embedding space proximity attacks | ✅ Complete |

## Blog

Read our research writeups and technical deep-dives:

🔗 **[blog.aetherguard.ai](https://blog.aetherguard.ai)**

## Research Areas

- **Vector Database Security** — Tenant isolation failures, namespace bypass, embedding inversion
- **RAG Pipeline Attacks** — Prompt injection through retrieval, context poisoning, data exfiltration
- **Embedding Model Risks** — Inversion attacks, membership inference, model extraction
- **LLM Toolchain Audits** — Framework misconfigurations, insecure defaults, supply chain risks

## Getting Started

Each research project lives in its own directory with a self-contained environment and reproducible methodology.

```bash
git clone https://github.com/aetherguard/aetherguard-research.git
cd aetherguard-research/<project>
```

Refer to the individual project README for setup instructions.

## Responsible Disclosure

All research follows responsible disclosure practices. Vulnerabilities are reported to affected vendors with a coordinated timeline before public release. Our goal is to improve the security posture of the AI ecosystem, not to enable exploitation.

## License

Research use only. See individual project directories for specific terms.

---

**AetherGuard** — Securing the infrastructure layer of AI.
