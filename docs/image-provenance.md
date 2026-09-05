# Container image provenance

The reference pair currently runs an unpushed local build. The head and worker
have different local image IDs because the image was built separately; neither
has a repository digest. Its private local tag is intentionally absent from
this repository.

The distributable candidate is Tony Dennis' public multi-architecture image,
pinned by manifest digest in `.env.example`:

```
ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
```

Its published build is the upstream SM121 vLLM correction chain plus DFlash2.
The runtime patch generator additionally verifies the exact `modelopt.py` and
`kernel_warmup.py` source hashes before altering them. If the image differs, it
fails rather than silently applying a fuzzy patch.

The public digest still needs a complete two-node cold-boot gate on the
reference pair before it can be called byte-for-byte deployment-proven. Until
that gate is recorded, distinguish these two claims:

- the recipe and local image build are cold-boot proven; and
- the pinned public distribution image is the candidate reproduction path.

Do not replace the digest with a mutable tag in production.
