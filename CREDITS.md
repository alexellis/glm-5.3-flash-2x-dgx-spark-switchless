# Credits and provenance

This recipe integrates work from:

- Z.ai for GLM-5.3-Flash and the corrected chat template;
- LibertAIDAI for the ModelOpt NVFP4 checkpoint;
- incoai for the GLM-5.3-Flash DFlash2 drafter;
- the vLLM and FlashInfer projects;
- Tony Dennis and contributors for the DGX Spark SM121 and DFlash2 serving
  stack;
- Jacopo Nardiello for the acceptance-guided adaptive-k scheduler, retained
  with its upstream Apache-2.0 provenance header;
- MiaAI-Lab for the accepted-prefix EMA, 2/4/7 verification policy, and
  batch-uniform CUDA-graph strategy; and
- NVIDIA for DGX Spark, CUDA, and NCCL.

The scripts and documentation authored in this repository are MIT licensed.
Models, container images, downloaded templates, vLLM, FlashInfer, CUDA, and
NCCL retain their respective upstream licences. This repository does not copy
the third-party container patch stack.
