# Chimera 

**Flow-Anchor Poisoning: Clean-Label Backdoors in Flow-Matching Diffusion Transformers**

> Research project investigating data poisoning vulnerabilities in next-generation DiT architectures (FLUX.1, Stable Diffusion 3)

## Overview

Chrimara implements **Flow-Anchor Poisoning**, a novel clean-label attack targeting Flow-Matching Diffusion Transformers (DiTs). Unlike prior work (Nightshade, BadDiffusion) designed for U-Net LDMs, this attack exploits:

- **Global attention dynamics** in DiT architectures
- **Linear CFG coupling** in flow-matching objectives  
- **Scale-conditional activation** (backdoors trigger at high guidance scales)
