# Inference environment for SAT-Pro (sat/SAT, github.com/zhaoziheng/SAT), run the same
# way as the repo's own inference.py: torchrun-launched (DDP is unconditional even for
# 1 GPU -- see build_maskformer()/Text_Encoder() in model/build_model.py and
# model/text_encoder.py), UNET-L vision backbone + 'ours' (BioLORD-based) text encoder,
# weights loaded via model.load_state_dict() from the published SAT_Pro.pth /
# text_encoder.pth checkpoints (staged on the cluster's model directory convention, not
# vendored in this repo).
#
# Kept as its OWN image rather than extending grounder.sif, even though SAT's
# requirements.txt (positional_encodings, monai, torch, einops, transformers, scipy,
# nibabel, pandas, openpyxl) has no compiled CUDA extensions and nothing that actually
# conflicts with grounder.sif's own stack -- so a plain runtime (not devel) base image
# is enough here, unlike biomedparse-v2/Dockerfile's detectron2 build.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Matches grounder.sif's own HF cache convention (Dockerfile.grounder). SAT's text
# encoder (model/knowledge_encoder.py -> model/text_tower.py) calls
# AutoModel.from_pretrained('FremyCompany/BioLORD-2023-C') (and its tokenizer) at
# construction time, before the SAT_Pro/text_encoder checkpoint weights are loaded on
# top of it. This cluster has no internet access, so FremyCompany/BioLORD-2023-C must
# be pre-downloaded on a machine with internet access and staged into this same HF
# cache directory (standard huggingface_hub snapshot layout) before running -- see
# submission_scripts/evaluation/submit_evaluate_sat_pretrained.sh.
ENV HF_HOME=/cache/huggingface

# Only torch/numpy actually ship with this base image -- transformers/nibabel/scipy
# do NOT (verified against sat/SAT/requirements.txt directly after the same class of
# missing-dependency bug turned up in segvol/Dockerfile's own "already ships with the
# base image" assumption for scipy there). matplotlib/tqdm added for this repo's own
# eval scripts' visualization/progress bar (SAT itself needs neither).
#
# torch is pinned to match the base image's own 2.6.0, and monai to 1.4.0 (the latest
# release that doesn't require torch>=2.8), because letting pip resolve either
# unpinned upgrades torch out from under this base image's ALREADY-matched, ABI-linked
# torchvision/torchaudio builds (both ship with the base image too, despite neither
# being pip-installed here) -- torch itself gets silently bumped to satisfy monai's
# unpinned torch>=2.8.0 floor, but torchvision/torchaudio are never reinstalled to
# match, so the first real use of either crashes: torchaudio with "undefined symbol:
# ...deleteNode..." (an ABI mismatch OSError), torchvision with "partially initialized
# module 'torchvision' has no attribute 'extension' (most likely due to a circular
# import)" -- both surfacing deep in transformers' lazy-loaded BertModel import, not
# at the point that actually caused them.
#
# dynamic_network_architectures (model/maskformer.py's PlainConvUNet/InitWeights_He --
# SAT's UNet-L vision backbone) is a genuine runtime dependency that ISN'T in SAT's own
# requirements.txt, but it is deliberately NOT pip-installed here (do not add it back):
# SAT depends on its own customized fork (PlainConvUNet with no num_classes arg, a
# backbone-only variant unlike every public PyPI release of this package), vendored at
# sat/SAT/model/dynamic-network-architectures-main per SAT's own README ("cd model &&
# pip install -e dynamic-network-architectures-main"). Installing the PyPI package
# instead silently substitutes the wrong PlainConvUNet and fails with "missing 1
# required positional argument: 'num_classes'". Since that vendored fork lives in the
# bind-mounted repo, not this image's build context, it can't be pip-installed at
# build time either -- scripts/evaluation/evaluate_sat_{onc,ed}.py instead add it to
# sys.path at runtime, ahead of site-packages.
RUN pip install --no-cache-dir \
        torch==2.6.0 \
        positional_encodings \
        monai==1.4.0 \
        einops \
        transformers \
        scipy \
        nibabel \
        pandas \
        openpyxl \
        matplotlib \
        tqdm

# The SAT source (sat/SAT) and grounder repo are expected to be bind-mounted here at
# run time, e.g.:
#   docker run --gpus all -v /path/to/grounder:/workspace -w /workspace <image> \
#       torchrun --nproc_per_node=1 scripts/evaluation/evaluate_sat_ed.py --sat-repo sat/SAT ...
# No ENTRYPOINT/CMD is set on purpose, so `docker run ... bash` and
# `docker run ... torchrun ...` both work as plain commands.
WORKDIR /workspace
