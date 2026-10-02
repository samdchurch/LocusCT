"""
Shared bootstrap utilities for loading Merlin (StanfordMIMI/Merlin CT
foundation model, pip package `merlin-vlm`) on our cluster, where the
Singularity container filesystem is read-only and compute nodes have no
outbound internet access.

Originally written as private functions inside
scripts/embedding/precompute_merlin_image_embeddings.py; factored out here
so models/merlin_encoder.py (a trainable encoder, not just a standalone
precompute script) can reuse the exact same checkpoint-staging/HF-redirect
logic without duplicating it. That script now imports from this module
instead of defining its own copies.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PATCHED = {"resnet152": False, "clinical_longformer": False}


def disable_resnet152_imagenet_download() -> None:
    """Stop merlin's ImageEncoder from downloading torchvision's ImageNet-pretrained
    ResNet152 init weights.

    merlin.models.build.ImageEncoder unconditionally calls
    torchvision.models.resnet152(pretrained=True) before immediately overwriting
    every weight via Merlin's own checkpoint's load_state_dict() -- the ImageNet
    init is discarded and never actually used. Compute nodes here have no
    outbound internet, so that download just fails; patch it to build an
    uninitialized ResNet152 instead of attempting it.

    Idempotent -- safe to call more than once (e.g. once from the precompute
    script, once from model construction) without double-wrapping.
    """
    if _PATCHED["resnet152"]:
        return

    import torchvision.models as tv_models

    original_resnet152 = tv_models.resnet152

    def resnet152_no_download(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original_resnet152(*args, **kwargs)

    tv_models.resnet152 = resnet152_no_download
    _PATCHED["resnet152"] = True


def redirect_clinical_longformer(local_dir: str) -> None:
    """Redirect merlin's hardcoded yikuan8/Clinical-Longformer Hugging Face Hub loads
    to a local directory staged ahead of time.

    merlin.models.build.TextEncoder always constructs itself (even in ImageEmbedding
    mode) via AutoModel.from_pretrained("yikuan8/Clinical-Longformer") and
    AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer") -- a hardcoded repo id
    with no override argument. This cluster has no internet access at all, so those
    calls are patched to load local_dir instead whenever that exact repo id is passed.
    local_dir should be a plain save_pretrained()-style directory (config.json,
    tokenizer files, model weights), not a Hugging Face Hub cache's internal
    snapshots/ layout.

    Idempotent -- safe to call more than once without double-wrapping.
    """
    if _PATCHED["clinical_longformer"]:
        return

    from transformers import AutoModel, AutoTokenizer

    target_repo_id = "yikuan8/Clinical-Longformer"

    def make_redirecting_from_pretrained(original):
        def redirecting_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
            if pretrained_model_name_or_path == target_repo_id:
                pretrained_model_name_or_path = local_dir
            return original(pretrained_model_name_or_path, *args, **kwargs)
        return redirecting_from_pretrained

    AutoModel.from_pretrained = make_redirecting_from_pretrained(AutoModel.from_pretrained)
    AutoTokenizer.from_pretrained = make_redirecting_from_pretrained(AutoTokenizer.from_pretrained)
    _PATCHED["clinical_longformer"] = True


def ensure_checkpoint_staged(merlin_model_dir: str) -> None:
    """Verify the Merlin checkpoint is present where merlin.Merlin() expects it.

    The installed merlin package lives on the container's read-only squashfs
    image, so nothing at job time can write into it -- the submission script
    binds --merlin-model-dir directly onto that expected path instead. This
    just fails fast with a clear message if that bind is missing/misconfigured,
    rather than letting Merlin() hit an HF Hub download attempt.
    """
    import merlin.models.load as merlin_load

    checkpoint_name = merlin_load.MODEL_CONFIGS["default"]["checkpoint"]
    target = Path(merlin_load.__file__).resolve().parent / "checkpoints" / checkpoint_name
    if not target.exists():
        raise FileNotFoundError(
            f"Merlin checkpoint {checkpoint_name!r} not found at {target} (expected to be "
            f"bind-mounted from --merlin-model-dir {merlin_model_dir} -- check the submission "
            "script's --bind flags)."
        )


def load_merlin_i3resnet(model_dir: str, clinical_longformer_dir: str):
    """Load Merlin's image-only model and return just its I3ResNet152 encoder.

    Runs the three bootstrap steps above, builds Merlin(ImageEmbedding=True),
    and extracts .model.encode_image.i3_resnet -- the plain nn.Module holding
    conv1/bn1/relu/maxpool/layer1..4. MerlinArchitecture.__init__ always
    constructs a Clinical-Longformer TextEncoder too (~150M params), even in
    ImageEmbedding=True mode; since i3_resnet is a plain Python reference, it
    survives freeing that unused submodule's parent objects.
    """
    ensure_checkpoint_staged(model_dir)
    disable_resnet152_imagenet_download()
    redirect_clinical_longformer(clinical_longformer_dir)

    from merlin import Merlin

    wrapper = Merlin(ImageEmbedding=True)
    i3_resnet = wrapper.model.encode_image.i3_resnet
    del wrapper.model.encode_text
    del wrapper
    return i3_resnet
