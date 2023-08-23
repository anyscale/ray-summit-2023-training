import matplotlib.pyplot as plt
import pandas as pd
import torch

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from ray.data import read_images
from ray.data.preprocessors import TorchVisionPreprocessor
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer
from transformers import CLIPTextModel
from torchvision import transforms
from typing import Dict


def get_train_dataset(
    model_dir,
    instance_images_dir,
    class_images_dir,
    instance_prompt,
    class_prompt,
    image_resolution=512,
):
    """
    Build a dataset for fine-tuning the Dreambooth model.

    Args:
        model_dir (str): Directory containing the pre-trained model.
        instance_images_dir (str): Directory containing instance images.
        class_images_dir (str): Directory containing class images.
        instance_prompt (str): Instance prompt text.
        class_prompt (str): Class prompt text.
        image_resolution (int, optional): Image resolution. Defaults to 512.

    Returns:
        ray.data.dataset.Dataset: Training dataset.
    """
    
    ### Step 1: Loading Images and Duplication ###

    # Load both directories of images as Ray Datasets
    instance_dataset = read_images(instance_images_dir)
    class_dataset = read_images(class_images_dir)

    # We now duplicate the instance images multiple times to make the
    # two sets contain exactly the same number of images.
    # This is so we can zip them up during training to compute the
    # prior preserving loss in one pass.
    #
    # Example: If we have 200 class images (for regularization) and 5 instance
    # images of our subject, then we'll duplicate the instance images 40 times
    # so that our dataset looks like:
    #
    #     instance_image_0, class_image_0
    #     instance_image_1, class_image_1
    #     instance_image_2, class_image_2
    #     instance_image_3, class_image_3
    #     instance_image_4, class_image_4
    #     instance_image_0, class_image_5
    #     ...
    dup_times = class_dataset.count() // instance_dataset.count()

    # Duplicate input images with Ray Data map_batches
    instance_dataset = instance_dataset.map_batches(
        lambda df: pd.concat([df] * dup_times), batch_format="pandas"
    )

    ### Step 2: Tokenization ###

    # Load tokenizer for tokenizing the image prompts.
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        subfolder="tokenizer",
    )

    def tokenize(prompt):
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()

    # Get the token ids for both prompts.
    class_prompt_ids = tokenize(class_prompt)[0]
    instance_prompt_ids = tokenize(instance_prompt)[0]

    ### Step 3: Image Preprocessing ###
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomCrop(image_resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    instance_ds_preprocessor = TorchVisionPreprocessor(
        columns=["image"], output_columns=["instance_image"], transform=transform
    )
    class_ds_preprocessor = TorchVisionPreprocessor(
        columns=["image"], output_columns=["class_image"], transform=transform
    )

    ### Step 4: Apply Preprocessing Steps as Ray Dataset Operations ###

    # For each dataset:
    # - perform image preprocessing
    # - drop the original image column
    # - add a new column with the tokenized prompts

    instance_dataset = (
        instance_ds_preprocessor.transform(instance_dataset)
        .drop_columns(["image"])
        .add_column("instance_prompt_ids", lambda df: [instance_prompt_ids] * len(df))
    )

    class_dataset = (
        class_ds_preprocessor.transform(class_dataset)
        .drop_columns(["image"])
        .add_column("class_prompt_ids", lambda df: [class_prompt_ids] * len(df))
    )

    ### Step 5: Dataset Size Adjustment ###

    # We may have too many duplicates of the instance images, so limit the
    # dataset size so that len(instance_dataset) == len(class_dataset)
    final_size = min(instance_dataset.count(), class_dataset.count())

    ### Step 6: Zip Images ###

    train_dataset = (
        instance_dataset.limit(final_size)
        .repartition(final_size)
        .zip(class_dataset.limit(final_size).repartition(final_size))
    )

    print("Training dataset schema after pre-processing:")
    print(train_dataset.schema())

    ### Step 7: Random Shuffling ###

    return train_dataset.random_shuffle()


def collate(batch, dtype):
    """Build Torch training batch.

    B = batch size
    (C, W, H) = (channels, width, height)
    L = max length in tokens of the text guidance input

    Input batch schema (see `get_train_dataset` on how this was setup):
        instance_images: (B, C, W, H)
        class_images: (B, C, W, H)
        instance_prompt_ids: (B, L)
        class_prompt_ids: (B, L)

    Output batch schema:
        images: (2 * B, C, W, H)
            All instance images in the batch come before the class images:
            [instance_images[0], ..., instance_images[B-1], class_images[0], ...]
        prompt_ids: (2 * B, L)
            Prompt IDs are ordered the same way as the images.

    During training, a batch will be chunked into 2 sub-batches for
    prior preserving loss calculation.
    """

    images = torch.cat([batch["instance_image"], batch["class_image"]], dim=0)
    images = images.to(memory_format=torch.contiguous_format).to(dtype)

    batch_size = len(batch["instance_prompt_ids"])

    prompt_ids = torch.cat(
        [batch["instance_prompt_ids"], batch["class_prompt_ids"]], dim=0
    ).reshape(batch_size * 2, -1)

    return {
        "images": images,
        "prompt_ids": prompt_ids,  # Token ids should stay int.
    }


# Third helper is all the bits and pieces from the training piece


def prior_preserving_loss(model_pred, target, weight):
    """
    Calculate prior-preserving loss.

    Args:
        model_pred (torch.Tensor): Model's prediction.
        target (torch.Tensor): Target tensor.
        weight (float): Weight of the prior loss.

    Returns:
        torch.Tensor: Prior-preserving loss.
    """
    # Chunk the noise and model_pred into two parts and compute
    # the loss on each part separately.
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)

    # Compute instance loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    # Compute prior loss
    prior_loss = F.mse_loss(
        model_pred_prior.float(), target_prior.float(), reduction="mean"
    )

    # Add the prior loss to the instance loss.
    return loss + weight * prior_loss


def get_target(scheduler, noise, latents, timesteps):
    """
    Get the target for loss depending on the prediction type.

    Args:
        scheduler: Scheduler object.
        noise (torch.Tensor): Noise tensor.
        latents (torch.Tensor): Latent tensor.
        timesteps (int): Number of timesteps.

    Returns:
        torch.Tensor: Target tensor for loss calculation.
    """
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unknown prediction type {pred_type}")


def load_models(config):
    """
    Load pre-trained Stable Diffusion models.

    Args:
        config (dict): Model configuration.

    Returns:
        Tuple: Loaded model components.
    """
    # Load all models in bfloat16 to save GRAM.
    # For models that are only used for inferencing,
    # full precision is also not required.
    dtype = torch.bfloat16

    text_encoder = CLIPTextModel.from_pretrained(
        config["model_dir"],
        subfolder="text_encoder",
        torch_dtype=dtype,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model_dir"],
        subfolder="scheduler",
        torch_dtype=dtype,
    )

    # VAE is only used for inference, keeping weights in full precision is not required.
    vae = AutoencoderKL.from_pretrained(
        config["model_dir"],
        subfolder="vae",
        torch_dtype=dtype,
    )
    # We are not training VAE part of the model.
    vae.requires_grad_(False)

    # Convert unet to bf16 to save GRAM.
    unet = UNet2DConditionModel.from_pretrained(
        config["model_dir"],
        subfolder="unet",
        torch_dtype=dtype,
    )
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    unet_trainable_parameters = unet.parameters()
    text_trainable_parameters = text_encoder.parameters()

    torch.cuda.empty_cache()

    return (
        text_encoder,
        noise_scheduler,
        vae,
        unet,
        unet_trainable_parameters,
        text_trainable_parameters,
    )


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns a state dict containing just the attention processor parameters.

    Args:
        unet: UNet2DConditionModel object.

    Returns:
        Dict[str, torch.tensor]: State dictionary of attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            param_name = f"{attn_processor_key}.{parameter_key}"
            attn_processors_state_dict[param_name] = parameter
    return attn_processors_state_dict

def show_images(filenames):
    """
    Display a set of images using matplotlib.

    Args:
        filenames (list): List of filenames to display.
    """
    fig, axs = plt.subplots(1, len(filenames), figsize=(4 * len(filenames), 4))
    for i, filename in enumerate(filenames):
        ax = axs if len(filenames) == 1 else axs[i]
        ax.imshow(plt.imread(filename))
        ax.axis("off")
    plt.show()