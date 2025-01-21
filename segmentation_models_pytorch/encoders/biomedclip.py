from open_clip import create_model_from_pretrained
import torch


def load_biomedclip_model(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    # make sure to only retain the trunk of the CLIP model
    model, _ = load_clip_vision_model(model_name).trunk
    return model


def load_biomedclip_preprocessor(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    # make sure to only retain the trunk of the CLIP model
    _, preprocessor = load_clip_vision_model(model_name).trunk
    return preprocessor


def load_clip_vision_model(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    model, preprocessor = create_model_from_pretrained(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model.visual, preprocessor
