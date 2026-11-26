"""G3 retriever (image / text / GPS encoders) used by both backbones.

Model definition from the official G3 implementation
(https://github.com/Applied-Machine-Learning-Lab/G3, Apache-2.0), modified:
checkpoint loading and embedding helpers were added for this repository.
"""

import torch
import torch.nn as nn
from pyproj import Proj, Transformer
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from .rff.layers import GaussianEncoding


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class CustomLocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        for i, s in enumerate(self.sigma):
            self.add_module(f"LocEnc{i}", LocationEncoderCapsule(sigma=s))

        proj_wgs84 = Proj("epsg:4326")
        proj_mercator = Proj("epsg:3857")
        self.transformer = Transformer.from_proj(
            proj_wgs84, proj_mercator, always_xy=True
        )

    def forward(self, input):
        # input: Tensor of shape (batch_size, 2) on correct device
        # Project lat/lon CPU for pyproj
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        projected = self.transformer.transform(lon, lat)  # x, y arrays
        # Build coords as [y, x]
        coords = [[y, x] for x, y in zip(*projected)]
        device = input.device
        # Create tensor directly on correct device
        location = torch.tensor(coords, dtype=torch.float32, device=device)
        # Normalize
        location = location / 20037508.3427892

        # Initialize features on same device
        batch_size = location.size(0)
        location_features = torch.zeros((batch_size, 512), device=device)

        # Accumulate capsule outputs
        for i in range(self.n):
            location_features += self._modules[f"LocEnc{i}"](location)

        return location_features


class G3(nn.Module):
    def __init__(self):
        super(G3, self).__init__()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.vision_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.text_processor = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.vision_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection

        self.logit_scale1 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale2 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale3 = nn.Parameter(torch.tensor(3.99))

        self.location_encoder = CustomLocationEncoder()
        self.vision_projection_else_1 = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
        )
        self.text_projection_else = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
        )
        self.vision_projection_else_2 = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
        )
        self.location_projection_else = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 768)
        )

        # Freeze CLIP
        for param in [
            self.vision_model,
            self.vision_projection,
            self.text_model,
            self.text_projection,
        ]:
            param.requires_grad_(False)

    def forward(self, images, texts, longitude, latitude, return_loss=True):
        vision_output = self.vision_model(images)[1]
        text_output = self.text_model(**texts)[1]
        image_embeds = self.vision_projection(vision_output)
        text_embeds = self.text_projection(text_output)

        # Location embedding
        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        # Phase 1: image-text
        image_embeds_1 = self.vision_projection_else_1(image_embeds)
        text_embeds_1 = self.text_projection_else(text_embeds)
        image_embeds_1 = image_embeds_1 / image_embeds_1.norm(p=2, dim=-1, keepdim=True)
        text_embeds_1 = text_embeds_1 / text_embeds_1.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale1.exp()
        logits_ti = torch.matmul(text_embeds_1, image_embeds_1.t()) * logit_scale
        logits_it = logits_ti.t()
        loss1 = self.clip_loss(logits_ti) if return_loss else None

        # Phase 2: image-location
        image_embeds_2 = self.vision_projection_else_2(image_embeds)
        location_embeds_2 = self.location_projection_else(location_embeds)
        image_embeds_2 = image_embeds_2 / image_embeds_2.norm(p=2, dim=-1, keepdim=True)
        location_embeds_2 = location_embeds_2 / location_embeds_2.norm(
            p=2, dim=-1, keepdim=True
        )
        logit_scale = self.logit_scale2.exp()
        logits_li = torch.matmul(location_embeds_2, image_embeds_2.t()) * logit_scale
        logits_il = logits_li.t()
        loss2 = self.clip_loss(logits_li) if return_loss else None

        total_loss = (loss1 + loss2) if return_loss else None
        return {
            "logits_per_texts_with_images": logits_ti,
            "logits_per_images_with_texts": logits_it,
            "logits_per_locations_with_images": logits_li,
            "logits_per_images_with_locations": logits_il,
            "loss": total_loss,
            "vision_output": vision_output,
            "text_output": text_output,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        cap_loss = self.contrastive_loss(similarity)
        img_loss = self.contrastive_loss(similarity.t())
        return (cap_loss + img_loss) / 2.0


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_g3(checkpoint_path: str, device="cpu") -> G3:
    """Load a G3 ``state_dict`` checkpoint (e.g. ``g3.pth`` from Jia-py/G3-checkpoint)."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = G3()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_images(model: G3, pixel_values: torch.Tensor):
    """Return L2-normalized (base, image->text, image->location) embeddings."""
    vision_output = model.vision_model(pixel_values)[1]
    base = model.vision_projection(vision_output)
    img_txt = model.vision_projection_else_1(base)
    img_loc = model.vision_projection_else_2(base)
    norm = lambda x: x / x.norm(p=2, dim=-1, keepdim=True)
    return norm(base), norm(img_txt), norm(img_loc)
