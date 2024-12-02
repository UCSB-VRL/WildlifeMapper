import torch
import torch.nn as nn
from segment_anything import sam_model_registry
import torchvision.transforms as transforms
from typing import Optional, Tuple, Type

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze image encoder,just train the hfc adaptor
        for name, param in self.image_encoder.named_parameters():
            if "hfc_embed" in name:
                param.requires_grad = True
            elif "hfc_attn" in name:
                param.requires_grad = True
            elif "patch_embed" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        #train mask decoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        
    def fft(self, img, rate=0.125):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        x = img.tensors
        device = x.device
        x = transforms.Grayscale()(x)
        mask = torch.ones(x.shape).to(device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 0

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        # fft = fft * (1 - mask)
        fft = fft * mask

        fft_hires = torch.fft.ifftshift(fft)
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)

        return inv

    def forward(self, image, box):
        # do not compute gradients for image encoder and prompt encoder
        mask = self.fft(image)
        
        # hfc_embedding = self.hfc_adaptor(mask)
        #with torch.no_grad():
        image_embedding = self.image_encoder(image.tensors, mask)  # (B, 256, 64, 64)
        
        # import pdb; pdb.set_trace()
        #compute gradients for prompt encoder
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.tensors.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        #TODO : prompt encoder only works for batch size 1, fix it later for b>1
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=None,
        #     boxes=box_torch,
        #     masks=None,
        # )
        out = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=None,  # (B, 2, 256)
            dense_prompt_embeddings=None,  # (B, 256, 64, 64)
            multimask_output=False,
            hfc_embed=None,
        )
        return out