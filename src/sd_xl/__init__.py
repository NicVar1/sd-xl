from pathlib import Path

import torch
from diffusers import AutoencoderKL, AutoPipelineForText2Image


def main():
    print("Cargando modelo ...")

    # VAE corregido para float16
    # El VAE original de SDXL da imagenes negras en float16
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        dtype=torch.float16,
    )

    modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        dtype=torch.float16,
        variant="fp16",
    )
    modelo = modelo.to("cuda")

    # Cuda - nvidia
    # mps - Mac con procesadores MX
    # xpu - Grafica intel IRIS

    prompt = input("Escribe el prompt de la imagen: ")
    negative_prompt = (
        "blurry, low quality, low resolution, worst quality, jpeg artifacts, "
        "deformed, distorted, disfigured, bad anatomy, bad proportions, "
        "extra limbs, extra fingers, missing fingers, mutated hands, "
        "poorly drawn face, poorly drawn hands, ugly, duplicate, cropped, "
        "out of frame, watermark, signature, text, logo"
    )

    print("Generando la imagen...")

    imagen = modelo(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps =  30,
        guidance_scale=7.0,
        height=1024,
        width=1024,
    ).images[0]

    ruta = Path("Imagen.png").resolve()
    imagen.save(ruta)
    print(f"Imagen guardada en: {ruta}")
