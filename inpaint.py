from PIL import Image

from stable_diffusion_xl.stable_diffusion_xl import StableDiffusionXL

model = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)
img = model.inpaint(
    "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.",
    reference_image="/path/to/dog.jpg",
    inpaint_mask="/path/to/dog_mask.png",
    mask_blur_strength=5,
    unconditional_guidance_scale=8.0,
    reference_image_strength=0.9,
    num_steps=50,
)
Image.fromarray(img[0]).save("out.jpg")
print("Saved at out.jpg")
