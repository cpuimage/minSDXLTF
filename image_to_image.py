from PIL import Image

from stable_diffusion_xl.stable_diffusion_xl import StableDiffusionXL

model = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)
img = model.image_to_image(
    "a cute girl.",
    unconditional_guidance_scale=7.5,
    reference_image="/path/to/a_girl.jpg",
    reference_image_strength=0.8,
    num_steps=50,
)
Image.fromarray(img[0]).save("out.jpg")
print("Saved at out.jpg")
