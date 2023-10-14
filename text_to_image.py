from PIL import Image

from stable_diffusion_xl.stable_diffusion_xl import StableDiffusionXL

model = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)
img = model.text_to_image(
    "a cute girl.",
    num_steps=25,
    seed=123456,
)
Image.fromarray(img[0]).save("girl.jpg")
print("Saved at girl.jpg")
