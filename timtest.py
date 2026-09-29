from PIL import Image

input_file = "Bar/Texture/wood 3.png"
output_file = "Bar/Texture/wood_fixed.png"

image = Image.open(input_file)

print("Detected format:", image.format)

image.save(output_file, "PNG")

print("Created:", output_file)