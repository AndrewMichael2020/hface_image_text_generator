import os
import glob
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load the pretrained processor and model with trust_remote_code=True
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", trust_remote_code=True
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", trust_remote_code=True
)

# Specify the directory where your images are
image_dir = "/img"
image_exts = ["jpg", "jpeg", "png"]  # specify the image file extensions to search for

# Check if the image directory exists
if not os.path.isdir(image_dir):
    print(f"Directory {image_dir} does not exist.")
    exit(1)

# Gather all image paths from the directory with given extensions
images = []
for ext in image_exts:
    images.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))

# Check if any images were found
if not images:
    print(f"No images found in directory {image_dir}.")
    exit(1)

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
    # Iterate over each image file found
    for img_path in images:
        try:
            # Load your image
            raw_image = Image.open(img_path).convert('RGB')

            # Process the image for captioning
            inputs = processor(raw_image, return_tensors="pt")

            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)

            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Write the caption to the file, prepended by the image file name
            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
