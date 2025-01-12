import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model with trust_remote_code if needed
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
url = "https://www.nvidia.com/en-us/project-digits/"

try:
    # Download the page
    response = requests.get(url)
    response.raise_for_status()
except requests.RequestException as e:
    print(f"Failed to retrieve the web page: {e}")
    exit(1)

# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all img elements on the page
img_elements = soup.find_all('img')

if not img_elements:
    print("No images found on the page.")
    exit(1)

captions_list = []  # List to store generated captions

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
    # Iterate over each img element
    for img_element in img_elements:
        img_url = img_element.get('src')

        # Skip if the image URL is missing, is an SVG, or too small
        if not img_url or 'svg' in img_url or '1x1' in img_url:
            continue

        # Correct the URL if it's protocol-relative
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        # Skip URLs that don't start with http:// or https://
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue

        try:
            # Download the image
            img_response = requests.get(img_url)
            img_response.raise_for_status()

            # Convert the image data to a PIL Image
            raw_image = Image.open(BytesIO(img_response.content))
            # Skip very small images likely to be icons
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue

            raw_image = raw_image.convert('RGB')

            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            output_tokens = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(output_tokens[0], skip_special_tokens=True)

            # Store and write the caption
            captions_list.append((img_url, caption))
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

# Print the list of generated descriptions
for img_url, caption in captions_list:
    print(f"{img_url}: {caption}")
