# Try Mask Sampling of the input image


```python
!pip install openai
```

    Collecting openai
      Downloading openai-0.27.8-py3-none-any.whl (73 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m73.6/73.6 kB[0m [31m4.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests>=2.20 in /usr/local/lib/python3.10/dist-packages (from openai) (2.27.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai) (4.65.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from openai) (3.8.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (1.26.16)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2023.5.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.4)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (23.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (6.0.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.9.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.3.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.3.1)
    Installing collected packages: openai
    Successfully installed openai-0.27.8



```python
!nvidia-smi
```

    Fri Jul  7 09:25:52 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   44C    P8    12W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+



```python
import os
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import openai
import base64
from PIL import Image
import imageio
import numpy


os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')
```


```python
# Create a function to create a binary mask
def create_mask(image_path, alpha, mask_path):
    image = Image.open(image_path)
    np_image = np.array(image)
    np_mask = np.random.choice([0, 1], size=np_image.shape[:2], p=[1-alpha, alpha])
    masked_image = np_image * np_mask[:, :, None]
    mask = Image.fromarray(masked_image.astype(np.uint8))
    mask.save(mask_path)

def generate_image(input_image_path, mask_image_path, prompt):
    with open(input_image_path, "rb") as image_file, open(mask_image_path, "rb") as mask_file:
        response = openai.Image.create_edit(
            image=image_file.read(),  # The original image
            mask=mask_file.read(),  # The mask
            prompt=prompt,  # The description of the full new image
            n=1,
            size='256x256'
        )
    image_url = response['data'][0]['url']
    return image_url



def convert_image_to_rgba(image_path):
    image = Image.open(image_path)
    rgba_image = image.convert('RGBA')
    rgba_image_path = image_path.rsplit('.', 1)[0] + '_rgba.png'
    rgba_image.save(rgba_image_path)
    return rgba_image_path
```


```python
# Generate and save multiple images using DALL-E
numbers_generated = 20
image_urls = []
directory = "./generated_images"
if not os.path.exists(directory):
    os.makedirs(directory)

input_image_path = "./after_inpainting.png"
mask_image_path = "./mask.png"

# Create a mask
create_mask(input_image_path, 0.75, mask_image_path)

# Convert input image and mask to RGBA
input_image_path = convert_image_to_rgba(input_image_path)
mask_image_path = convert_image_to_rgba(mask_image_path)

# Now you can call your generate_image function
image_url = generate_image(input_image_path, mask_image_path, "a building on fire")
```


```python
# Generate images
import time

start_time = time.time()

for i in range(numbers_generated):
    image_url = generate_image(input_image_path, mask_image_path, "a building on fire")
    image_urls.append(image_url)

    # Download the image from the URL
    image_bytes = requests.get(image_url).content

    # Convert the image bytes to a PIL Image object
    image = Image.open(BytesIO(image_bytes))

    # Save the image to a file
    image.save(os.path.join(directory, f"generated_image_{i}.png"))

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
```

    Execution time: 251.7172327041626 seconds


## Plot the images that are generated using `matplotlib`


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

directory = "./generated_images"
num_images = 20

fig, axs = plt.subplots(4, 5, figsize=(14, 12))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

for i in range(num_images):
    row = i // 5
    col = i % 5

    img = mpimg.imread(os.path.join(directory, f'generated_image_{i}.png'))
    axs[row, col].imshow(img)
    axs[row, col].axis('off')
    axs[row, col].set_aspect('equal')  # Maintain aspect ratio
    axs[row, col].set_adjustable('box')  # Prevent image from stretching

plt.show()

```


    
![png](Mask_Sampling_inpainted_input_files/Mask_Sampling_inpainted_input_8_0.png)
    


## Generate GIF using Interpolation


```python
import imageio
import numpy as np

# Read images
images = []
for i in range(numbers_generated):
    images.append(imageio.imread(os.path.join(directory, f"generated_image_{i}.png")))

# Interpolate
interpolated_images = []
num_inter_frames = 5  # Number of interpolated frames between each pair of consecutive frames
for i in range(len(images) - 1):
    img1 = images[i]
    img2 = images[i+1]

    # Interpolate and add inter frames
    for t in range(num_inter_frames):
        alpha = t / num_inter_frames
        interpolated_img = img1 * (1 - alpha) + img2 * alpha
        interpolated_images.append(interpolated_img.astype(np.uint8))

# Add the last image
interpolated_images.append(images[-1])

# Define the duration for each frame
durations = [0.1, 0.2, 0.5]

# Save as GIF
for i, duration in enumerate(durations):
    filename = f'generated_images_duration_{duration}.gif'
    imageio.mimsave(os.path.join(directory, filename), interpolated_images, duration=duration)

```

    <ipython-input-22-3d760341f175>:7: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      images.append(imageio.imread(os.path.join(directory, f"generated_image_{i}.png")))



```python

```
