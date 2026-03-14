import os
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\akshi\Documents\Ai crop proj\dataset\PlantVillage_Color"

# List classes
classes = os.listdir(DATASET_PATH)
print("Total classes:", len(classes))

# Count images
total_images = 0
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    num_images = len(os.listdir(cls_path))
    total_images += num_images
    print(f"{cls}: {num_images}")

print("\nTotal images:", total_images)

# Show sample images
plt.figure(figsize=(10, 10))
for i, cls in enumerate(classes[:9]):
    img_path = os.path.join(DATASET_PATH, cls, os.listdir(os.path.join(DATASET_PATH, cls))[0])
    img = Image.open(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.show()
