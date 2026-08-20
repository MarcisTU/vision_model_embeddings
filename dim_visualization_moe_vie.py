import colorsys
import os
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from moe_vie.src.open_clip import create_model_and_transforms, get_tokenizer, image_to_device


# Helper function to convert RGB to CIELAB space for visual distance checking
def rgb_to_lab(rgb):
    r, g, b = rgb
    r = (r / 12.92) if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = (g / 12.92) if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = (b / 12.92) if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    fx = (x / 0.95047) ** (1/3) if (x / 0.95047) > 0.008856 else (7.787 * (x / 0.95047)) + (16/116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16/116)
    fz = (z / 1.08883) ** (1/3) if (z / 1.08883) > 0.008856 else (7.787 * (z / 1.08883)) + (16/116)
    
    return (116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz)


if __name__ == "__main__":
    # data from https://www.robots.ox.ac.uk/~vgg/data/pets/
    IMAGE_DATA_PATH = "/home/marcis/dl_projects/dimens_reduct_viz_images/moe_vie/data_test_pets"
    OUTPUT_PATH = "/home/marcis/dl_projects/dimens_reduct_viz_images/moe_vie/outputs"
    MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    BATCH_SIZE = 4

    # inititalize model
    model, _, preprocess = create_model_and_transforms(
        "MoEViE-B16-224",  # other sizes available https://huggingface.co/collections/facebook/moevie
        pretrained=True,
        force_preprocess_cfg=dict(
            patch_size=16, size_range=(224, 224), center_crop=True, window_size=1
        ),
        image_mean=MEAN, image_std=STD,
        device="cuda"
    )
    model = model.cuda().eval()


    # process local image data
    dataset = []
    print(os.getcwd())
    with open(f"{IMAGE_DATA_PATH}/annotations/trainval.txt", "r", encoding="utf-8") as file:
        for idx_l, line in enumerate(file):
            image_name = line.split(" ")[0]
            image_path = f"{IMAGE_DATA_PATH}/images/{image_name}.jpg"
            label_id = int(line.split(" ")[1])
            label = " ".join(image_name.split("_")[:-1])

            dataset.append(
                {
                    "image_path": image_path,
                    "label_id": label_id,
                    "label": label.lower()
                }
            )


    all_embeddings = []
    all_labels = []
    print("Extracting dataset embeddings...")
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i : i + BATCH_SIZE]
        images_batch = [sample["image_path"] for sample in batch]
        labels_batch = [sample["label_id"] for sample in batch]
        
        # Prepare batch samples (convert PIL image to preprocessed tensor with dummy label)
        preprocessed_samples = [
            (preprocess(Image.open(path).convert("RGB")), 0) 
            for path in images_batch
        ]
        
        packed, _ = preprocess.collate_fn(preprocessed_samples)
        packed = image_to_device(packed, "cuda", torch.float32, mean=MEAN, std=STD)

        with torch.no_grad(), torch.autocast("cuda"):
            image_features = model.encode_image(packed, normalize=True)
            all_embeddings.append(image_features.cpu().numpy())

        all_labels.extend(labels_batch)

    # Combine all batch embeddings into a single 2D array [N, embedding_dim]
    embeddings_matrix = np.vstack(all_embeddings)

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_matrix)


    print("Plotting results...")

    id_to_label = {sample["label_id"]: sample["label"] for sample in dataset}
    unique_ids = np.unique(all_labels)
    unique_labels = [id_to_label[label_id] for label_id in unique_ids]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    all_labels_np = np.array(all_labels)
    
    # Generate colors using Golden Ratio hue spacing
    golden_ratio = 0.618033988749895
    colors = [
        colorsys.hsv_to_rgb((i * golden_ratio) % 1.0, 0.8, 0.85) 
        for i in range(len(unique_ids))
    ]

    # Convert generated palette to LAB
    lab_colors = [rgb_to_lab(c) for c in colors]
    
    # Marker pool to rotate through when colors are visually close (deltaE < threshold)
    marker_shapes = ["o", "s", "^", "v", "D", "X", "*"]
    markers = ["o"] * len(unique_ids)
    delta_e_threshold = 28.0  # Perceptual color distance limit

    for i in range(len(unique_ids)):
        for j in range(i):
            # Calculate Euclidean distance in CIELAB (CIE76 delta E)
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(lab_colors[i], lab_colors[j])))
            if dist < delta_e_threshold and markers[i] == markers[j]:
                # Cycle to the next marker shape to break ambiguity
                curr_idx = marker_shapes.index(markers[j])
                markers[i] = marker_shapes[(curr_idx + 1) % len(marker_shapes)]


    for i, label_id in enumerate(unique_ids):
        mask = all_labels_np == label_id
        ax.scatter(
            embedding_2d[mask, 0], 
            embedding_2d[mask, 1], 
            color=colors[i], 
            marker=markers[i],
            label=unique_labels[i], 
            s=18, 
            alpha=0.8
        )

    ax.set_title("UMAP Projection of Oxford-Pets Image Embeddings")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.12), 
        ncol=min(5, len(unique_ids)),  # Adjusts columns dynamically (up to 5)
        frameon=True,
        fontsize="small"
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_PATH}/umap_scatter_pets_moevit.png",
        bbox_inches="tight" # Ensures legend doesn't get clipped on export
    )
