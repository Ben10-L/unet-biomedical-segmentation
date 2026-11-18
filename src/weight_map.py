import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_weight_map(instance_mask, w0=10, sigma=5):

    instance_mask = np.asarray(instance_mask)

    if instance_mask.ndim != 2:
        raise ValueError("Expected a 2D instance mask")

    # Binary segmentation mask
    mask = (instance_mask > 0).astype(np.int64)

    # 1. Class balancing
    classes, counts = np.unique(
        mask,
        return_counts=True
    )

    wc = np.zeros(mask.shape, dtype=np.float32)

    for cls, count in zip(classes, counts):

        class_weight = mask.size / (
            len(classes) * count
        )

        wc[mask == cls] = class_weight

    # 2. Identify individual cells
    cell_ids = np.unique(instance_mask)
    cell_ids = cell_ids[cell_ids > 0]

    # At least two cells are needed
    if len(cell_ids) < 2:
        return wc

    # 3. Distance to each cell
    distances = []

    for cell_id in cell_ids:

        cell = instance_mask == cell_id

        distance = distance_transform_edt(~cell)

        distances.append(distance)

    distances = np.stack(distances, axis=0)

    # 4. Two nearest cells
    distances = np.sort(distances, axis=0)

    d1 = distances[0]
    d2 = distances[1]

    # 5. Border weighting
    border_weight = w0 * np.exp(
        -((d1 + d2) ** 2) / (2 * sigma ** 2)
    )

    # Apply extra weights to background pixels
    border_weight *= (mask == 0)

    # 6. Final weight map
    weight_map = wc + border_weight

    return weight_map.astype(np.float32)