import numpy as np
from scipy.ndimage import map_coordinates


def elastic_deformation(image, mask, sigma=10, seed=None):

    if image.ndim != 2 or image.shape != mask.shape:
        raise ValueError("Expected matching 2D image and mask")

    rng = np.random.default_rng(seed)

    h, w = image.shape

    # 1. Random displacement on a 3x3 grid
    dx_grid = rng.normal(0, sigma, (3, 3))
    dy_grid = rng.normal(0, sigma, (3, 3))

    # 2. Dense interpolation coordinates
    gy, gx = np.meshgrid(
        np.linspace(0, 2, h),
        np.linspace(0, 2, w),
        indexing="ij"
    )

    # 3. Cubic interpolation
    dx = map_coordinates(
        dx_grid, [gy, gx],
        order=3, mode="reflect"
    )

    dy = map_coordinates(
        dy_grid, [gy, gx],
        order=3, mode="reflect"
    )

    # 4. Original pixel coordinates
    y, x = np.meshgrid(
        np.arange(h),
        np.arange(w),
        indexing="ij"
    )

    coordinates = [y + dy, x + dx]

    # 5. Transform the image
    deformed_image = map_coordinates(
        image.astype(np.float32),
        coordinates,
        order=1,
        mode="reflect"
    )

    # 6. Transform the segmentation mask
    deformed_mask = map_coordinates(
        mask,
        coordinates,
        order=0,
        mode="reflect"
    )

    return deformed_image, deformed_mask