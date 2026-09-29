"""Image/depth preprocessing shared with the backbones' own input pipeline.

Only the three helpers the CO3Dv2 calibration loader needs are here.

Treat them as fixed, not as replaceable "utility code" -- the
518px / divisible-by-14 resize convention below has to agree exactly with
`vggt.utils.load_fn.load_and_preprocess_images`, which preprocesses the IMAGES the same
scene's depths and intrinsics are paired with, and the shipped Fisher head-importance
tables were calibrated through this exact arithmetic. Rewriting it "more cleanly" would
silently re-scale the calibration.
"""

import cv2
import numpy as np
import open3d as o3d
import torch

# VGGT's input resolution, and the patch size its resize convention keeps images
# divisible by. Both are the backbone's, not ours -- see the module docstring.
TARGET_SIZE = 518
PATCH_MULTIPLE = 14


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    return pts


def update_K(K, height, width, mode="crop"):
    """Rescale a pinhole intrinsic matrix through the same resize+crop/pad the images go
    through, so `K` still describes the preprocessed image rather than the original."""
    K_new = K.copy()

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    target_size = TARGET_SIZE

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / PATCH_MULTIPLE) * PATCH_MULTIPLE
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / PATCH_MULTIPLE) * PATCH_MULTIPLE
    else:  # mode == "crop"
        new_width = target_size
        new_height = round(height * (new_width / width) / PATCH_MULTIPLE) * PATCH_MULTIPLE

    # Update K -> scaling
    sx = new_width / width
    sy = new_height / height
    K_new[0, 0] *= sx  # fx
    K_new[1, 1] *= sy  # fy
    K_new[0, 2] *= sx  # cx
    K_new[1, 2] *= sy  # cy

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        K_new[1, 2] -= start_y

    # For pad mode, pad to make a square of target_size x target_size
    if mode == "pad":
        h_padding = target_size - new_height
        w_padding = target_size - new_width
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_left = w_padding // 2
            K_new[0, 2] += pad_left
            K_new[1, 2] += pad_top

    return K_new


def load_and_preprocess_depths(depth_path_list, mode="crop", convert_to_float16=False):
    """Depth counterpart of `vggt.utils.load_fn.load_and_preprocess_images`: same resize,
    crop and pad decisions, so depth[i] stays pixel-aligned with image[i].

    `convert_to_float16=True` reinterprets CO3Dv2's depth PNGs, which store float16
    bit patterns in a 16-bit integer image, rather than casting the integers.
    """
    # Check for empty list
    if len(depth_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    depths = []
    shapes = set()
    target_size = TARGET_SIZE

    # First process all images and collect their shapes
    for image_path in depth_path_list:
        # Open image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if convert_to_float16:
            img = img.view(np.float16)
            img = img.astype(np.float32)
        height, width = img.shape

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                # Make divisible by 14
                new_height = round(height * (new_width / width) / PATCH_MULTIPLE) * PATCH_MULTIPLE
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / PATCH_MULTIPLE) * PATCH_MULTIPLE
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / PATCH_MULTIPLE) * PATCH_MULTIPLE

        # Resize with new dimensions (width, height)
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)
        img = torch.from_numpy(np.array(img)).float().unsqueeze(0)  # Convert to tensor (0, 255)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        depths.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in depths:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        depths = padded_images

    depths = torch.stack(depths)  # concatenate images

    # Ensure the shape is (1, C, H, W) when a single depth was requested
    if len(depth_path_list) == 1 and depths.dim() == 3:
        depths = depths.unsqueeze(0)

    masks = []
    for depth in depths:
        masks.append(depth > 0)
    masks = torch.stack(masks)

    return depths, masks
