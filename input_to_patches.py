# Just an example:

# Create example values
# height = 224 # H ("The training resolution is 224.")
# width = 224 # W
# color_channels = 3 # C
# patch_size = 16 # P

# # Calculate N (number of patches)
# number_of_patches = int((height * width) / patch_size**2)
# print(f"Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}")

# # Input shape (this is the size of a single image)
# embedding_layer_input_shape = (height, width, color_channels)

# # Output shape
# embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)

# print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
# print(f"Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}")