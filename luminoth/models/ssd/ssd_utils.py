import numpy as np


def generate_anchors_reference(ratios, scales, num_anchors, endpoint_output):
    """
    For each ratio we will get an anchor TODO
    Args: TODO
    Returns: convention (x_min, y_min, x_max, y_max) TODO
    """
    heights = np.zeros(num_anchors)
    widths = np.zeros(num_anchors)
    # Because the ratio of 1 we will use the scale sqrt(scale[i] * scale[i+1])
    #  or sqrt(scale[i_max] * scale[i_max]). So we will have to use just
    # `num_anchors` - 1 ratios to generate the anchors
    if scales.shape[0] > 1:
        widths[0] = heights[0] = (np.sqrt(scales[0] * scales[1]) *
                                  endpoint_output[0])
    else:
        widths[0] = heights[0] = scales[0]
    ratios = ratios[:num_anchors - 1]
    heights[1:] = scales[0] / np.sqrt(ratios) * endpoint_output[0]
    widths[1:] = scales[0] * np.sqrt(ratios) * endpoint_output[1]

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    return anchors
