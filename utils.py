import cv2
import gc
import numpy as np

def read_img(img_path):
  return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

def draw_masks_fromList(
    image, chosen_index, contours, origin_size_mask,
    labels, colors, alpha = 0.4,
    contour_color = (0,0,0), contour_line_weight = 3):
  masked_image = image.copy()
  contour_list = []
  for i, mask_index in enumerate(chosen_index):
    contour = contours[mask_index]
    contour_list.append(contour)
    mask = cv2.drawContours(np.zeros(origin_size_mask), [contour], -1, (255), -1)
    # mask[offset_masks[i][0]:offset_masks[i][1],...] = masks_generated[i]

    if mask.shape[0]!= image.shape[0] or mask.shape[1]!= image.shape[1]:
      mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # bbox, max_contour = mask_to_bbox(mask, return_contour=True)
    # contour_list.append(max_contour)

    masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                            np.asarray(colors[int(labels[i][-1])], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)
    del mask

  gc.collect()

  image = cv2.addWeighted(image, alpha, masked_image, (1-alpha), 0)
  image = cv2.drawContours(image, contour_list, -1, contour_color, contour_line_weight)

  return image
