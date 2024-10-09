import streamlit as st
import time 
import base64
import requests
import random
import numpy as np
import cv2
import gc
from PIL import Image

def resize_img_with_padding(im, target_size:tuple):
  # im = Image.open(img_path)
  # im = cv2.imread(img_path)
  # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

  h, w, _ = im.shape  # old_size[0] is in (width, height) format
  old_size = (w,h)
  # desired_size = target_size

  ratio = min([y/x for x,y in zip(old_size,target_size)])
  new_size = tuple([int(x*ratio) for x in old_size])

  im = cv2.resize(im, new_size)
  # create a new image and paste the resized on it

  new_im = Image.new("RGB", target_size)
  new_im.paste(
      Image.fromarray(im),
      (
        max(0,target_size[0]-new_size[0])//2,
        max(0,target_size[1]-new_size[1])//2,
      )
  )
  # del im
  # clear_mem()
  return np.array(new_im) #, new_size, ratio

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
   
def draw_one_bbox(img, xyxy, label, color = (255,200,150), thickness=3, draw_mask = False):
  x1, y1, x2, y2 = xyxy
  if not draw_mask:
    img = cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    img = cv2.putText(
          img,
          str(label),
          (x1, y1 - 10),
          fontFace = cv2.FONT_HERSHEY_SIMPLEX,
          fontScale = 0.6,
          color = (255, 255, 255),
          thickness=2
      )
  else:
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    alpha = 0.6  # Transparency factor.

    # Following line overlays transparent rectangle
    # over the image
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
  return img

def draw_bboxes(img, boxlist, color = (255,200,150), thickness=3, color_space=None, draw_mask=False, color_list=None):
  img = img.copy()
  for idx, xyxy in enumerate(boxlist):
    color_ = color if color_list is None else color_list[idx]
    img = draw_one_bbox(img, xyxy, 0, color_, thickness, draw_mask)
  return img

# =====================================================
st.set_page_config(layout='wide', page_title='Detect Anything')

with st.sidebar:
    server_url = st.text_input("Server URL",value="")
    set_button = st.button("Apply")

if set_button and server_url!="":
    st.session_state["server_url"] = server_url
    st.toast('Successfully Apply')
    time.sleep(.5)


with st.expander("Send Request"):
    col1, col2 = st.columns(2)
    with col1:
        st.title("Query Image")
        query_img = st.file_uploader("Image",key="query_img_key")
        if query_img is not None:
            st.image(query_img.getvalue())
            with open(f"query.jpg", "wb+") as fh:
                fh.write(query_img.getvalue())

    with col2:
        st.title("Ref Image")
        ref_img = st.file_uploader("Image",key="ref_img_key")
        if ref_img is not None:
            st.image(ref_img.getvalue())
    submit = st.button(label="Process")

allow_show_img = False
if submit:
    allow_show_img = True
    with st.status("Processing data data...") as status:
        start = time.time()
        query_img_encode = base64.b64encode(query_img.getvalue()).decode('utf-8')
        ref_img_encode = base64.b64encode(ref_img.getvalue()).decode('utf-8')
        # st.write(ref_img_encode)
        # st.image(BytesIO(base64.b64decode(ref_img_encode)))
        # with open("test.jpg","wb+") as f:
        #     f.write(base64.b64decode(ref_img_encode))

        status.update(
            label="Send request to AI", state="running", expanded=False
        )

        data = {
            'query_image':query_img_encode,
            'ref_image':ref_img_encode,
        }

        # st.image(read_img('query.jpg'))

        status.update(
            label="AI is working", state="running", expanded=False
        )
        
        x = requests.post(f'{st.session_state["server_url"]}/predict',
                        json = data,
                        headers = {
                            'ngrok-skip-browser-warning': '1'
                        },
                        timeout=120,
                        )
        
        # status.update(
        #     label="AI is working", state="running", expanded=False
        # )
        return_data = x.json()
        during = time.time()-start
        if 'return_data' not in st.session_state:
            st.session_state["return_data"] = return_data
            st.session_state["return_data"]['during'] = during
        else:
            st.session_state["return_data"] = return_data
            st.session_state["return_data"]['during'] = during

        status.update(
            label="Processing complete!", state="complete", expanded=False
        )
        # print(return_data)
        # if 'return_data' not in st.session_state:
            
    

if st.session_state.get("return_data") is not None:
    with st.expander("Postproccess"):
        valid_conf = st.slider("Valid Confident Score", 0.0, 1.0, 0.5, 0.01)
        minimum_conf = st.slider("Minimum Confident Score", 0.0, 1.0, 0.3, 0.01)
        assert minimum_conf < valid_conf, "Minimum Confident Score < Valid Confident Score"
        if st.button("Show Image"):
            allow_show_img = True

if allow_show_img:
    allow_show_img = False
    return_data = st.session_state.get("return_data")
    index_valid = np.array(return_data['scores']) > minimum_conf
    conf = np.array(return_data['scores'])[index_valid]
    col11, col12, col13, col14, col15 = st.columns(5)
    # st.write(f"Number of objects: {return_data['num_obj']}")
    # st.write(f"Confident: {np.array(return_data['box_conf_list']).mean()}")
    col11.metric("Number of objects", f"{(conf>valid_conf).sum()}")
    col12.metric("Total", f"{index_valid.sum()}")
    if len(conf[conf>valid_conf]) > 0:
        col13.metric("Confident", f"{conf[conf>valid_conf].mean()*100: .0f}% +/-{conf[conf>valid_conf].std()*100: .0f}%")
    else:
        col13.metric("Confident", f"{0: .0f}% +/-{0: .0f}%")
    col14.metric("Time Response", f"{return_data['during']: .2f}s")
    col15.metric("Time Proccess", f"{return_data['proccesing_time']: .2f}s")
    # st.image(BytesIO(base64.b64decode(return_data['img'])))

    h,w = return_data['box_img_shape'][1:]
    print(return_data['box_img_shape'])

    # mimg = draw_masks_fromList(
    #     cv2.resize(read_img('query.jpg'),(w,h)),
    #     # list(range(len(return_data['contour_list']))),
    #     list(range(index_valid.sum())),
    #     [np.array(contour) for contour_index, contour in enumerate(return_data['contour_list']) if index_valid[contour_index] == True],
    #     return_data['box_img_shape'][:-1],
    #     labels=(np.array(return_data['scores'])[index_valid]>valid_conf).astype(int).reshape(-1,1),
    #     # labels=[[0]]*len(mask_list[0]),
    #     colors=[(152, 43, 28), (197, 255, 149)],
    #     alpha = 0.4,
    #     contour_color = (2, 21, 38), contour_line_weight = 3
    # )
    # print(f"bbox:{len([ return_data['bbox_list'][i] for i in index_valid])}")
    pallet_colors=[(152, 43, 28), (197, 255, 149)]
    mimg = draw_bboxes(
       resize_img_with_padding(read_img('query.jpg'),(h,w)), 
       np.array(return_data['bbox_list'])[index_valid].tolist(), 
       color = (255,200,150), 
       thickness=3, 
       draw_mask=True, 
       color_list=[ pallet_colors[i] for i in  (np.array(return_data['scores'])[index_valid]>valid_conf).astype(int).tolist() ])

    st.image(cv2.resize(mimg, None, fx=0.6,fy=0.5))

    # data ={
    #     'ID': user_id,
    #     'bbox_list':bbox_list,
    #     'scores':scores,
    #     'box_img_shape':box_img_shape,
    #     'proccesing_time':end_time-start_time,
    # }

    

    # u = Image.fromarray(mimg)
    
