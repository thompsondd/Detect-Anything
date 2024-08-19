import streamlit as st
import time 
import base64
import requests
from io import BytesIO
import numpy as np
from utils import draw_masks_fromList, read_img
import cv2

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

    h,w = return_data['box_img_shape'][:-1]

    mimg = draw_masks_fromList(
        cv2.resize(read_img('query.jpg'),(w,h)),
        # list(range(len(return_data['contour_list']))),
        list(range(index_valid.sum())),
        [np.array(contour) for contour_index, contour in enumerate(return_data['contour_list']) if index_valid[contour_index] == True],
        return_data['box_img_shape'][:-1],
        labels=(np.array(return_data['scores'])[index_valid]>valid_conf).astype(int).reshape(-1,1),
        # labels=[[0]]*len(mask_list[0]),
        colors=[(152, 43, 28), (197, 255, 149)],
        alpha = 0.4,
        contour_color = (2, 21, 38), contour_line_weight = 3
    )

    st.image(cv2.resize(mimg, None, fx=0.6,fy=0.5))

    # data ={
    #     'ID': user_id,
    #     'bbox_list':bbox_list,
    #     'contour_list':contour_list,
    #     'scores':scores,
    #     'box_img_shape':box_img_shape,
    #     'proccesing_time':end_time-start_time,
    # }

    

    # u = Image.fromarray(mimg)
    
