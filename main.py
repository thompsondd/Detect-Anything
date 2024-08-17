import streamlit as st
import time 
import base64
import requests
from io import BytesIO
import numpy as np

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
    with col2:
        st.title("Ref Image")
        ref_img = st.file_uploader("Image",key="ref_img_key")
        if ref_img is not None:
            st.image(ref_img.getvalue())
    submit = st.button(label="Process")

if submit:
    with st.status("Processing data data...") as status:
        start = time.time()
        query_img_encode = base64.b64encode(query_img.getvalue()).decode('utf-8')
        ref_img_encode = base64.b64encode(ref_img.getvalue()).decode('utf-8')
        # st.write(ref_img_encode)
        # st.image(BytesIO(base64.b64decode(ref_img_encode)))
        # with open("test.jpg","wb+") as f:
        #     f.write(base64.b64decode(ref_img_encode))
        data = {
            'query_image':query_img_encode,
            'ref_image':ref_img_encode,
        }

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
        return_data = x.json()

        status.update(
            label="Processing complete!", state="complete", expanded=False
        )
    during = time.time()-start
    conf = np.array(return_data['box_conf_list'])
    # st.write(f"Number of objects: {return_data['num_obj']}")
    # st.write(f"Confident: {np.array(return_data['box_conf_list']).mean()}")
    col11, col12, col13 = st.columns(3)
    col11.metric("Number of objects", f"{return_data['num_obj']}")
    col12.metric("Confident", f"{conf.mean()*100: .0f}% +/-{conf.std()*100: .0f}%")
    col13.metric("Time Response", f"{during: .2f}s")
    st.image(BytesIO(base64.b64decode(return_data['img'])))
    