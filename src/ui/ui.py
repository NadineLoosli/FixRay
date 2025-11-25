# filepath: fixray-streamlit-app/fixray-streamlit-app/src/components/ui.py

import streamlit as st

def create_slider(label, min_value, max_value, default_value, step):
    return st.slider(label, min_value=min_value, max_value=max_value, value=default_value, step=step)

def create_button(label, key=None):
    return st.button(label, key=key)

def display_image(image, caption):
    st.image(image, caption=caption, use_column_width=True)

def show_message(message, message_type='info'):
    if message_type == 'success':
        st.success(message)
    elif message_type == 'error':
        st.error(message)
    else:
        st.info(message)