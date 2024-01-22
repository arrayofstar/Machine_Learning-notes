# -*- coding: utf-8 -*-
# @Time    : 2023/6/15 15:21
# @Author  : Dreamstar
# @File    : day29-Hugging Face-搭建零样本学习文本分类器.py
# @Link    : https://30days.streamlit.app/?challenge=Day29
# @Desc    : 使用 Hugging Face 和 Streamlit 开发一个零样本学习的文本分类器，
#            使用 Hugging Face 的 API 参考以及 Distilbart 创建一个零样本学习的文本分类器

import pandas as pd
# Import for API calls
import requests
import streamlit as st
# Imports for aggrid
from st_aggrid import AgGrid
from st_aggrid import GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
# Import for navbar
from streamlit_option_menu import option_menu
# Import for dynamic tagging
from streamlit_tags import st_tags

# Import for loading interactive keyboard shortcuts into the app
from dashboard_utils.gui import load_keyboard_class

##############################################################################################################
# 控制整体布局
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"
##############################################################################################################
# 设置布局、标题和logo
st.set_page_config(layout=layout, page_title="Zero-Shot Text Classifier", page_icon="🤗")

# The class below is for adding some formatting to the shortcut button on the left sidebar.
load_keyboard_class()
##############################################################################################################
c1, c2 = st.columns([0.4, 2])
# The block of code below is to display the title, logos and introduce the app.

c1, c2 = st.columns([0.4, 2])

with c1:
    st.image("logo.png",width=110,)
with c2:
    st.caption("")
    st.title("Zero-Shot Text Classifier")

st.write("")
st.markdown(
    """

Classify keyphrases fast and on-the-fly with this mighty app. No ML training needed!

Create classifying labels (e.g. `Positive`, `Negative` and `Neutral`), paste your keyphrases, and you're off!

"""
)

st.write("")
st.sidebar.image(
    "30days_logo.png",
)
st.sidebar.write("")
##############################################################################################################
# 侧边导航
with st.sidebar:
    selected = option_menu(
        "",
        ["Demo", "Unlocked Mode"],
        icons=["bi-joystick", "bi-key-fill"],
        menu_icon="",
        default_index=0,
    )

# The code below is to display the shortcuts.
st.sidebar.header("Shortcuts")
st.sidebar.write(
    '<span class="kbdx">G</span>  &nbsp; GitHub',
    unsafe_allow_html=True,
)

st.sidebar.write(
    '<span class="kbdx">&thinsp;.&thinsp;</span>  &nbsp; GitHub Dev (VS Code)',
    unsafe_allow_html=True,
)

# The block of code below is to display information about Streamlit.

st.sidebar.markdown("---")

# Sidebar
st.sidebar.header("About")

st.sidebar.markdown(
    """

App created by [Datachaz](https://twitter.com/DataChaz) using 🎈[Streamlit](https://streamlit.io/) and [HuggingFace](https://huggingface.co/inference-api)'s [Distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3) model.

"""
)

st.sidebar.markdown(
    "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
)

st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
- [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
- [Blog](https://blog.streamlit.io/how-to-master-streamlit-for-data-science/) (How to master Streamlit for data science)
"""
)

st.sidebar.header("Deploy")
st.sidebar.markdown(
    "You can quickly deploy Streamlit apps using [Streamlit Cloud](https://streamlit.io/cloud) in just a few clicks."
)
##############################################################################################################


def main():
    st.caption("")


if selected == "Demo":
    # ADD CODE FOR DEMO HERE
    API_KEY = st.secrets["API_KEY"]

    API_URL = (
        "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
    )

    headers = {"Authorization": f"Bearer {API_KEY}"}

    with st.form(key="my_form"):
        multiselectComponent = st_tags(
            label="",
            text="Add labels - 3 max",
            value=["Transactional", "Informational"],
            suggestions=[
                "Navigational",
                "Transactional",
                "Informational",
                "Positive",
                "Negative",
                "Neutral",
            ],
            maxtags=3,
        )

        new_line = "\n"
        nums = [
            "I want to buy something in this store",
            "How to ask a question about a product",
            "Request a refund through the Google Play store",
            "I have a broken screen, what should I do?",
            "Can I have the link to the product?",
        ]

        sample = f"{new_line.join(map(str, nums))}"

        MAX_LINES = 5
        text = st.text_area(
            "Enter keyphrase to classify",
            sample,
            height=200,
            key="2",
            help="At least two keyphrases for the classifier to work, one per line, "
                 + str(MAX_LINES)
                 + " keyphrases max as part of the demo",
        )
        lines = text.split("\n")  # A list of lines
        linesList = []
        for x in lines:
            linesList.append(x)
        linesList = list(dict.fromkeys(linesList))  # Remove dupes
        linesList = list(filter(None, linesList))  # Remove empty

        if len(linesList) > MAX_LINES:
            st.info(
                f"🚨 Only the first "
                + str(MAX_LINES)
                + " keyprases will be reviewed. Unlock that limit by switching to 'Unlocked Mode'"
            )
        linesList = linesList[:MAX_LINES]

        submit_button = st.form_submit_button(label="Submit")

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning("❄️ There is no keyphrases to classify")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and not multiselectComponent:
        st.warning("❄️ You have not added any labels, please add some! ")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and len(multiselectComponent) == 1:
        st.warning("❄️ Please make sure to add at least two labels for classification")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:
            st.session_state.valid_inputs_received = True

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            # Unhash to check status codes from the API response
            # st.write(response.status_code)
            return response.json()

        listtest = ["I want a refund", "I have a question"]
        listToAppend = []

        for row in linesList:
            output2 = query(
                {
                    "inputs": row,
                    "parameters": {"candidate_labels": multiselectComponent},
                    "options": {"wait_for_model": True},
                }
            )

            listToAppend.append(output2)

            df = pd.DataFrame.from_dict(output2)

        st.success("✅ Done!")

        df = pd.DataFrame.from_dict(listToAppend)

        st.caption("")
        st.markdown("### Check classifier results")
        st.caption("")

        st.checkbox(
            "Widen layout",
            key="widen",
            help="Tick this box to toggle the layout to 'Wide' mode",
        )

        st.caption("")

        # This is a list comprehension to convert the decimals to percentages
        f = [[f"{x:.2%}" for x in row] for row in df["scores"]]

        # This code is for re-integrating the labels back into the dataframe
        df["classification scores"] = f
        df.drop("scores", inplace=True, axis=1)

        # This code is to rename the columns
        df.rename(columns={"sequence": "keyphrase"}, inplace=True)

        # The code below is for Ag-grid

        gb = GridOptionsBuilder.from_dataframe(df)
        # enables pivoting on all columns
        gb.configure_default_column(
            enablePivot=True, enableValue=True, enableRowGroup=True
        )
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.configure_side_bar()
        gridOptions = gb.build()

        response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            height=300,
            fit_columns_on_grid_load=False,
            configure_side_bar=True,
        )

        # The code below is for the download button

        cs, c1 = st.columns([2, 2])

        with cs:

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode("utf-8")

            csv = convert_df(df)  #

            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="results.csv",
                mime="text/csv",
            )

elif selected == "Unlocked Mode":
    # ADD CODE FOR 'Unlocked Mode' HERE
    pass

api_token = st.secrets["API_TOKEN"]


API_KEY = st.secrets["API_KEY"]

API_URL = (
        "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
    )

headers = {"Authorization": f"Bearer {API_KEY}"}


if __name__ == "__main__":
    main()