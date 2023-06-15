# -*- coding: utf-8 -*-
# @Time    : 2023/6/15 16:28
# @Author  : Dreamstar
# @File    : day30-streamlit应用之艺术.py
# @Link    : https://30days.streamlit.app/?challenge=Day30
# @Desc    : 创建一个解决真实世界问题的 Streamlit 应用。
#            1. 接收用户输入的 YouTube 链接
#            2. 对链接进行文本处理，提取出 YouTube 视频独特的标识 ID
#            3. 用这个 YouTube 视频的 ID 作为一个自定义函数的输入，获取然后显示 YouTube 视频的缩略图


import streamlit as st

st.title('🖼️ yt-img-app')
st.header('YouTube Thumbnail Image Extractor App - YouTube缩略图提取程序')
with st.expander('About this app - 关于这个APP'):
    st.write('This app retrieves the thumbnail image from a YouTube video. - 此应用程序从YouTube视频中检索缩略图')
# 图片设置
st.sidebar.header("Settings")
img_dict = {'Max': 'maxresdefault', 'High': 'hqdefault', 'Medium': 'mqdefault', 'Standard': 'sddefault'}
selected_img_quality = st.sidebar.selectbox('Select image quality', ['Max', 'High', 'Medium', 'Standard'])
img_quality = img_dict[selected_img_quality]

yt_url = st.text_input('Paste YouTube URL', 'https://youtu.be/JwSS70SZdyM')


def get_ytid(input_url):
    if 'youtu.be' in input_url:
        ytid = input_url.split('/')[-1]
    if 'youtube.com' in input_url:
        ytid = input_url.split('=')[-1]
    return ytid

# Display YouTube thumbnail image
if yt_url != '':
    ytid = get_ytid(yt_url) # yt or yt_url
    yt_img = f'http://img.youtube.com/vi/{ytid}/{img_quality}.jpg'
    st.image(yt_img)
    st.write('YouTube video thumbnail image URL: ', yt_img)
else:
    st.write('☝️ Enter URL to continue ...')


def main():
    st.markdown("")


if __name__ == '__main__':
    main()