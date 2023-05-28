# 用于处理

import gradio as gr
import pandas as pd


def handle_excel(file, select_sheet):
    df_data = pd.read_excel(file.name, sheet_name=select_sheet, header=0)

    df_data = df_data.iloc[1:, :]

    temp_before = df_data['岩性'].iloc[0]
    start_depth =  df_data['开始深度'].iloc[0]
    df_output = pd.DataFrame([], columns=df_data.columns)
    for idx in range(df_data.shape[0]):
        temp_now = df_data['岩性'].iloc[idx]
        if temp_now != temp_before:
            df = df_data.iloc[idx-1:idx].copy()
            df['开始深度'].loc[idx] = start_depth
            df_output = pd.concat([df_output, df], axis=0)
            start_depth = df_data['开始深度'].iloc[idx]
            temp_before = df_data['岩性'].iloc[idx]
    df_output.to_csv("test.csv", index=False, encoding='utf_8_sig')
    return "## 成功保存至test.csv"


def get_sheet_list(upload_file):
    path = upload_file.name
    df = pd.read_excel(path, sheet_name=None)
    sheet_list = list(df.keys())
    return gr.update(choices=sheet_list), "## 已成功选择文件"


if __name__ == '__main__':
    with gr.Blocks() as main_interface:
        status_bar = gr.Markdown("## 选择你需要处理的xlsx文件和对应的sheet")
        with gr.Column():
            img = gr.Image(value='ref_img.png')
            upload_file = gr.UploadButton(label="选择文件", value='岩心分类方法.xlsx', file_types=[".xlsx", ".xls"],
                                          file_count='single')
            select_sheet = gr.Dropdown(label="选择sheet", value='请先选择要处理的文件')
        btn = gr.Button("开始处理")
        upload_file.upload(fn=get_sheet_list, inputs=upload_file, outputs=[select_sheet, status_bar])
        btn.click(fn=handle_excel, inputs=[upload_file, select_sheet], outputs=status_bar)
    main_interface.launch()
