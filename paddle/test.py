# import paddle
# print(paddle.__version__)
# paddle.utils.run_check()


import pandas as pd


df_data = pd.read_excel(r"C:\Users\mengf\Desktop\本科毕业设计\岩心分类方法.xlsx", sheet_name='图版法', header=0)

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
df_output.to_csv("test.csv", index=False)


