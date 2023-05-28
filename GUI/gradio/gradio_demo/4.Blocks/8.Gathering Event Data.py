import gradio as gr

with gr.Blocks() as demo:
    turn = gr.Textbox("X", interactive=True, label="Turn")
    board = gr.Dataframe(value=[["", "", ""]] * 3, interactive=True, type="array")

    def place(board, turn, evt: gr.SelectData):
        if evt.value:
            return board, turn
        board[evt.index[0]][evt.index[1]] = turn
        turn = "O" if turn == "X" else "X"
        return board, turn

    board.select(place, [board, turn], [board, turn])

demo.launch()