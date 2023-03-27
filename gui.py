from threading import Thread
import tkinter as tk
from typing import Callable
from prob_reversi import DiscColor, Position

class GameGUI:
    MARGIN_RATIO = 0.05 
    DEFAULT_FONT_NAME = "TkDefaultFont"
    COORD_FONT_SIZE = 24
    COORD_TEXT_COLOR = "white"
    LINE_WIDTH = 5
    DISC_SIZE_RATIO = 0.85
    PROB_FONT_SIZE = 24
    PROB_TEXT_COLOR = "white"

    def __init__(self, worker: Callable[[None], None], window_size = 512) -> None:
        self.__game_worker = worker
        self.__pos = Position(4)

        self.__root = tk.Tk()
        self.__root.geometry(f"{window_size}x{window_size}")

        self.__canvas = tk.Canvas(self.__root, width=window_size, height=window_size, background="green")
        self.__canvas.pack(expand=True)

    def set_position(self, pos: Position):
        self.__pos = pos

    def start(self):
        thread = Thread(target=self.__game_worker)
        thread.start()
        self.__mainloop()
        thread.join()

    def __mainloop(self):
        root = self.__root
        while True:
            try:
                self.__draw_board()
                root.update_idletasks()
                root.update()
            except:
                break

    def __draw_board(self):
        canvas = self.__canvas
        canvas.delete("all")
        canvas.create_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), fill="green")
        
        pos = self.__pos
        self.__draw_coordinate(pos.SIZE)
        self.__draw_grid(pos.SIZE)
        self.__draw_discs_and_probs(pos)
        canvas.update()

    def __draw_coordinate(self, board_size):
        canvas = self.__canvas
        margin = canvas.winfo_width() * GameGUI.MARGIN_RATIO
        size = canvas.winfo_width() - margin * 2.0
        grid_size = size / board_size
        half_grid_size = grid_size * 0.5

        c0, c1 = margin + half_grid_size, margin * 0.5
        font = (GameGUI.DEFAULT_FONT_NAME, GameGUI.COORD_FONT_SIZE)
        text_color = GameGUI.COORD_TEXT_COLOR
        for i in range(board_size):
            canvas.create_text(c0, c1, text=chr(ord('A') + i), fill=text_color, font=font)
            canvas.create_text(c1, c0, text=str(i + 1), fill=text_color, font=font)
            c0 += grid_size

    def __draw_grid(self, board_size):
        canvas = self.__canvas
        margin = canvas.winfo_width() * GameGUI.MARGIN_RATIO
        size = canvas.winfo_width() - margin * 2.0
        grid_size = size / board_size

        line_width = GameGUI.LINE_WIDTH
        for i in range(board_size + 1):
            c = margin + i * grid_size
            canvas.create_line(c, margin, c, margin + size, fill="black", width=line_width)
            canvas.create_line(margin, c, margin + size, c, fill="black", width=line_width)

    def __draw_discs_and_probs(self, pos: Position):
        canvas = self.__canvas
        margin = canvas.winfo_width() * GameGUI.MARGIN_RATIO
        size = canvas.winfo_width() - margin * 2.0
        grid_size = size / pos.SIZE
        half_grid_size = grid_size * 0.5
        disc_size = grid_size * GameGUI.DISC_SIZE_RATIO
        disc_margin = (grid_size - disc_size) * 0.5

        font = (GameGUI.DEFAULT_FONT_NAME, GameGUI.PROB_FONT_SIZE)
        text_color = GameGUI.PROB_TEXT_COLOR
        for i in range(pos.SIZE):
            for j in range(pos.SIZE):
                disc = pos.get_square_color_at(pos.convert_coord2D_to_coord1D(i, j))

                if disc != DiscColor.NULL:
                    x = margin + i * grid_size + disc_margin
                    y = margin + j * grid_size + disc_margin
                    color = "black" if disc == DiscColor.BLACK else "white" 
                    canvas.create_oval(x, y, x + disc_size, y + disc_size, fill=color, outline=color)
                else:
                    x = margin + i * grid_size + half_grid_size
                    y = margin + j * grid_size + half_grid_size
                    prob = pos.TRANS_PROB[pos.convert_coord2D_to_coord1D(i, j)]
                    canvas.create_text(x, y, text=f"{prob:.1f}", fill=text_color, font=font)



        

        