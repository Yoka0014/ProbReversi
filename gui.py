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

    def __init__(self, worker: Callable[[None], None], window_size=512) -> None:
        self.__game_worker = worker
        self.__pos = Position(4)

        self.__root = tk.Tk()
        self.__root.geometry(f"{window_size}x{window_size}")
        self.__root.protocol("WM_DELETE_WINDOW", self.__on_closing)
        self.__destroy_flag = False

        self.__canvas = tk.Canvas(self.__root, width=window_size, height=window_size, background="green")
        self.__canvas.bind("<Button-1>", self.__on_canvas_click)
        self.__canvas.pack(expand=True)

        self.__redraw_flag = False

        # event handlers
        self.board_clicked: Callable[[int], None] = lambda: None
        self.window_closed: Callable[[], None] = lambda: None

    def set_position(self, pos: Position):
        self.__pos = pos
        self.__redraw_flag = True

    def start(self):
        thread = Thread(target=self.__game_worker)
        thread.start()
        self.__mainloop()
        thread.join()

    def __mainloop(self):
        root = self.__root
        while not self.__destroy_flag:
            try:
                if self.__redraw_flag:
                    self.__redraw_flag = False
                    self.__draw_board()
                root.update_idletasks()
                root.update()
            except:
                break

    def __on_closing(self):
        self.__destroy_flag = True
        self.__root.destroy()
        self.window_closed()

    def __on_canvas_click(self, event: tk.Event):
        margin = self.__canvas.winfo_width() * GameGUI.MARGIN_RATIO
        size = self.__canvas.winfo_width() - margin * 2.0
        grid_size = size / self.__pos.SIZE
        x, y = event.x - margin, event.y - margin
        coord = self.__pos.convert_coord2D_to_coord1D(int(x / grid_size), int(y / grid_size))
        self.board_clicked(coord)

    def __draw_board(self):
        canvas = self.__canvas
        canvas.delete("all")
        canvas.create_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), fill="green")

        pos = self.__pos
        self.__draw_coordinate(pos.SIZE)
        self.__draw_grid(pos.SIZE)
        self.__draw_discs_and_probs(pos)
        self.__draw_moves(pos)
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

    def __draw_moves(self, pos: Position):
        canvas = self.__canvas
        margin = canvas.winfo_width() * GameGUI.MARGIN_RATIO
        size = canvas.winfo_width() - margin * 2.0
        grid_size = size / pos.SIZE
        disc_size = grid_size * GameGUI.DISC_SIZE_RATIO
        disc_margin = (grid_size - disc_size) * 0.5

        color = "black" if self.__pos.side_to_move == DiscColor.BLACK else "white"
        width = GameGUI.LINE_WIDTH
        for move in pos.get_next_moves():
            i, j = pos.convert_coord1D_to_coord2D(move)
            x = margin + i * grid_size + disc_margin
            y = margin + j * grid_size + disc_margin
            canvas.create_oval(x, y, x + disc_size, y + disc_size, outline=color, width=width)
