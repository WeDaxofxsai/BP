import numpy as np
import tkinter as tk
from tkinter import messagebox


class GridApp:
    def __init__(self, root):
        self.root = root
        self.grid_rows = 9
        self.grid_cols = 7
        self.grid_data = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        self.buttons = []

        self.create_grid()
        self.create_output_button()

    def create_grid(self):
        for r in range(self.grid_rows):
            row_buttons = []
            for c in range(self.grid_cols):
                button = tk.Button(
                    self.root,
                    width=4,
                    height=2,
                    bg="white",
                    command=lambda x=r, y=c: self.toggle_cell(x, y),
                )
                button.grid(row=r, column=c, padx=2, pady=2)
                row_buttons.append(button)
            self.buttons.append(row_buttons)

    def toggle_cell(self, row, col):
        # Toggle the cell state
        if self.grid_data[row, col] == 0:
            self.grid_data[row, col] = 1
            self.buttons[row][col].config(bg="black")
        else:
            self.grid_data[row, col] = 0
            self.buttons[row][col].config(bg="white")

    def create_output_button(self):
        output_button = tk.Button(
            self.root, text="Output Array", command=self.output_array
        )
        output_button.grid(
            row=self.grid_rows, column=0, columnspan=self.grid_cols, pady=10
        )

    def output_array(self):
        # Show the array as a message box
        output = str(self.grid_data.tolist()) + ","
        # messagebox.showinfo("Grid Data", output)
        with open("aaaa.txt", "a") as file:
            file.write(output + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("7x9 Grid App")
    app = GridApp(root)
    root.mainloop()
