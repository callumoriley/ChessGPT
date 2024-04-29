import chess

from chessboard import display
from time import sleep

board = chess.Board()

move_list = [
    'e4', 'e5',
    'Qh5', 'Nc6',
    'Bc4', 'Nf6',
    'Qxf7'
]

game_board = display.start(board.fen())
while not display.check_for_quit():
    if move_list:
        board.push_san(move_list.pop(0))
        display.update(board.fen(), game_board)
    sleep(1)
display.terminate()