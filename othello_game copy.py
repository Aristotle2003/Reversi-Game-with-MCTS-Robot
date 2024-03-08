#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
This module contains the main Othello game which maintains the board, score, and
players.  

Credit: Dr. Daniel Bauer
"""
import sys
import subprocess
import numpy as np
from threading import Timer
from othello_shared import find_lines, get_possible_moves, play_move, get_score

class InvalidMoveError(RuntimeError):
    pass


class AiTimeoutError(RuntimeError):
    pass


class Player(object):
    def __init__(self, color, name="Human"):
        self.name = name
        self.color = color

    def get_move(self, manager):
        pass  


class AiPlayerInterface(Player):

    TIMEOUT = 10

    def __init__(self, filename, color):
        self.color = color
        self.process = subprocess.Popen([sys.executable, '-u', filename], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.process.stdin.flush()
        name = self.process.stdout.readline().decode("ASCII").strip()
        print("AI introduced itself as: {}".format(name))
        self.name = name
        self.process.stdin.write((str(color)+"\n").encode("ASCII"))
        self.process.stdin.flush()

    def timeout(self): 
        sys.stderr.write("{} timed out.".format(self.name))
        self.process.kill() 
        self.timed_out = True

    def get_move(self, manager):
        white_score, dark_score = get_score(manager.board)
        self.process.stdin.write("SCORE {} {}\n".format(white_score, dark_score).encode("ASCII"))
        self.process.stdin.flush()
        self.process.stdin.write("{}\n".format(str(tuple(map(tuple, manager.board)))).encode("ASCII"))
        self.process.stdin.flush()

        timer = Timer(AiPlayerInterface.TIMEOUT, lambda: self.timeout())
        self.timed_out = False
        timer.start()

        # Wait for the AI call
        move_s = self.process.stdout.readline().decode("ASCII")

        if self.timed_out:  
            raise AiTimeoutError
        timer.cancel()
        i_s, j_s = move_s.strip().split()
        i = int(i_s)
        j = int(j_s)
        return i, j
    
    def kill(self, manager):
        white_score, dark_score = get_score(manager.board)
        self.process.stdin.write("FINAL {} {}\n".format(white_score, dark_score).encode("ASCII"))
        self.process.kill() 


class OthelloGameManager(object):

    def __init__(self, dimension=4):

        self.dimension = dimension
        self.board = self.create_initial_board()
        self.current_player = 1
            
    def create_initial_board(self):
        board = np.zeros((self.dimension, self.dimension))
        i = self.dimension // 2 - 1
        board[i, i] = 2
        board[i+1, i+1] = 2
        board[i+1, i] = 1
        board[i, i+1] = 1
        return board

    def play(self, i, j):
        if self.board[j, i] != 0:
           raise InvalidMoveError("Occupied square.")
        lines = find_lines(self.board, i, j, self.current_player)
        if not lines:  
           raise InvalidMoveError("Invalid Move.")
     
        self.board = play_move(self.board, self.current_player, i, j) 
        self.current_player = 1 if self.current_player == 2 else 2

    def get_possible_moves(self):
        return get_possible_moves(self.board, self.current_player)


def play_game(game, player1, player2):

    players = [None, player1, player2]

    while True: 
        player_obj = players[game.current_player]
        possible_moves = game.get_possible_moves() 
        if not possible_moves: 
            p1score, p2score = get_score(game.board)
            print("FINAL: {} (dark) {}:{} {} (light)".format(player1.name, p1score, p2score, player2.name))
            player1.kill(game)
            player2.kill(game)
            break 
        else: 
            color = "dark" if game.current_player == 1 else "light"
            try: 
                i, j = player_obj.get_move(game)
                print("{} ({}) plays {},{}".format(player_obj.name, color, i, j))
                game.play(i,j)
            except AiTimeoutError:
                print("{} ({}) timed out!".format(player_obj.name, color))
                print("FINAL: {} (dark) {}:{} {} (light)".format(player_obj.name, p1score, p2score, player2.name))
                player1.kill(game)
                player2.kill(game)
                break
