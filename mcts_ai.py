#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = children
        self.value = v
        self.N = N

    def get_child(self, state):
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def select(node, alpha):
    while not get_possible_moves(node.state, node.player) == []:
        possible_moves = get_possible_moves(node.state, node.player)
        if len(possible_moves) != len(node.children):
            # Node has unexplored children
            return node
        # Select child with highest UCT score
        uct_values = [child.value / child.N + alpha * np.sqrt(np.log(node.N) / child.N) for child in node.children]
        node = node.children[np.argmax(uct_values)]
    return node

def expand(node):
    untried_moves = [move for move in get_possible_moves(node.state, node.player) if all(not np.array_equal(play_move(node.state, node.player, move[0], move[1]), child.state) for child in node.children)]
    if untried_moves:
        move = random.choice(untried_moves)
        new_state = play_move(node.state, node.player, move[0], move[1])
        new_node = Node(new_state, 3 - node.player, node, [], 0, 0)  # Switch players
        node.children.append(new_node)
        return new_node
    return node  # If no moves are left, return the node itself

def simulate(node):
    current_state, current_player = node.state, node.player
    while True:
        moves = get_possible_moves(current_state, current_player)
        if not moves:
            break
        move = random.choice(moves)
        current_state = play_move(current_state, current_player, move[0], move[1])
        current_player = 3 - current_player
    return compute_utility(current_state)

def backprop(node, utility):
    while node is not None:
        node.N += 1  # Increment the number of visits to the node

        # Adjust utility based on the player
        # For the light player (2), use the utility directly
        # For the dark player (1), use the negative utility
        player_utility = utility if node.player == 2 else -utility

        # Update the node's value with the new average
        node.value = ((node.value * (node.N - 1)) + player_utility) / node.N

        # Move to the parent node to continue the backpropagation
        node = node.parent





def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for i in range(rollouts):
        leaf = select(root, alpha)
        new = expand(leaf)
        utility = simulate(new)
        backprop(new, utility)

    move = None
    plays = 0
    for m in get_possible_moves(state, player):
        s = play_move(state, player, m[0], m[1])
        if root.get_child(s).N > plays:
            plays = root.get_child(s).N
            move = m

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()