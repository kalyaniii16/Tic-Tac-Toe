import json
import random

EMPTY = " "
PLAYER = "X"
AI = "O"

WIN_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

def check_winner(board, player):
    return any(all(board[i] == player for i in combo) for combo in WIN_COMBOS)

def is_draw(board):
    return EMPTY not in board and not check_winner(board, PLAYER) and not check_winner(board, AI)

def get_available_moves(board):
    return [i for i in range(9) if board[i] == EMPTY]

def make_move(board, move, player):
    return board[:move] + player + board[move+1:]

def get_reward(board):
    if check_winner(board, AI):
        return 1
    elif check_winner(board, PLAYER):
        return -1
    elif is_draw(board):
        return 0.5
    return 0

EPISODES = 1000000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1  # Lower exploration, more focused

q_table = {}

def choose_action(state, available, explore=True):
    if explore and (random.random() < EPSILON or state not in q_table):
        return random.choice(available)
    q_values = q_table[state]
    return max(available, key=lambda a: q_values.get(a, 0.0))

for _ in range(EPISODES):
    board = EMPTY * 9
    history = []

    # Player (X) moves first randomly (no learning)
    while True:
        # Player's turn
        player_moves = get_available_moves(board)
        if not player_moves:
            break
        player_move = random.choice(player_moves)
        board = make_move(board, player_move, PLAYER)

        if check_winner(board, PLAYER) or is_draw(board):
            break

        # AI's turn (learns)
        state = board
        moves = get_available_moves(board)
        move = choose_action(state, moves, explore=True)
        board = make_move(board, move, AI)
        reward = get_reward(board)
        history.append((state, move))

        if reward != 0:
            break

    # Update Q-table
    for i in reversed(range(len(history))):
        state, move = history[i]
        if state not in q_table:
            q_table[state] = {}
        future_q = 0.0
        if i < len(history) - 1:
            next_state = history[i + 1][0]
            future_q = max(q_table.get(next_state, {}).values(), default=0.0)
        old_q = q_table[state].get(move, 0.0)
        q_table[state][move] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_q - old_q)

# Save the trained table
with open("q_table.json", "w") as f:
    json.dump(q_table, f, indent=2)

print("✅ Hard AI training complete. q_table.json saved.")
