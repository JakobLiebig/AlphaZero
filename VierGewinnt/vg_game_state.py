import game_state
import numpy as np

WHITE = -1
BLACK = 1

class GameState(game_state.Base):
    def initial(height, width):
        grid = np.zeros([height, width])
        num_empty_tiles = height * width
        game_over = False
        reward = np.array([0], dtype=np.float32)
        active_player = WHITE
        
        return GameState(grid, num_empty_tiles, game_over, reward, active_player)
    
    def __init__(self,
                 grid,
                 num_empty_tiles,
                 game_over,
                 reward,
                 active_player):
        self.grid = grid
        self.num_empty_tiles = num_empty_tiles
        self.game_over = game_over
        self.reward = reward # reward for reaching this state!!
        self.active_player = active_player
       
    def __str__(self):
        string = ""
        
        for row in range(self.grid.shape[0]):
            for column in range(self.grid.shape[1]):
                cur_tile = self.grid[row, column]
                
                if cur_tile == WHITE:
                    string += "O"
                elif cur_tile == BLACK:
                    string += "X"
                else:
                    string += "."
            string += "\n"
        
        return string
        
    def step(self, place_column):
        grid, place_row = GameState.__place_piece(self.grid, place_column, self.active_player)
        
        num_empty_tiles = max(self.num_empty_tiles - 1, 0)
        
        terminal = (GameState.__contains_four(grid, place_row, place_column)
                       or num_empty_tiles <= 0)
        
        if terminal:
            reward = np.array([1.], dtype=np.float32)
        else:
            reward = np.array([0.], dtype=np.float32)
        
        active_player = GameState.__opponent_of(self.active_player)
        
        return GameState(grid, num_empty_tiles, terminal, reward, active_player)

    def is_terminal(self):
        return self.game_over

    def get_reward(self):
        return self.reward

    def generate_possible_actions(self):
        possible_actions = []
        
        for index, tile in enumerate(self.grid[0]):
            if tile == 0:
                possible_actions.append(index)
        
        return possible_actions
    
    def generate_state(self):
        white_pieces = np.equal(self.grid, WHITE).astype(np.float32)
        black_pieces = np.equal(self.grid, BLACK).astype(np.float32)
        
        if self.active_player == WHITE:
            state = np.stack([white_pieces, black_pieces])
        else:
            state = np.stack([black_pieces, white_pieces])
        
        return state
    
    def generate_mask(self):
        return np.equal(self.grid[0], 0.)
    
    def __place_piece(grid, place_column, colour):
        new_grid = np.copy(grid)
        
        for row in range(grid.shape[0] - 1, -1, -1):
            cur_piece = grid[row, place_column]
            
            if cur_piece == 0:
                new_grid[row, place_column] = colour
                
                break
        
        return new_grid, row
    
    def __contains_four(grid, place_row, place_column):
        active_player = grid[place_row, place_column]
        
        directions = {(1, 1) : 1,
                      (1, 0) : 1,
                      (0, 1) : 1,
                      (1,-1) : 1}
        queue = []
        
        for y_direction, x_direction in directions.keys():
            queue.append((place_row, place_column, y_direction, x_direction))
            
            queue.append((place_row, place_column, y_direction * -1, x_direction * -1))
        
        while len(queue) > 0:
            y, x, y_direction, x_direction = queue[0]
            
            next_y = y + y_direction
            next_x = x + x_direction
            
            if (GameState.__is_valid_position(grid, next_y, next_x)
                and grid[(next_y, next_x)] == active_player):
                queue.append((next_y, next_x, y_direction, x_direction))

                y_direction, x_direction = GameState.__normalize_direction((y_direction, x_direction))
                directions[(y_direction, x_direction)] += 1
            queue.pop(0)
        
            if 4 in directions.values():
                return True

        return False

    def __is_valid_position(grid, y, x):
        return 0 <= y < grid.shape[1] and 0 <= x < grid.shape[0]
    
    def __normalize_direction(dir):
        if dir == (-1, -1):
            return (1, 1)
        elif dir == (-1, 0):
            return (1, 0)
        elif dir == (0, -1):
            return (0, 1)
        elif dir == (-1, 1):
            return (1, -1) 
        
        return dir

    def __opponent_of(player):
        if player == WHITE:
            return BLACK
        elif player == BLACK:
            return WHITE