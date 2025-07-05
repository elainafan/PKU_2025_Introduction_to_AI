import numpy as np
from Maze import Maze


action = {
    0: np.array([-1, 0]),
    1: np.array([1, 0]),
    2: np.array([0, -1]),
    3: np.array([0, 1])
}


class Pacman:
    def __init__(self, pos: np.ndarray):
        """
        Initialize the Pacman object.
        :param pos: The initial position of the Pacman.
        """
        self.pos = pos
        self.score = 0
        self.id = 4
        self.available = None  # The available directions for the Pacman to move to.
        
    def move(self, direction, maze: Maze):
        """
        Move the Pacman to the given direction.
        :param direction: The direction to move to.
        :param maze: The maze object.
        """
        self.pos += action[direction]
        if maze.grid[self.pos[0], self.pos[1]] == 2:  # Pacman eats a food.
            self.score += 1
            maze.grid[self.pos[0], self.pos[1]] = 0
        maze.pacman_pos[self.id] = self.pos

    def get_available_directions(self, maze: Maze):
        """
        Get the available directions for the Pacman to move to.
        :param maze: The maze object.
        """
        x, y = self.pos
        directions = []
        if x > 0 and maze.grid[x-1, y] != 1:
            directions.append(0)
        if x < maze.grid.shape[0]-1 and maze.grid[x+1, y] != 1:
            directions.append(1)
        if y > 0 and maze.grid[x, y-1] != 1:
            directions.append(2)
        if y < maze.grid.shape[1]-1 and maze.grid[x, y+1] != 1:
            directions.append(3)
        self.available = directions



class Ghost(Pacman):
    def __init__(self, pos: np.ndarray):
        """
        Initialize the Ghost object.
        :param pos: The initial position of the Ghost.
        """
        super().__init__(pos)
        self.id = 3

    def move(self, maze: Maze):
        """
        Move the ghost randomly.
        :param maze: The maze object.
        """
        self.get_available_directions(maze)
        direction = np.random.choice(self.available)
        self.pos += action[direction]
        maze.pacman_pos[self.id] = self.pos
