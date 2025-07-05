import numpy as np
import random


class Maze:
    def __init__(self, size: np.ndarray, p: float = 0.2):
        """
        Initialize the maze object.
        :param size: The size of the maze.
        :param p: The probability of a wall being removed.
        """
        self.height = size[0]
        self.width = size[1]
        self.rows = (self.height - 1) // 2
        self.cols = (self.width - 1) // 2

        # To simplify the maze, we don't use random generation.
        # self.p = p
        # self.grid = np.ones((self.height, self.width), dtype=int)  # 0 for empty, 1 for wall, 2 for food

        # Don't change the grid!
        self.grid = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 1, 0, 0, 1, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 1, 0, 0, 1, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1]])
        self.pacman_pos = {}
        self.foods = []

        # self.generate_maze()

    '''
    def generate_maze(self):
        """
        Generate a maze using the randomized Kruskal's algorithm, then delete some walls randomly.
        """
        walls = [[{'right': True, 'down': True} for _ in range(self.cols)] for _ in range(self.rows)]
        edges = []

        class DisjointSet:
            def __init__(self, size):
                self.parent = list(range(size))
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)
                if root_x != root_y:
                    self.parent[root_y] = root_x
        
        for x in range(self.rows):
            for y in range(self.cols - 1):
                edges.append(('right', x, y))
        for x in range(self.rows - 1):
            for y in range(self.cols):
                edges.append(('down', x, y))
        
        random.shuffle(edges)
        ds = DisjointSet(self.rows * self.cols)
        
        for direction, x, y in edges:
            cell1 = x * self.cols + y
            if direction == 'right':
                cell2 = x * self.cols + (y + 1)
            else:
                cell2 = (x + 1) * self.cols + y
            
            if ds.find(cell1) != ds.find(cell2):
                ds.union(cell1, cell2)
                walls[x][y][direction] = False
        
        for x in range(self.rows):
            for y in range(self.cols):
                self.grid[2*x+1][2*y+1] = 0
                if not walls[x][y]['right']:
                    self.grid[2*x+1][2*y+2] = 0
                if not walls[x][y]['down']:
                    self.grid[2*x+2][2*y+1] = 0
        
        for x in range(1, self.height-1):
            for y in range(1, self.width-1):
                if self.grid[x, y] == 1 and random.random() < self.p:
                    self.grid[x, y] = 0
    '''

    def get_empty_cells(self):
        """
        Get the empty cells in the maze.
        :return: A list of empty cells.
        """
        return [(x, y) for x in range(1, self.height-1) for y in range(1, self.width-1) if self.grid[x, y] == 0]

    def add_food(self, num: int):
        """
        Add food to the maze.
        :param num: The number of food to add.
        """
        empty_cells = self.get_empty_cells()
        self.foods = random.sample(empty_cells, num)
        for x, y in self.foods:
            self.grid[x, y] = 2
    
    def add_pacman(self, id: int) -> np.ndarray:
        """
        Add the Pacman or Ghost to the maze.
        :param id: The id of the Pacman.
        :return: The position of the Pacman.
        """
        empty_cells = self.get_empty_cells()
        self.pacman_pos[id] = np.array(random.choice(empty_cells))
        return self.pacman_pos[id]
        