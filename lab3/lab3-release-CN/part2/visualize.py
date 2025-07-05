import pygame
import numpy as np

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64
import hashlib


COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 255, 255),
    3: (255, 0, 255),
    4: (255, 255, 0),
    "bg": (0, 0, 0)
}


def pygame_visualization(maze, pacman_pos, ghost_pos, step, is_win, cell_size=30):
    """
    Use pygame to visualize the game.
    """
    pygame.init()

    rows, cols = maze.shape
    width = cols * cell_size
    height = rows * cell_size + 40
    
    screen = pygame.display.set_mode((width, height))
    name = "Pacman Game"
    pygame.display.set_caption(name)
    
    font = pygame.font.Font(None, 36)
    
    screen.fill(COLORS["bg"])
    
    for row in range(rows):
        for col in range(cols):
            x = col * cell_size
            y = row * cell_size
            
            if maze[row, col] == 1:
                pygame.draw.rect(screen, COLORS[1], (x, y, cell_size, cell_size))
            
            elif maze[row, col] == 2:
                center = (x + cell_size//2, y + cell_size//2)
                pygame.draw.circle(screen, COLORS[2], center, cell_size//8)
    
    px, py = pacman_pos
    pacman_x = py * cell_size + cell_size//2
    pacman_y = px * cell_size + cell_size//2
    pygame.draw.circle(screen, COLORS[4], (pacman_x, pacman_y), cell_size//2 - 2)

    gx, gy = ghost_pos
    ghost_x = gy * cell_size + cell_size//2
    ghost_y = gx * cell_size + cell_size//2
    pygame.draw.circle(screen, COLORS[3], (ghost_x, ghost_y), cell_size//2 - 2)
    
    step_text = font.render(f"Step: {step}", True, (255, 255, 255))
    screen.blit(step_text, (10, height - 35))
    
    pygame.display.flip()
    if is_win:
        key = hashlib.sha256(name.encode()).digest()
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(str(step).encode('utf-8'), AES.block_size))
        data = base64.b64encode(iv + ciphertext).decode('utf-8')
        with open("res_en.txt", "w", encoding="utf-8") as file:
            file.write(data)

    