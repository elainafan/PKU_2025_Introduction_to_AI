import random
import numpy as np
from openai import OpenAI

from Pacman import Pacman, Ghost
from Maze import Maze
import visualize
import LLM


def main():
    step_cnt = 0
    is_win = False
    while True:
        step_cnt += 1
        visualize.pygame_visualization(maze.grid, pacman.pos, ghost.pos, step_cnt, is_win)
        places.append(pacman.pos.copy())

        # Pacman move
        pacman.get_available_directions(maze)
        available = pacman.available

        if not available:
            print("No moves left! Game Over.")
            break
        elif len(available) == 1:
            pacman.move(available[0], maze)
        else:
            choice, response = LLM.get_ai_move(client, model_name, maze, file, places, available)  # Get AI move
            AI_responses.append(response)
            if choice not in available:
                print("AI move not available!")
                break
            pacman.move(choice, maze)

        if pacman.pos[0] == ghost.pos[0] and pacman.pos[1] == ghost.pos[1]:
            print("Pacman is eaten by the ghost! Game Over.")
            break

        # Ghost move
        # ghost.move(maze)  # We assume that the ghost don't move in this version, just for more easy to AI
        # if pacman.pos[0] == ghost.pos[0] and pacman.pos[1] == ghost.pos[1]:
        #     print("Pacman is eaten by the ghost! Game Over.")
        #     break

        # Check if all foods are eaten
        if 2 not in maze.grid:
            with open("res.txt", "w", encoding="utf-8") as res_file:
                res_file.write(str(step_cnt))
            is_win = True
            visualize.pygame_visualization(maze.grid, pacman.pos, ghost.pos, step_cnt, is_win)
            print("All foods eaten! You win!")
            break


if __name__ == "__main__":
    # Don't change the seed here
    np.random.seed(1)
    random.seed(1)


    # Alternatively, you can use other LLM models like GPT-4, etc.
    API_KEY = ""  # TODO: Replace with your own Deepseek API key
    model_name = ""  # TODO: Choose the model, "deepseek-chat" == deepseek-v3, "deepseek-reasoner" == deepseek-r1


    maze_size = np.array([8, 8])
    food_num = 20
    maze = Maze(maze_size, p=0.3)
    maze.add_food(food_num)

    pacman = Pacman(maze.add_pacman(4))
    ghost = Ghost(maze.add_pacman(3))
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    places = []
    AI_responses = []
    
    file = open("log.txt", "w+", encoding="utf-8")
    file.truncate(0)
    main()
