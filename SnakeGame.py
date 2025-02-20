import random
import math
import numpy as np
import pygame

# DIRECTION BUTTONS:
# 1 = LEFT
# 2 = UP
# 3 = RIGHT
# 0 = DOWN

# INITIALIZING PYGAME
pygame.init()
pygame.font.init()

# CONSTANTS
WIDTH = 500
HEIGHT = 500

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

FPS = 60000

# FONT FOR TEXT ON SCREEN
score_font = pygame.font.Font(None,25)

# SETTING SCREEN
screen=pygame.display.set_mode((WIDTH,HEIGHT))

# SETTING CLOCK
clock=pygame.time.Clock()

# DEFAULT VALUES FOR START OF GAME
def start():
    snake_head = [50, 50]
    snake_body = [[50, 50], [40, 50], [30, 50]]
    food_pos = [random.randint(1, 50) * 10, random.randint(1, 50) * 10]
    score = 0
    return snake_head, snake_body, food_pos, score

def draw_snake(snake_body, screen):
    for part in snake_body:
        pygame.draw.rect(screen, GREEN, pygame.Rect(part[0], part[1], 10, 10))

def draw_food(food_pos, screen):
    pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

def collision_with_food(food_pos, score):
    food_pos = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return food_pos, score

def collision_with_bound(snake_head):
    if snake_head[0] >= WIDTH or snake_head[0] < 0 or snake_head[1] >= HEIGHT or snake_head[1] < 0:
        return 1
    else:
        return 0

def collision_with_self(snake_head, snake_body):
    if snake_head in snake_body[1:]:
        return 1
    else:
        return 0

def grow_and_move_snake(snake_head, snake_body, food_pos, button_direction, score):
    if button_direction == 3:
        snake_head[0] += 10
    elif button_direction == 1:
        snake_head[0] -= 10
    elif button_direction == 0:
        snake_head[1] += 10
    else:
        snake_head[1] -= 10

    if snake_head == food_pos:
        food_pos, score = collision_with_food(food_pos, score)
        snake_body.insert(0, list(snake_head))
    else:
        snake_body.insert(0, list(snake_head))
        snake_body.pop()

    return snake_body, food_pos, score

def snake_food_dist(snake_body,food_pos):
    return np.linalg.norm(np.array(food_pos) - np.array(snake_body[0]))

def snake_angle_with_food(snake_body, food_pos):
    food_dir = np.array(food_pos) - np.array(snake_body[0])
    snake_dir = np.array(snake_body[0]) - np.array(snake_body[1])

    food_direct_norm = np.linalg.norm(food_dir)
    snake_direct_norm = np.linalg.norm(snake_dir)

    if snake_direct_norm == 0:
        snake_direct_norm = 10
    if food_direct_norm == 0:
        food_direct_norm = 10

    normalized_food_dir = food_dir / food_direct_norm
    normalized_snake_dir = snake_dir / snake_direct_norm
    
    dot_product = np.dot(normalized_snake_dir,normalized_food_dir)
    angle_magnitude = np.arccos(np.clip(dot_product,-1,1))
    
    cross_prod = snake_dir[0]*food_dir[1] - snake_dir[1]*food_dir[0]
    sign = np.sign(cross_prod)

    signed_angle = sign*angle_magnitude / math.pi

    # signed_angle = math.atan2(normalized_food_dir[1] * normalized_snake_dir[0] - normalized_food_dir[0] * normalized_snake_dir[1],
    #     normalized_food_dir[1] * normalized_snake_dir[1] + normalized_food_dir[0] * normalized_snake_dir[0]) / math.pi

    return signed_angle, snake_dir, normalized_food_dir, normalized_snake_dir

def direction_button_gen(new_dir):
    button_direction = 0
    if new_dir.tolist() == [10, 0]:
        button_direction = 3
    elif new_dir.tolist() == [-10, 0]:
        button_direction = 1
    elif new_dir.tolist() == [0, 10]:
        button_direction = 0
    else:
        button_direction = 2

    return button_direction

def generate_direction(snake_body, direction):
    current_dir = np.array(snake_body[0]) - np.array(snake_body[1])
    left_dir = np.array([current_dir[1], -current_dir[0]])
    right_dir = np.array([-current_dir[1], current_dir[0]])

    new_dir = current_dir

    if direction == -1:
        new_dir = left_dir
    if direction == 1:
        new_dir = right_dir

    button_direction = direction_button_gen(new_dir)

    return direction, button_direction

def is_direction_blocked(snake_body, current_dir):
    next_step = snake_body[0] + current_dir
    if collision_with_bound(next_step) == 1 or collision_with_self(next_step.tolist(), snake_body) == 1:
        return 1
    else:
        return 0

def blocked(snake_body):
    current_dir = np.array(snake_body[0]) - np.array(snake_body[1])

    left_dir = np.array([current_dir[1], -current_dir[0]])
    right_dir = np.array([-current_dir[1], current_dir[0]])

    is_front_blocked = is_direction_blocked(snake_body, current_dir)
    is_left_blocked = is_direction_blocked(snake_body, left_dir)
    is_right_blocked = is_direction_blocked(snake_body, right_dir)

    return current_dir, is_front_blocked, is_left_blocked, is_right_blocked

def play(snake_head, snake_body, food_pos, button_direction, score, screen, clock):
    crashed = False
    while not crashed:
        try:
            events = pygame.event.get()
        except Exception as e:
            print(f"Error fetching events: {e}")
            events = []
        for event in events:
            try:
                if event.type == pygame.QUIT:
                    crashed = True
                    pygame.quit()
            except Exception as e:
                print(f"Error handling event: {e}")

        screen.fill(BLACK)

        draw_food(food_pos, screen)
        draw_snake(snake_body, screen)

        snake_body, food_pos, score = grow_and_move_snake(snake_head, snake_body, food_pos, button_direction, score)

        pygame.display.set_caption("Snake Game with Genetic Algorithm")

        score_surface = score_font.render("Score:"+str(score),True,WHITE)
        screen.blit(score_surface,(400,20))

        pygame.display.update()
        clock.tick(FPS)

        return snake_body, food_pos, score