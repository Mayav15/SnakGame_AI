from SnakeGame import *
from FFNN import *

def run_game_with_ML(screen, clock, weights):
    in_game_score = 0
    max_score = 0
    penalty = 0
    steps_per_game = 2500
    score_all = 0
    test_games = 1

    if np.any(np.isnan(weights)):
        raise ValueError(f"Weights contain NaNs: {weights}")

    for _ in range(test_games):

        snake_head, snake_body, food_pos, score = start()
        count_same_dir = 0
        prev_dir = 0

        for _ in range(steps_per_game):
            current_dir, is_front_blocked, is_left_blocked, is_right_blocked = blocked(snake_body)
            angle, snake_dir, normalized_food_dir, normalized_snake_dir = snake_angle_with_food(snake_body, food_pos)
            
            network_input = np.array([is_left_blocked, is_front_blocked, is_right_blocked, normalized_food_dir[0],
                                    normalized_snake_dir[0], normalized_food_dir[1], normalized_snake_dir[1]]).reshape(-1, 7)
            
            output = forward_prop(network_input, weights)
            
            predicted_dir = np.argmax(np.array(output)) - 1

            if predicted_dir == prev_dir:
                count_same_dir += 1
            else:
                count_same_dir = 0
                prev_dir = predicted_dir

            new_dir = np.array(snake_body[0]) - np.array(snake_body[1])

            if predicted_dir == -1:
                new_dir = np.array([new_dir[1], -new_dir[0]])
            if predicted_dir == 1:
                new_dir = np.array([-new_dir[1], new_dir[0]])

            button_direction = direction_button_gen(new_dir)

            next_step = snake_body[0] + current_dir
            
            if collision_with_bound(snake_body[0]) == 1 or collision_with_self(next_step.tolist(), snake_body) == 1:
                penalty += 150
                break
            else:
                penalty += 0

            snake_body, food_pos, score = play(snake_head, snake_body, food_pos, button_direction, score, screen, clock)

            if score > max_score:
                max_score = score

            if count_same_dir > 5 and predicted_dir != 0:
                score_all -= 1
            else:
                score_all += 2
            
            in_game_score = score
        
        penalty += int(snake_food_dist(snake_body,food_pos))
    
    score_all += in_game_score

    return score_all - penalty + max_score * 5000, in_game_score