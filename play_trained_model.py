# play_trained_model.py
import pygame
import torch
from typing import Any, Tuple

from game_core import Snake, Apple, Agent
from ml_model import SnakeNet  # Required to instantiate model before loading state_dict
from game_constants import (
    SCREEN_BASE_WIDTH,
    SCREEN_BASE_HEIGHT,
    MOVE_INTERVAL_PLAY,
    LINE_WIDTH,
    OUTER_RECT_X,
    OUTER_RECT_Y,
    OUTER_RECT_WIDTH,
    OUTER_RECT_HEIGHT,
    CELL_WIDTH,
    CELL_HEIGHT,
    COLS,
    ROWS,
    WHITE,
    BLACK,
    SNAKE_COLOR,
    APPLE_COLOR,
)

MODEL_PATH = "best_snake_model.pt"

def initialize_game_elements_for_play() -> Tuple[Snake, Apple, Agent, bool]:
    """Load the trained model and create game objects."""
    game_snake = Snake()
    game_apple = Apple()
    game_apple.respawn(game_snake.body)

    try:
        trained_model = SnakeNet() # Ensure input_size matches agent's get_inputs
        trained_model.load_state_dict(torch.load(MODEL_PATH))
        trained_model.eval()
        ai_agent = Agent(model=trained_model)
        print(f"🐍 Model '{MODEL_PATH}' loaded successfully for gameplay.")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file '{MODEL_PATH}' not found. Please train the model first by running train.py.")
        return None, None, None, True # game_snake, game_apple, ai_agent, error_occurred
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, True
    
    return game_snake, game_apple, ai_agent, False

def draw_game_grid(screen: Any, pygame_instance: Any) -> None:
    """Draw the outer rectangle and grid lines."""
    pygame_instance.draw.rect(screen, WHITE, (OUTER_RECT_X, OUTER_RECT_Y, OUTER_RECT_WIDTH, OUTER_RECT_HEIGHT), LINE_WIDTH)
    for col_idx in range(1, COLS):
        x_pos = OUTER_RECT_X + col_idx * CELL_WIDTH
        pygame_instance.draw.line(screen, BLACK, (x_pos, OUTER_RECT_Y + 1), (x_pos, OUTER_RECT_Y + OUTER_RECT_HEIGHT - 2))
    for row_idx in range(1, ROWS):
        y_pos = OUTER_RECT_Y + row_idx * CELL_HEIGHT
        pygame_instance.draw.line(screen, BLACK, (OUTER_RECT_X + 1, y_pos), (OUTER_RECT_X + OUTER_RECT_WIDTH - 2, y_pos))

def display_score(screen: Any, pygame_instance: Any, score: int, font: Any) -> None:
    """Render the current score in the top left corner."""
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (OUTER_RECT_X, OUTER_RECT_Y - 40))

def display_centered_message(
    screen: Any,
    pygame_instance: Any,
    text: str,
    font: Any,
    y_offset: int = 0,
    color: Tuple[int, int, int] = WHITE,
) -> None:
    """Show a message centered on the screen."""
    message_surface = font.render(text, True, color)
    message_rect = message_surface.get_rect(center=(SCREEN_BASE_WIDTH // 2, SCREEN_BASE_HEIGHT // 2 + y_offset))
    screen.blit(message_surface, message_rect)

def main_play_loop() -> None:
    """Run a Pygame loop letting the AI play the game."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_BASE_WIDTH, SCREEN_BASE_HEIGHT))
    pygame.display.set_caption("Snake Game - AI Playing")
    clock = pygame.time.Clock()
    font_main = pygame.font.SysFont(None, 36)
    font_large = pygame.font.SysFont(None, 48)

    game_snake, game_apple, ai_agent, error_loading = initialize_game_elements_for_play()
    if error_loading:
        # Display error message on screen if Pygame initialized
        running_error_display = True
        while running_error_display:
            for event_error in pygame.event.get():
                if event_error.type == pygame.QUIT or (event_error.type == pygame.KEYDOWN and event_error.key == pygame.K_q) :
                    running_error_display = False
            screen.fill(BLACK)
            display_centered_message(screen, pygame, f"Model '{MODEL_PATH}' not found.", font_main, y_offset=-20, color=APPLE_COLOR)
            display_centered_message(screen, pygame, "Please run train.py first.", font_main, y_offset=20, color=APPLE_COLOR)
            display_centered_message(screen, pygame, "Press Q to quit.", font_main, y_offset=60)
            pygame.display.flip()
            clock.tick(15)
        pygame.quit()
        return


    current_score = 0
    game_active_state = "start"  # "start", "playing", "game_over"
    last_ai_move_time = 0
    running = True

    while running:
        time_now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game_active_state == "start":
                    game_active_state = "playing"
                    last_ai_move_time = time_now # Reset timer for first move
                elif game_active_state == "game_over":
                    if event.key == pygame.K_r:
                        game_snake, game_apple, ai_agent, error_loading = initialize_game_elements_for_play() # Re-init
                        if error_loading: running=False; break # Exit if model vanished
                        current_score = 0
                        game_active_state = "start"
                    elif event.key == pygame.K_q:
                        running = False
        
        screen.fill(BLACK)

        if game_active_state == "start":
            display_centered_message(screen, pygame, "Press any key to Play with AI", font_large)
        
        elif game_active_state == "playing":
            if time_now - last_ai_move_time > MOVE_INTERVAL_PLAY:
                inputs = ai_agent.get_inputs(game_snake, game_apple, (COLS, ROWS))
                action = ai_agent.decide_direction(inputs)
                
                current_dir = game_snake.direction # Similar logic as in trainer to avoid instant reversal
                if not ((action == "UP" and current_dir == "DOWN") or \
                        (action == "DOWN" and current_dir == "UP") or \
                        (action == "LEFT" and current_dir == "RIGHT") or \
                        (action == "RIGHT" and current_dir == "LEFT")):
                    game_snake.next_direction = action
                
                game_snake.move()
                last_ai_move_time = time_now

                if (game_snake.x, game_snake.y) == game_apple.coords:
                    game_snake.grow = True
                    current_score += 1
                    game_apple.respawn(game_snake.body)
                
                if game_snake.check_collision():
                    game_active_state = "game_over"
            
            draw_game_grid(screen, pygame)
            game_snake.draw(screen, pygame)
            game_apple.draw(screen, pygame)
            display_score(screen, pygame, current_score, font_main)

        elif game_active_state == "game_over":
            display_centered_message(screen, pygame, "Game Over!", font_large, y_offset=-50)
            display_centered_message(screen, pygame, f"Final Score: {current_score}", font_main, y_offset=0)
            display_centered_message(screen, pygame, "Press 'R' to Restart, 'Q' to Quit", font_main, y_offset=50)

        pygame.display.flip()
        clock.tick(15) # Control game speed for visibility

    pygame.quit()

if __name__ == "__main__":
    main_play_loop()