# game_core.py
import random
from typing import Any, List, Optional, Tuple

import torch

from ml_model import SnakeNet
from game_constants import (
    ROWS, COLS, CELL_WIDTH, CELL_HEIGHT, LINE_WIDTH,
    INNER_CELL_WIDTH, INNER_CELL_HEIGHT, OUTER_RECT_X, OUTER_RECT_Y,
    SNAKE_COLOR, APPLE_COLOR,
)

class Snake:
    """Simple snake with position, body and movement logic."""

    def __init__(self) -> None:
        self.x: int = COLS // 2
        self.y: int = ROWS // 2
        self.body: List[Tuple[int, int]] = [(self.x, self.y)]
        self.direction: str = "RIGHT"  # Initial direction
        self.next_direction: str = self.direction
        self.grow: bool = False
        self.steps: int = 0

    def move(self) -> None:
        """Advance the snake one step in its next_direction."""
        self.direction = self.next_direction
        self.steps += 1

        if self.direction == "RIGHT": self.x += 1
        elif self.direction == "LEFT": self.x -= 1
        elif self.direction == "UP": self.y -= 1
        elif self.direction == "DOWN": self.y += 1

        new_head = (self.x, self.y)
        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def check_collision(self) -> bool:
        """Return True if the snake collides with a wall or itself."""
        if not (0 <= self.x < COLS and 0 <= self.y < ROWS):
            return True
        if (self.x, self.y) in self.body[1:]:
            return True
        return False

    def draw(self, screen: Any, pygame_instance: Any) -> None:
        """Render the snake on the given pygame surface."""
        for segment in self.body:
            pygame_instance.draw.rect(
                screen, SNAKE_COLOR,
                (OUTER_RECT_X + segment[0] * CELL_WIDTH + LINE_WIDTH,
                 OUTER_RECT_Y + segment[1] * CELL_HEIGHT + LINE_WIDTH,
                 INNER_CELL_WIDTH, INNER_CELL_HEIGHT)
            )

class Apple:
    """Randomly positioned apple eaten by the snake."""

    def __init__(self) -> None:
        self.x: int = random.randint(0, COLS - 1)
        self.y: int = random.randint(0, ROWS - 1)
        self.coords: Tuple[int, int] = (self.x, self.y)

    def respawn(self, snake_body: List[Tuple[int, int]]) -> None:
        """Place the apple at a random free location."""
        while True:
            self.x = random.randint(0, COLS - 1)
            self.y = random.randint(0, ROWS - 1)
            self.coords = (self.x, self.y)
            if self.coords not in snake_body:
                break

    def draw(self, screen: Any, pygame_instance: Any) -> None:
        """Render the apple on the given pygame surface."""
        pygame_instance.draw.rect(
            screen, APPLE_COLOR,
            (OUTER_RECT_X + self.x * CELL_WIDTH + LINE_WIDTH,
             OUTER_RECT_Y + self.y * CELL_HEIGHT + LINE_WIDTH,
             INNER_CELL_WIDTH, INNER_CELL_HEIGHT)
        )

class Agent:
    """Wrapper around ``SnakeNet`` that provides game specific helpers."""

    def __init__(self, model: Optional[SnakeNet] = None) -> None:
        self.model: SnakeNet = model if model is not None else SnakeNet()
        self.model.eval()

    def get_weights(self) -> torch.Tensor:
        """Return the model parameters as a 1-D tensor."""
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def set_weights(self, flat_weights: torch.Tensor) -> None:
        """Load flattened weights into the model."""
        pointer = 0
        for param in self.model.parameters():
            num_params = param.numel()
            param.data.copy_(flat_weights[pointer:pointer + num_params].view_as(param))
            pointer += num_params

    def get_inputs(self, snake: Snake, apple: Apple, grid_dims: Tuple[int, int]) -> List[int]:
        """Return the model input vector describing the game state."""
        head_x, head_y = snake.x, snake.y
        apple_x, apple_y = apple.x, apple.y
        body = snake.body
        
        inputs = []
        # Apple relative position (4 inputs)
        inputs.append(1 if apple_x < head_x else 0)  # Apple left
        inputs.append(1 if apple_x > head_x else 0)  # Apple right
        inputs.append(1 if apple_y < head_y else 0)  # Apple up
        inputs.append(1 if apple_y > head_y else 0)  # Apple down

        # Danger detection (wall or body) helper
        def is_cell_dangerous(pos_x, pos_y):
            if not (0 <= pos_x < COLS and 0 <= pos_y < ROWS): return 1 # Wall
            if (pos_x, pos_y) in body[1:]: return 1 # Self
            return 0

        # Danger straight, left, right relative to current direction (3 inputs)
        # Danger straight
        dx, dy = 0, 0
        if snake.direction == "UP": dy = -1
        elif snake.direction == "DOWN": dy = 1
        elif snake.direction == "LEFT": dx = -1
        elif snake.direction == "RIGHT": dx = 1
        inputs.append(is_cell_dangerous(head_x + dx, head_y + dy))

        # Danger to the left (relative to snake's current direction)
        left_dir = self.turn_left(snake.direction)
        ldx, ldy = 0, 0
        if left_dir == "UP": ldy = -1
        elif left_dir == "DOWN": ldy = 1
        elif left_dir == "LEFT": ldx = -1
        elif left_dir == "RIGHT": ldx = 1
        inputs.append(is_cell_dangerous(head_x + ldx, head_y + ldy))

        # Danger to the right (relative to snake's current direction)
        right_dir = self.turn_right(snake.direction)
        rdx, rdy = 0, 0
        if right_dir == "UP": rdy = -1
        elif right_dir == "DOWN": rdy = 1
        elif right_dir == "LEFT": rdx = -1
        elif right_dir == "RIGHT": rdx = 1
        inputs.append(is_cell_dangerous(head_x + rdx, head_y + rdy))
        
        # Current direction one-hot encoded (4 inputs)
        current_dir_encoding = [0, 0, 0, 0]  # UP, DOWN, LEFT, RIGHT
        if snake.direction == "UP": current_dir_encoding[0] = 1
        elif snake.direction == "DOWN": current_dir_encoding[1] = 1
        elif snake.direction == "LEFT": current_dir_encoding[2] = 1
        elif snake.direction == "RIGHT": current_dir_encoding[3] = 1
        inputs.extend(current_dir_encoding)
        
        return inputs

    def decide_direction(self, inputs_list: List[int]) -> str:
        """Choose a movement direction based on network output."""
        x = torch.tensor(inputs_list, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
        action_idx = torch.argmax(logits).item()
        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"] # Model output maps to these
        return possible_actions[action_idx]

    def turn_left(self, current_direction: str) -> str:
        """Return the direction obtained by turning left from ``current_direction``."""
        return {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}[current_direction]

    def turn_right(self, current_direction: str) -> str:
        """Return the direction obtained by turning right from ``current_direction``."""
        return {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}[current_direction]