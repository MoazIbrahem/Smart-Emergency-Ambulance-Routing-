import numpy as np
from PIL import Image as PILImage, ImageDraw
from enum import Enum
import random
import gymnasium as gym


# Definitions
class LightColor(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2


class Cell(Enum):
    ROAD = 0
    HOUSE = 1
    TRAFFIC = 2
    HOSPITAL = 3
    CAR = 4


class GridAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


# Environment
class SmartAmbulanceEnv(gym.Env):
    """
    Custom Gymnasium Environment for a Smart Ambulance pathfinding task.

    Attributes:
        size (int): Grid dimensions (size x size).
        _grid (numpy.ndarray): 2D array representing the map layout.
        traffic_lights (dict): Stores (r, c) positions and current [LightColor, timer].
        car_positions (set): Current coordinates of dynamic car obstacles.
        car_move_rate (int): Frequency of car movements (every N steps).
    """

    def __init__(self, size=20, seed=42):
        self.size = size
        self.rows = size
        self.cols = size
        self._state = 0

        random.seed(seed)
        np.random.seed(seed)

        self._grid = np.zeros((size, size), dtype=int)
        self.car_move_counter = 0
        self.car_move_rate = 3  # Every how many steps cars move

        # Houses
        for _ in range(20):
            r, c = random.randint(0, size - 1), random.randint(0, size - 1)
            if (r, c) != (0, 0) and (r, c) != (size - 1, size - 1):
                self._grid[r, c] = Cell.HOUSE.value

        # Moving Cars
        for _ in range(10):
            r, c = random.randint(0, size - 1), random.randint(0, size - 1)
            if self._grid[r, c] == Cell.ROAD.value and (r, c) != (0, 0):
                self._grid[r, c] = Cell.CAR.value

        # Save Car Positions
        self.car_positions = set()
        for r in range(size):
            for c in range(size):
                if self._grid[r, c] == Cell.CAR.value:
                    self.car_positions.add((r, c))

        # Traffic Lights
        self.traffic_lights = {}
        for _ in range(15):
            r, c = random.randint(0, size - 1), random.randint(0, size - 1)
            if self._grid[r, c] == Cell.ROAD.value and (r, c) != (0, 0):
                self._grid[r, c] = Cell.TRAFFIC.value
                self.traffic_lights[(r, c)] = [random.choice(list(LightColor)), 0]

        self._grid[size - 1, size - 1] = Cell.HOSPITAL.value

        self.observation_space = gym.spaces.Discrete(size * size)
        self.action_space = gym.spaces.Discrete(len(GridAction))

    def update_lights(self):
        """Updates the state and timers of all traffic lights in the grid."""
        for loc in self.traffic_lights:
            color, counter = self.traffic_lights[loc]
            counter += 1
            limit = (
                10
                if color == LightColor.GREEN
                else 5 if color == LightColor.YELLOW else 10
            )
            if counter >= limit:
                next_c = (
                    LightColor.YELLOW
                    if color == LightColor.GREEN
                    else (
                        LightColor.RED
                        if color == LightColor.YELLOW
                        else LightColor.GREEN
                    )
                )
                self.traffic_lights[loc] = [next_c, 0]
            else:
                self.traffic_lights[loc] = [color, counter]

    def move_cars(self):
        """
        Handles dynamic obstacle movement.
        Cars move randomly to adjacent road cells while avoiding houses,
        hospitals, traffic lights, and other cars.
        """
        self.car_move_counter += 1

        if self.car_move_counter % self.car_move_rate != 0:
            return
        BLOCKED = {Cell.HOUSE.value, Cell.HOSPITAL.value, Cell.TRAFFIC.value}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        new_positions = set()
        amb_r, amb_c = divmod(self._state, self.cols)

        for r, c in list(self.car_positions):
            shuffled = directions[:]
            random.shuffle(shuffled)
            moved = False
            for dr, dc in shuffled:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self._grid[nr, nc] in BLOCKED:
                    continue
                if (nr, nc) in new_positions:
                    continue
                if (nr, nc) == (amb_r, amb_c):
                    continue

                self._grid[r, c] = Cell.ROAD.value
                self._grid[nr, nc] = Cell.CAR.value
                new_positions.add((nr, nc))
                moved = True
                break
            if not moved:
                new_positions.add((r, c))

        self.car_positions = new_positions

    def reset(self, seed=None, options=None):
        """Resets the ambulance to the start (0,0) and randomizes light phases."""
        self._state = 0
        for loc in self.traffic_lights:
            self.traffic_lights[loc] = [random.choice(list(LightColor)), 0]
        return self._state, {}

    def step(self, action):
        """
        Executes one time step in the environment.
        Args:
            action (int): Direction to move (0:Left, 1:Down, 2:Right, 3:Up).
        Returns:
            tuple: (new_state, reward, done, truncated, info)
        """
        self.update_lights()
        self.move_cars()

        r, c = divmod(self._state, self.cols)
        moves = {
            GridAction.LEFT.value: (0, -1),
            GridAction.DOWN.value: (1, 0),
            GridAction.RIGHT.value: (0, 1),
            GridAction.UP.value: (-1, 0),
        }
        dr, dc = moves[action]
        nr = max(0, min(r + dr, self.rows - 1))
        nc = max(0, min(c + dc, self.cols - 1))

        cell = self._grid[nr, nc]
        new_state = self._state
        reward = -1
        done = False

        if cell == Cell.TRAFFIC.value:
            color = self.traffic_lights[(nr, nc)][0]
            if color == LightColor.RED:
                reward = -10
            elif color == LightColor.YELLOW:
                reward = -3
                new_state = nr * self.cols + nc
            else:
                reward = -1
                new_state = nr * self.cols + nc
        elif cell == Cell.HOUSE.value:
            reward = -20
        elif cell == Cell.CAR.value:
            reward = -15
        else:
            new_state = nr * self.cols + nc
            if cell == Cell.HOSPITAL.value:
                reward = 100
                done = True
            else:
                reward = -1

        prev_state = self._state
        self._state = new_state
        # check if new state is same as previous
        if new_state == prev_state:
            reward = -5

        return new_state, reward, done, False, {}

    def render_frame(self, state, lights_snapshot):
        """
        Generates an RGB image representing the current grid state.
        Args:
            state (int): Current ambulance position.
            lights_snapshot (dict): Current state of traffic lights for rendering.
        Returns:
            numpy.ndarray: The rendered frame.
        """
        cell_size = 40
        img = PILImage.new(
            "RGB", (self.cols * cell_size, self.rows * cell_size), "#808080"
        )
        draw = ImageDraw.Draw(img)

        for r in range(self.rows):
            for c in range(self.cols):
                rect = [
                    c * cell_size,
                    r * cell_size,
                    (c + 1) * cell_size,
                    (r + 1) * cell_size,
                ]
                draw.rectangle(rect, outline="black", width=1)
                val = self._grid[r, c]

                # House
                if val == Cell.HOUSE.value:
                    x1 = c * cell_size + 5
                    y1 = r * cell_size + 10
                    x2 = (c + 1) * cell_size - 5
                    y2 = (r + 1) * cell_size - 5
                    draw.rectangle([x1, y1, x2, y2], fill="#8B4513", outline="black")
                    draw.polygon(
                        [(x1, y1), ((x1 + x2) // 2, y1 - 10), (x2, y1)], fill="#A52A2A"
                    )
                    draw.rectangle([x1 + 8, y2 - 12, x1 + 16, y2], fill="#654321")
                    draw.rectangle([x2 - 15, y1 + 5, x2 - 5, y1 + 15], fill="#ADD8E6")

                # Hospital
                elif val == Cell.HOSPITAL.value:
                    draw.rectangle(rect, fill="white", outline="black")
                    cx = c * cell_size + cell_size // 2
                    cy = r * cell_size + cell_size // 2
                    draw.line([cx, cy - 10, cx, cy + 10], fill="red", width=4)
                    draw.line([cx - 10, cy, cx + 10, cy], fill="red", width=4)

                # Car
                elif val == Cell.CAR.value:
                    x1 = c * cell_size + 6
                    y1 = r * cell_size + 14
                    x2 = (c + 1) * cell_size - 6
                    y2 = (r + 1) * cell_size - 8
                    draw.rounded_rectangle(
                        [x1, y1, x2, y2], radius=6, fill="#3498db", outline="black"
                    )
                    draw.rectangle([x1 + 6, y1 - 6, x2 - 6, y1 + 4], fill="#2980b9")
                    draw.rectangle([x1 + 8, y1 - 4, x2 - 8, y1 + 2], fill="#add8e6")
                    draw.ellipse([x1 + 2, y2 - 6, x1 + 8, y2], fill="black")
                    draw.ellipse([x2 - 8, y2 - 6, x2 - 2, y2], fill="black")

                # Traffic Light
                elif val == Cell.TRAFFIC.value:
                    color_obj = lights_snapshot.get((r, c))[0]
                    c_hex = (
                        "#27ae60"
                        if color_obj == LightColor.GREEN
                        else "#f1c40f" if color_obj == LightColor.YELLOW else "#e74c3c"
                    )
                    draw.ellipse(
                        [
                            c * cell_size + 10,
                            r * cell_size + 10,
                            (c + 1) * cell_size - 10,
                            (r + 1) * cell_size - 10,
                        ],
                        fill=c_hex,
                        outline="black",
                    )

        # Ambulance
        r_curr, c_curr = divmod(state, self.cols)
        x1 = c_curr * cell_size + 8
        y1 = r_curr * cell_size + 12
        x2 = (c_curr + 1) * cell_size - 8
        y2 = (r_curr + 1) * cell_size - 12
        draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=6, fill="#f1c40f", outline="black", width=2
        )
        draw.rectangle([x1 + 6, y1 + 4, x2 - 6, y1 + 10], fill="#add8e6")
        draw.ellipse([x1 + 2, y2 - 6, x1 + 8, y2], fill="black")
        draw.ellipse([x2 - 8, y2 - 6, x2 - 2, y2], fill="black")

        return np.array(img)


# DEVELOPER DOCUMENTATION: ALGORITHM IMPLEMENTATION GUIDE
"""
GUIDE FOR IMPLEMENTING ALGORITHMS RL:

1. ENVIRONMENT INTERFACE:
   - Reset the environment using `state, info = env.reset()`.
   - Take actions using `next_state, reward, done, truncated, info = env.step(action)`.
   - Action Mapping: {0: Left, 1: Down, 2: Right, 3: Up}.

2. STATE SPACE:
   - The state is a Discrete integer from 0 to (size*size - 1).
   - Coordinate Conversion: `row, col = divmod(state, env.cols)`.

3. REWARD SYSTEM & CONSTRAINTS:
   - Default step/Green light: -1
   - Yellow Light: -3 (Ambulance can pass but penalized)
   - Red Light: -10 (Ambulance stays in place if it attempts to enter)
   - Collision (House): -20 (Ambulance stays in place)
   - Collision (Moving Car): -15 (Ambulance stays in place)
   - Returning to the same state: -5
   - Goal (Hospital): +100 (Episode ends)

4. DYNAMIC OBSTACLES:
   - Dynamic cars move every `env.car_move_rate` steps.
   - Traffic lights cycle through Green -> Yellow -> Red automatically.

5. RENDERING FOR ANIMATION:
   - Required imports:
       import matplotlib.pyplot as plt
       from matplotlib.animation import FuncAnimation
       import copy
   - Store states and traffic light snapshots during the episode:
       - Save `state` at each step.
       - Save a deep copy of traffic lights using:
           copy.deepcopy(env.traffic_lights)
   - Use:
       env.render_frame(state, light_snapshot)
     to generate frames.
   - Create animation using Matplotlib:
       - Initialize a figure and axis using plt.subplots()
       - Use FuncAnimation to update frames over time
       - Display using plt.show()
   - Notes:
       - copy.deepcopy is important to avoid overwriting past light states.
       - This method works for visualization only and is independent of training.
   - Optional (for Tensor-based approaches):
       - You can convert frames to arrays using NumPy (already returned).
       - These frames can be stored and used later for video generation or deep learning models.
"""
