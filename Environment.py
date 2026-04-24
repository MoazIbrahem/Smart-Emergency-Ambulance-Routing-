import numpy as np
from PIL import Image as PILImage, ImageDraw
from enum import Enum
import random
import gymnasium as gym
from collections import deque


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
    Custom RL Environment for Smart Ambulance pathfinding.

    Interface (gym-compatible):
        state, info = env.reset()
        state, reward, done, trunc, info = env.step(action)
        Action mapping: {0: Left, 1: Down, 2: Right, 3: Up}

    Reward System:
        +100 : Reached hospital (episode ends)
          -1 : Normal step / green light
          -3 : Yellow light (passable)
          -5 : No movement (stayed in same cell)
         -10 : Red light (agent stays)
         -15 : Hit a car (agent stays)
         -20 : Hit a house (agent stays)
    """

    def __init__(self, size=20, seed=69, max_steps=500):
        self.size = size
        self.rows = size
        self.cols = size
        self._state = 0
        self.steps = 0
        self.max_steps = max_steps

        random.seed(seed)
        np.random.seed(seed)

        self._grid = np.zeros((size, size), dtype=int)
        self.car_move_counter = 0
        self.car_move_rate = 3

        # Generate Grid
        self._build_grid()

        self.observation_space = gym.spaces.Discrete(size * size)
        self.action_space = gym.spaces.Discrete(len(GridAction))

    def _build_grid(self):
        """Build grid and guarantee hospital is reachable via BFS."""
        for _ in range(50):
            self._grid = np.zeros((self.size, self.size), dtype=int)
            self.car_move_counter = 0
            # Houses
            for _ in range(20):
                r, c = random.randint(0, self.size - 1), random.randint(
                    0, self.size - 1
                )
                if (r, c) != (0, 0) and (r, c) != (self.size - 1, self.size - 1):
                    self._grid[r, c] = Cell.HOUSE.value
            # Cars
            for _ in range(10):
                r, c = random.randint(0, self.size - 1), random.randint(
                    0, self.size - 1
                )
                if self._grid[r, c] == Cell.ROAD.value and (r, c) != (0, 0):
                    self._grid[r, c] = Cell.CAR.value

            self.car_positions = {
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if self._grid[r, c] == Cell.CAR.value
            }
            #  Traffic lights
            self.traffic_lights = {}
            for _ in range(15):
                r, c = random.randint(0, self.size - 1), random.randint(
                    0, self.size - 1
                )
                if self._grid[r, c] == Cell.ROAD.value and (r, c) != (0, 0):
                    self._grid[r, c] = Cell.TRAFFIC.value
                    self.traffic_lights[(r, c)] = [random.choice(list(LightColor)), 0]

            self._grid[self.size - 1, self.size - 1] = Cell.HOSPITAL.value

            if self._is_reachable():
                return

        # empty grid
        print("Warning: using fallback empty grid")
        self._grid = np.zeros((self.size, self.size), dtype=int)
        self._grid[self.size - 1, self.size - 1] = Cell.HOSPITAL.value
        self.car_positions = set()
        self.traffic_lights = {}

    def _is_reachable(self):
        """BFS: verify path exists from (0,0) to hospital."""
        PASSABLE = {
            Cell.ROAD.value,
            Cell.TRAFFIC.value,
            Cell.CAR.value,
            Cell.HOSPITAL.value,
        }
        visited, queue, goal = set(), deque([(0, 0)]), (self.size - 1, self.size - 1)
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.size
                    and 0 <= nc < self.size
                    and (nr, nc) not in visited
                    and self._grid[nr, nc] in PASSABLE
                ):
                    queue.append((nr, nc))
        return False

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
        """Move each car randomly every car_move_rate steps."""
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
        self._state = 0
        self.steps = 0
        self.car_move_counter = 0
        for loc in self.traffic_lights:
            self.traffic_lights[loc] = [random.choice(list(LightColor)), 0]
        return self._state, {}

    def step(self, action):
        self.steps += 1
        self.update_lights()
        self.move_cars()

        r, c = divmod(self._state, self.cols)
        moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        dr, dc = moves[action]
        nr = max(0, min(r + dr, self.rows - 1))
        nc = max(0, min(c + dc, self.cols - 1))

        cell = self._grid[nr, nc]
        new_state = self._state
        reward, done = -1, False

        if cell == Cell.TRAFFIC.value:
            color = self.traffic_lights[(nr, nc)][0]
            if color == LightColor.RED:
                reward = -10
            elif color == LightColor.YELLOW:
                reward, new_state = -3, nr * self.cols + nc
            else:
                reward, new_state = -1, nr * self.cols + nc
        elif cell == Cell.HOUSE.value:
            reward = -20
        elif cell == Cell.CAR.value:
            reward = -15
        else:
            new_state = nr * self.cols + nc
            if cell == Cell.HOSPITAL.value:
                reward, done = 100, True
            else:
                reward = -1

        prev_state = self._state
        self._state = new_state
        if new_state == prev_state:
            reward = -5

        truncated = False
        if self.steps >= self.max_steps and not done:
            truncated = done = True

        return new_state, reward, done, truncated, {}

    def render_frame(self, state, lights_snapshot):
        cs = 40
        img = PILImage.new("RGB", (self.cols * cs, self.rows * cs), "#808080")
        draw = ImageDraw.Draw(img)

        for r in range(self.rows):
            for c in range(self.cols):
                rect = [c * cs, r * cs, (c + 1) * cs, (r + 1) * cs]
                draw.rectangle(rect, outline="black", width=1)
                val = self._grid[r, c]

                if val == Cell.HOUSE.value:
                    x1, y1 = c * cs + 5, r * cs + 10
                    x2, y2 = (c + 1) * cs - 5, (r + 1) * cs - 5
                    draw.rectangle([x1, y1, x2, y2], fill="#8B4513", outline="black")
                    draw.polygon(
                        [(x1, y1), ((x1 + x2) // 2, y1 - 10), (x2, y1)], fill="#A52A2A"
                    )
                    draw.rectangle([x1 + 8, y2 - 12, x1 + 16, y2], fill="#654321")
                    draw.rectangle([x2 - 15, y1 + 5, x2 - 5, y1 + 15], fill="#ADD8E6")

                elif val == Cell.HOSPITAL.value:
                    draw.rectangle(rect, fill="white", outline="black")
                    cx, cy = c * cs + cs // 2, r * cs + cs // 2
                    draw.line([cx, cy - 10, cx, cy + 10], fill="red", width=4)
                    draw.line([cx - 10, cy, cx + 10, cy], fill="red", width=4)

                elif val == Cell.CAR.value:
                    x1, y1 = c * cs + 6, r * cs + 14
                    x2, y2 = (c + 1) * cs - 6, (r + 1) * cs - 8
                    draw.rounded_rectangle(
                        [x1, y1, x2, y2], radius=6, fill="#3498db", outline="black"
                    )
                    draw.rectangle([x1 + 6, y1 - 6, x2 - 6, y1 + 4], fill="#2980b9")
                    draw.rectangle([x1 + 8, y1 - 4, x2 - 8, y1 + 2], fill="#add8e6")
                    draw.ellipse([x1 + 2, y2 - 6, x1 + 8, y2], fill="black")
                    draw.ellipse([x2 - 8, y2 - 6, x2 - 2, y2], fill="black")

                elif val == Cell.TRAFFIC.value:
                    color_obj = lights_snapshot.get((r, c))[0]
                    c_hex = (
                        "#27ae60"
                        if color_obj == LightColor.GREEN
                        else "#f1c40f" if color_obj == LightColor.YELLOW else "#e74c3c"
                    )
                    draw.ellipse(
                        [
                            c * cs + 10,
                            r * cs + 10,
                            (c + 1) * cs - 10,
                            (r + 1) * cs - 10,
                        ],
                        fill=c_hex,
                        outline="black",
                    )

        r_c, c_c = divmod(state, self.cols)
        x1, y1 = c_c * cs + 8, r_c * cs + 12
        x2, y2 = (c_c + 1) * cs - 8, (r_c + 1) * cs - 12
        draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=6, fill="#f1c40f", outline="black", width=2
        )
        draw.rectangle([x1 + 6, y1 + 4, x2 - 6, y1 + 10], fill="#add8e6")
        draw.ellipse([x1 + 2, y2 - 6, x1 + 8, y2], fill="black")
        draw.ellipse([x2 - 8, y2 - 6, x2 - 2, y2], fill="black")

        return np.array(img)
