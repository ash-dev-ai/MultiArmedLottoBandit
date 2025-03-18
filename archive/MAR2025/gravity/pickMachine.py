import pymunk
import pymunk.pygame_util
import random
import pygame
import csv


class GravityPickMachine:
    def __init__(self, num_balls, num_selected, mega_balls):
        self.num_balls = num_balls
        self.num_selected = num_selected
        self.mega_balls = mega_balls

    def run_simulation(self, drum_speed, air_jet_force, mixing_time):
        """
        Simulate the gravity pick machine using Pymunk.
        :param drum_speed: Speed of the drum.
        :param air_jet_force: Random force applied to balls.
        :param mixing_time: Mixing time in seconds.
        :return: Dictionary with selected white balls and Mega Ball.
        """
        selected_white = self._simulate_drum(self.num_balls, self.num_selected, drum_speed, air_jet_force, mixing_time)
        selected_mega = random.choice(range(1, self.mega_balls + 1))
        return {"White Balls": selected_white, "Mega Ball": selected_mega}

    def _simulate_drum(self, total_balls, balls_to_select, drum_speed, air_jet_force, mixing_time):
        """
        Simulate the ball mixing using Pymunk.
        """
        # Pymunk space setup
        space = pymunk.Space()
        space.gravity = (0, -900)

        # Create drum
        drum_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        drum_shape = pymunk.Circle(drum_body, 300)
        drum_shape.elasticity = 0.9
        drum_shape.friction = 0.5
        drum_body.position = (400, 400)
        space.add(drum_body, drum_shape)

        # Add balls
        ball_shapes = []
        for i in range(total_balls):
            ball_body = pymunk.Body(0.01, pymunk.moment_for_circle(0.01, 0, 10))
            ball_body.position = (random.randint(300, 500), random.randint(500, 700))
            ball_shape = pymunk.Circle(ball_body, 10)  # Ball radius ~10px
            ball_shape.elasticity = 0.9
            ball_shape.friction = 0.5
            space.add(ball_body, ball_shape)
            ball_shapes.append(ball_shape)

        # Simulate the drum spinning and mixing
        for step in range(int(mixing_time * 60)):  # Assume 60 FPS
            for ball in ball_shapes:
                # Apply random air jet forces
                force = (random.uniform(-air_jet_force, air_jet_force), random.uniform(-air_jet_force, air_jet_force))
                ball.body.apply_impulse_at_local_point(force)
            # Rotate the drum
            drum_body.angle += drum_speed * 0.01
            space.step(1 / 60)

        # Sort balls by their vertical position (lowest selected)
        ball_positions = [(ball.body.position.y, i + 1) for i, ball in enumerate(ball_shapes)]
        ball_positions.sort()  # Sort by height
        selected_balls = [ball[1] for ball in ball_positions[:balls_to_select]]
        return sorted(selected_balls)

    def save_results(self, filepath, results):
        """
        Save the results to a CSV file.
        :param filepath: Path to save the file.
        :param results: Dictionary of results.
        """
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["White Balls", "Mega Ball"])
            writer.writerow([", ".join(map(str, results["White Balls"])), results["Mega Ball"]])
