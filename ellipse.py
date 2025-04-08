import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import math


def draw_vehicles_with_two_ellipses():
    # Define vehicle dimensions (meters)
    VEHICLE_LENGTH = 5
    VEHICLE_WIDTH = 2

    # Define the centers of two vehicles.
    ego_center = (10, 4)
    obs_center = (20, 4)

    # Calculate bottom-left corners for drawing rectangles (centered on the vehicle's midpoint)
    ego_corner = (ego_center[0] - VEHICLE_LENGTH / 2, ego_center[1] - VEHICLE_WIDTH / 2)
    obs_corner = (obs_center[0] - VEHICLE_LENGTH / 2, obs_center[1] - VEHICLE_WIDTH / 2)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw the ego vehicle as a rectangle
    ego_rect = Rectangle(ego_corner, VEHICLE_LENGTH, VEHICLE_WIDTH,
                         edgecolor='green', facecolor='none', linewidth=2, label='Ego Vehicle')
    ax.add_patch(ego_rect)

    # Draw the obstacle vehicle as a rectangle
    obs_rect = Rectangle(obs_corner, VEHICLE_LENGTH, VEHICLE_WIDTH,
                         edgecolor='blue', facecolor='none', linewidth=2, label='Obstacle Vehicle')
    ax.add_patch(obs_rect)

    ax.plot(ego_center[0], ego_center[1], marker='x', color='orange', markersize=12, markeredgewidth=3,
            label='Ego Center')
    ax.plot(obs_center[0], obs_center[1], marker='x', color='orange', markersize=12, markeredgewidth=3,
            label='Obstacle Center')

    # --------------------- Original Ellipses (threshold = 1) ---------------------
    # The original ellipse has semi-axes VEHICLE_LENGTH/2 and VEHICLE_WIDTH/2.
    ego_ellipse_orig = Ellipse(ego_center, width=VEHICLE_LENGTH, height=VEHICLE_WIDTH,
                               edgecolor='green', facecolor='none', linestyle='--', linewidth=2,
                               label='Original Ellipse (Threshold = 1, Ego)')
    ax.add_patch(ego_ellipse_orig)

    obs_ellipse_orig = Ellipse(obs_center, width=VEHICLE_LENGTH, height=VEHICLE_WIDTH,
                               edgecolor='blue', facecolor='none', linestyle='--', linewidth=2,
                               label='Original Ellipse (Threshold = 1, Obstacle)')
    ax.add_patch(obs_ellipse_orig)

    # --------------------- Enlarged Ellipses (threshold = 4) ---------------------
    # Scale the semi-axes by sqrt(4) to achieve the boundary for ellipse value = 4.
    safety_scale = math.sqrt(4)
    new_width = VEHICLE_LENGTH * safety_scale
    new_height = VEHICLE_WIDTH * safety_scale

    ego_ellipse_safe = Ellipse(ego_center, width=new_width, height=new_height,
                               edgecolor='red', facecolor='none', linestyle='--', linewidth=2,
                               label='Safety Ellipse (Threshold = 4)')
    ax.add_patch(ego_ellipse_safe)

    obs_ellipse_safe = Ellipse(obs_center, width=new_width, height=new_height,
                               edgecolor='red', facecolor='none', linestyle='--', linewidth=2)
    ax.add_patch(obs_ellipse_safe)

    # --------------------- Formatting Plot ---------------------
    ax.set_xlabel("X Position (m)", fontsize=14)
    ax.set_ylabel("Y Position (m)", fontsize=14)
    ax.set_title("Safe Case with Threshold = 4", fontsize=16)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_vehicles_with_two_ellipses()
