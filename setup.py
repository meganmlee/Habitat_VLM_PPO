#!/usr/bin/env python3
"""
Simple Habitat HM3D Visualization Script
"""

import habitat
import matplotlib.pyplot as plt
import numpy as np


config = habitat.get_config(
    "/home/megan/Habitat_VLM_PPO/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d_with_semantic.yaml",
    overrides=[
        "habitat.dataset.data_path=/home/megan/Habitat_VLM_PPO/data/datasets/objectnav/hm3d/v2/train/train.json.gz"
    ]
)
# Create environment
env = habitat.Env(config=config)

print("Environment created!")
print(f"Action space: {env.action_space}")
print(f"Observation space keys: {env.observation_space.spaces.keys()}")

# Reset and get first observation
obs = env.reset()

print(f"\nEpisode Info:")
print(f"  Scene: {env.current_episode.scene_id}")
print(f"  Goal Object: {env.current_episode.object_category}")
print(f"  Start Position: {env.current_episode.start_position}")

# Visualize RGB and Depth
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if "rgb" in obs:
    axes[0].imshow(obs["rgb"])
    axes[0].set_title(f"RGB View - Goal: {env.current_episode.object_category}")
    axes[0].axis('off')

if "depth" in obs:
    axes[1].imshow(obs["depth"].squeeze(), cmap='viridis')
    axes[1].set_title("Depth View")
    axes[1].axis('off')

plt.tight_layout()
plt.savefig('quick_view.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Saved visualization to: quick_view.png")

# Take a few steps
print("\nTaking 10 random steps...")
for i in range(10):
    action = env.action_space.sample()
    obs = env.step(action)
    
    # Action mapping: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
    action_names = ['STOP', 'FORWARD', 'LEFT', 'RIGHT']
    print(f"  Step {i+1}: {action_names[action] if action < 4 else action}")

# Show final view
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if "rgb" in obs:
    axes[0].imshow(obs["rgb"])
    axes[0].set_title("After 10 Random Steps")
    axes[0].axis('off')

if "depth" in obs:
    axes[1].imshow(obs["depth"].squeeze(), cmap='viridis')
    axes[1].set_title("Depth After Steps")
    axes[1].axis('off')

plt.tight_layout()
plt.savefig('after_steps.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved final view to: after_steps.png")

# Close environment
env.close()
print("\n✓ Done!")