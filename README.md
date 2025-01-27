# Zombie Hunter with Reinforcement Learning

## Description
**Zombie Hunter with Reinforcement Learning** is a Python-based game that combines AI-powered gameplay with classic zombie-shooting action. The AI agent, controlled by a Deep Q-Network (DQN), learns to survive, shoot zombies, and progress through levels while maintaining its health and avoiding dangers. The game features reinforcement learning mechanics, dynamically spawning zombies, and a visually appealing interface using **Pygame**.

---

## Features
- **AI-Powered Gameplay**: The AI agent learns optimal strategies using reinforcement learning.
- **Levels and Progression**: Gradual difficulty increases with zombies starting from level 1 (25 zombies) to level 5 (300 zombies).
- **Dynamic Actions**: AI can move, shoot, and avoid zombies dynamically.
- **Health and Damage Mechanics**: AI health depletes if hit by zombies or if stationary for too long.
- **Blinking Animation**: Visual feedback for damage events.
- **Sound Effects**: Optional sound effects for shooting, taking damage, zombie deaths, and level progression.

---

## Technologies Used
- **Python**: Main programming language.
- **Pygame**: For game visuals and mechanics.
- **PyTorch**: For implementing the Deep Q-Network (DQN).
- **NumPy**: For numerical operations.
- **Random**: For spawning zombies at random locations.
- **Math**: For calculating distances and angles.

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/zombie-hunter-rl.git
   cd zombie-hunter-rl
Install Dependencies: Ensure you have Python installed. Then install the required libraries:
bash
Copy
Edit
pip install pygame torch numpy
Add Sound Effects (Optional):
Place shoot.wav, hit.wav, zombie_death.wav, and level_up.wav in the project directory. If these files are not found, the game will work without sound.
How to Play
Run the game:
bash
Copy
Edit
python zombie_hunter_rl.py
Observe the AI agent:
The agent learns to avoid damage, shoot zombies, and survive as long as possible.
Progress through levels:
Levels increase as zombies are cleared. The number of zombies increases with each level.
Game Over:
If the AI's health reaches zero, the game resets to level 1.
AI Mechanism
State Representation: Includes the AI's position, nearest zombie's position, and health.
Actions: Move up, down, left, right, or shoot at zombies.
Reward System:
Positive rewards for killing zombies and moving strategically.
Negative rewards for losing health or staying idle.
Training:
The DQN is trained with experiences stored in a replay buffer.
Target network updates every 100 steps for stable learning.
Game Details
Screen Size: 1200 x 700 pixels.
Frame Rate: 60 FPS.
AI Features:
Rotating animation for visual enhancement.
Blinking circle effect when damaged.
Zombie Mechanics:
Zombies spawn at random edges and move toward the AI.
Zombies have health bars and can deal damage on contact.
Bullet Mechanics:
Bullets track and hit zombies, reducing their health.
Future Enhancements
Add more zombie types (e.g., faster zombies, boss zombies).
Introduce power-ups for the AI (e.g., health packs, temporary speed boosts).
Allow user-controlled gameplay alongside AI.
Implement a leaderboard to track high scores.
Contributing
Fork the repository.
Create a feature branch:
bash
Copy
Edit
git checkout -b feature-name
Commit changes:
bash
Copy
Edit
git commit -m "Description of changes"
Push and create a pull request:
bash
Copy
Edit
git push origin feature-name
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Inspired by classic arcade zombie games.
Special thanks to the open-source community for providing tools like Pygame and PyTorch.
