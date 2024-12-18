import sys
from pathlib import Path

from src.multiagent_ppo import train_kiyotaki_wright

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.append(str(src_path))


if __name__ == "__main__":
    env, policy = train_kiyotaki_wright()
    print("\nTraining complete!")
