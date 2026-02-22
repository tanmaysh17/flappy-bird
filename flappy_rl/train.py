"""Training loop for headless and visual modes."""

import os
import numpy as np
from .game import FlappyBirdEnv
from .agent import DQNAgent


def train(args):
    env = FlappyBirdEnv()
    agent = DQNAgent(device=getattr(args, "device", "auto"))

    if getattr(args, "load", None):
        agent.load(args.load)
        print(f"Resumed from {args.load}")

    num_episodes = getattr(args, "num_episodes", 10000)
    checkpoint_every = getattr(args, "checkpoint_every", 500)
    visual = getattr(args, "visual", False)

    # logging lists
    episode_scores = []
    episode_rewards = []
    losses = []
    epsilons = []

    ui = None
    steps_per_frame = 1
    if visual:
        from .ui import TrainingUI, STEPS_PER_FRAME
        ui = TrainingUI()
        steps_per_frame = STEPS_PER_FRAME

    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        ep_losses = []
        step_in_frame = 0

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            total_reward += reward
            step_in_frame += 1

            # in visual mode, only render every N steps
            if ui and step_in_frame >= steps_per_frame:
                step_in_frame = 0
                quit_requested = ui.update(env, agent, episode_scores, losses, epsilons)
                if quit_requested:
                    agent.save("checkpoints/agent_interrupted.pt")
                    print("Training interrupted -- checkpoint saved.")
                    return

        episode_scores.append(env.score)
        episode_rewards.append(total_reward)
        losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        epsilons.append(agent.epsilon)

        if episode % 10 == 0:
            avg = np.mean(episode_scores[-100:])
            print(
                f"Ep {episode:5d} | Score: {env.score:3d} | "
                f"Avg100: {avg:.1f} | Eps: {agent.epsilon:.3f} | "
                f"Loss: {losses[-1]:.4f}"
            )

        if checkpoint_every and episode > 0 and episode % checkpoint_every == 0:
            path = f"checkpoints/agent_ep{episode}.pt"
            agent.save(path)
            print(f"  checkpoint -> {path}")

    agent.save("checkpoints/agent_final.pt")
    print("Training complete — saved checkpoints/agent_final.pt")

    if ui:
        ui.close()
