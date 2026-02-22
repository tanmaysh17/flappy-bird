"""CLI entry point for Flappy Bird RL."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Flappy Bird Reinforcement Learning")
    sub = parser.add_subparsers(dest="command")

    # ── train ─────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Train the DQN agent")
    tp.add_argument("--visual", action="store_true", help="Show live pygame UI during training")
    tp.add_argument("--num-episodes", type=int, default=10000)
    tp.add_argument("--checkpoint-every", type=int, default=500)
    tp.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    tp.add_argument("--load", type=str, default=None, help="Resume from checkpoint")

    # ── play ──────────────────────────────────────────────────────
    pp = sub.add_parser("play", help="Watch agent or play manually")
    pp.add_argument("--human", action="store_true", help="Play with keyboard (spacebar)")
    pp.add_argument("--model", type=str, default="checkpoints/agent_final.pt")
    pp.add_argument("--fps", type=int, default=60)

    args = parser.parse_args()

    if args.command == "train":
        from .train import train
        train(args)
    elif args.command == "play":
        from .ui import play_game
        play_game(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
