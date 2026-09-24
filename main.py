import torch
from models import VideoModel


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_segments = 8
    num_class = (10, 20, 30) # noun, verb & action

    model = VideoModel(
        num_class=num_class,
        num_segments=num_segments,
    ).to(device)
    model.eval()

    sample = torch.randn(batch_size, 3, num_segments, 224, 224, device=device) # batch X channels X num_segments X H X W

    with torch.inference_mode():
        verb_logits, noun_logits, action_logits = model(sample)

    print(f"Verb logits shape: {tuple(verb_logits.shape)}")
    print(f"Noun logits shape: {tuple(noun_logits.shape)}")
    print(f"Action logits shape: {tuple(action_logits.shape)}")


if __name__ == "__main__":
    main()
