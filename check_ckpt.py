import torch, sys

path = sys.argv[1] if len(sys.argv) > 1 else r"resources\3DSemiconductor\checkpoints\last_checkpoint.pytorch"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("Iterations:", ckpt["num_iterations"])
print("Epochs:    ", ckpt["num_epochs"])
print("Best val MeanIoU:", round(ckpt["best_eval_score"], 4))
lr = ckpt["optimizer_state_dict"]["param_groups"][0]["lr"]
print("Current LR:", lr)
