"""Train a small synthetic binary classifier on Jetson GPU (or CPU fallback)."""
import argparse
import random
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 100:
        parser.error('--epochs must be >= 1 and --samples must be >= 100')

    try:
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is not installed in this Python environment. See README.md.') from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch: {torch.__version__}')
    print(f'PyTorch CUDA build: {torch.version.cuda}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Compute device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('WARNING: Running on CPU. Check JetPack/PyTorch compatibility for GPU acceleration.')

    # Reproducible synthetic nonlinear classification task.
    x = torch.randn(args.samples, 2)
    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.4).long()
    # Train/test split using a seeded permutation.
    perm = torch.randperm(args.samples)
    split = int(args.samples * 0.8)
    train_ids, test_ids = perm[:split], perm[split:]
    x_train, y_train = x[train_ids].to(device), y[train_ids].to(device)
    x_test, y_test = x[test_ids].to(device), y[test_ids].to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch:>3}/{args.epochs} | loss: {loss.item():.4f}')

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()
    print(f'Test accuracy: {accuracy:.2%}')
    print(f'Training time: {elapsed:.3f} s')
    print('Demo completed successfully.')


if __name__ == '__main__':
    main()
