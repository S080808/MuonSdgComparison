set -e

echo ""
echo "[1/2] MLP на MNIST..."
python scripts/experiment1_mnist.py

echo ""
echo "[2/2] CNN на CIFAR-10..."
python scripts/experiment2_cifar10.py
