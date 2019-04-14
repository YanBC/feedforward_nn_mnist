# Description
A simple feedforward neural network to illustrate the inner working of the backpropagation algorithm. Here the default is a two layer feedforward network trained on mnist dataset. Of course, you can create a your own feedforward network with your favorite dataset as well.

# Demo
To see how it works:
1. Get the mnist dataset
2. Train your network
3. Compare the result from step 2 with a `Keras` feedforward network

## Get the mnist dataset
```bash
cd <project root>
cd data/
python3 get_mnist.py
```

## Train your network
```bash
cd <project root>
python3 main.py Homemade
```

## Compare results
```bash
cd <project root>
python3 main.py Keras
```