# Neural Network Project

A simple feedforward neural network built in Python using NumPy. It features customizable activation functions, cross-entropy loss, and backpropagation training.

## Features

* Modular neural network implementation
* Supports easy swapping of data loaders
* Clear code structure for activations, loss, and training
* Example dummy data included

## Getting Started

### Installation

First, install the required dependencies. You’ll mainly need NumPy.

```
pip3 install numpy
```

### Running the Training

To train the network using the example dummy data, run the training script.

```
python3 scripts/main.py
```

---

## Adding Your Own Data

The `data.py` file is designed as an open template where you can add your own data loader and preprocessing logic.

Your data loader should:

* Return a list (or iterable) of tuples `(x, y)`
* Each `x` must be a NumPy array shaped `(input_size, 1)`
* Each `y` must be a one-hot encoded NumPy array shaped `(num_classes, 1)`

---

## Project Structure

* `src/` - main code modules
* `scripts/` - runnable scripts like training
* `README.md` - this file
* `requirements.txt` - Python dependencies

---

## Contributing

Feel free to fork, modify, and submit pull requests! The project is designed to be modular and beginner-friendly.

---

# data.py template content (to paste directly into data.py)

```python
import numpy as np

def load_data():

    # Example: empty placeholder data
    data = []

    # TODO: Load your dataset here.
    # For example, if you have raw inputs and labels:
    # raw_X = ...  # shape (num_samples, input_size)
    # raw_labels = ...  # shape (num_samples,)
    #
    # num_classes = ... # number of classes in your dataset
    #
    # for x_raw, label in zip(raw_X, raw_labels):
    #     x = x_raw.reshape(-1, 1)  # convert to (input_size, 1)
    #     y = np.zeros((num_classes, 1))
    #     y[label, 0] = 1  # one-hot encode
    #     data.append((x, y))

    return data
```
