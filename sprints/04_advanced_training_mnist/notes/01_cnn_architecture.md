# Notes: CNN Architecture - Seeing the World Like a Computer

Imagine you're teaching a baby (a very smart, computer baby) to recognize a cat.

1.  **Old Way (Painful):** You describe _exactly_ what a cat looks like: pointy ears _here_, whiskers _this_ long, furry texture _all over_. If the cat turns its head? Your description breaks. Ugh.
2.  **CNN Way (Awesome):** You show the baby _parts_ of cats. "Look, pointy ear shapes!" "See this fuzzy stuff? That's fur!" "These lines? Whiskers!"
    - The baby learns to spot these _features_ (pointy ears, fur, whiskers) anywhere in a picture (**Convolution**).
    - It figures out the _most important_ features in small areas ("Yep, definitely pointy ears here!") and ignores minor details (**Pooling**).
    - It stacks these feature-spotters, learning simple features (lines, curves) first, then combining them into complex ones (eyes, noses, whole cat faces!) (**Deep Layers**).
    - Finally, it looks at all the features it found and makes a guess: "Based on pointy ears, fur, and whiskers... CAT!" (**Fully Connected Layer + Classification**).

CNNs are awesome because they learn the important features _themselves_, instead of you hand-coding them. They're **translation invariant** (a cat is a cat, even if it's in the corner of the picture) and build a **hierarchical** understanding (simple shapes -> complex objects).

## A More Detailed Look

Convolutional Neural Networks (CNNs or ConvNets) revolutionized computer vision. Before them, image recognition involved a lot of manual "feature engineering" – telling the computer _exactly_ what patterns to look for (like the "Old Way" above). This was brittle and didn't scale well.

### Key Concepts:

1.  **Convolution Layer:**

    - **The Core Idea:** Instead of looking at the whole image at once, small filters (kernels) slide across the image. Think of a tiny magnifying glass scanning for specific patterns (edges, corners, textures).
    - **Filter/Kernel:** A small matrix of weights. Each filter learns to detect a specific feature (e.g., a vertical edge, a specific color blob).
    - **Feature Map:** The output of applying a filter across the image. It shows where the filter found its specific feature. A convolution layer usually has many filters, producing multiple feature maps.
    - **Parameters:** Stride (how many pixels the filter jumps), Padding (adding borders to the image to control output size).
    - **Why it's cool:** _Parameter sharing_ (the same filter is used across the whole image, drastically reducing the number of weights compared to a fully connected layer) and _translation invariance_ (the filter finds the feature wherever it appears).

2.  **Activation Function (ReLU):**

    - **Purpose:** Introduce non-linearity, just like in regular neural networks. Applied after the convolution.
    - **Common Choice:** ReLU (`max(0, x)`) is popular because it's simple and works well.

3.  **Pooling Layer (e.g., Max Pooling):**

    - **Purpose:** Reduce the spatial dimensions (width/height) of the feature maps. Makes the network more robust to small variations in feature location and reduces computation.
    - **How it works (Max Pooling):** Slides a window across the feature map and takes the _maximum_ value in that window. Effectively keeps the strongest "signal" for a feature in that local region.
    - **Other types:** Average Pooling exists but Max Pooling is more common.

4.  **Stacking Layers (Going Deep):**

    - CNNs typically stack multiple Convolution -> Activation -> Pooling blocks.
    - **Hierarchical Feature Learning:** Early layers learn simple features (edges, corners). Deeper layers combine these simple features to detect more complex patterns (shapes, objects, faces).

5.  **Flattening:**

    - After several convolutional/pooling layers, the 3D feature maps (width x height x channels/filters) are flattened into a single long vector. This prepares the data for the final classification part.

6.  **Fully Connected Layers (FC):**
    - One or more standard neural network layers (like we used before) are added at the end.
    - **Purpose:** Take the high-level features learned by the convolutional layers and perform the final classification (e.g., decide if the image contains a cat, dog, or donut).
    - The last FC layer typically has as many neurons as there are classes, often followed by a Softmax activation for probability outputs.

### A Bit of History:

- **Neocognitron (1980s):** Fukushima developed an early model inspired by the human visual cortex, featuring layers that detected features hierarchically. A precursor to modern CNNs.
- **LeNet-5 (1990s):** Yann LeCun (a deep learning pioneer!) developed LeNet-5 for handwritten digit recognition (like MNIST!). It had the core structure of modern CNNs: convolution, pooling, and fully connected layers. It was highly influential but limited by computing power and data availability at the time.
- **AlexNet (2012):** This is the network that really kicked off the deep learning revolution in computer vision. Developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, it won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a huge margin. It was much deeper than LeNet, used ReLU activations (faster training!), employed dropout for regularization, and leveraged GPU acceleration. Its success showed the power of deep CNNs on large datasets.
- **Since AlexNet:** Many refinements and deeper/more efficient architectures have emerged (VGG, GoogLeNet, ResNet, etc.), but the fundamental principles pioneered by LeNet and popularized by AlexNet remain central.

## Connecting Theory to Practice: Our `SimpleCNN`

The code example we created in `sprints/04_advanced_training_mnist/results/01_define_cnn.py` is a practical implementation of these core ideas in their simplest form.

- It uses one `nn.Conv2d` layer to learn initial features.
- It applies `nn.ReLU` for non-linearity.
- It uses one `nn.MaxPool2d` layer to downsample and provide basic translation invariance.
- It `nn.Flatten`s the output to prepare for classification.
- It uses one `nn.Linear` layer as the final classifier.

This represents a very basic, minimal CNN architecture. Real-world CNNs are often much deeper (stacking many more conv/pool layers) and may incorporate more advanced techniques (different types of padding, strides, normalization layers, residual connections, etc.), but this `SimpleCNN` provides a solid foundation for understanding the fundamental building blocks. It's a good starting point for understanding the fundamental building blocks of CNNs. It looks like this in code:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
```


More involved CNNs will have multiple convolutional layers, each with its own filters and activation functions, and a final fully connected layer for classification. For example, the `SimpleCNN` we created in `sprints/04_advanced_training_mnist/results/01_define_cnn.py` is a simplified version of the VGG architecture, which is a popular CNN architecture for image classification. A more complex example is for instance the `ResNet` architecture. It looks like this:

```python
class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
``` 

