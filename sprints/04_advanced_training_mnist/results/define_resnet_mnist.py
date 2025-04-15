import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Optional

# Check for MPS (Apple Silicon GPU) availability, otherwise use CUDA or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """

    expansion: int = 1  # No expansion of channels in BasicBlock

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[
            nn.Module
        ] = None,  # Module to handle dimension changes in skip connection
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # Default to BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # Store the input for the skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If the dimensions changed (stride != 1 or channels changed),
        # apply the downsample layer to the identity (skip connection)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the skip connection
        out += identity
        # Apply ReLU after the addition
        out = self.relu(out)

        return out


class ResNetMNIST(nn.Module):
    """
    A simplified ResNet architecture adapted for MNIST (1-channel, 28x28 input).
    Uses BasicBlock.
    """

    def __init__(
        self,
        block: Type[BasicBlock],  # Type of block to use (BasicBlock)
        layers: List[
            int
        ],  # List indicating number of blocks per layer stage (e.g., [2, 2, 2, 2] for ResNet18)
        num_classes: int = 10,  # Number of output classes (10 for MNIST)
        zero_init_residual: bool = False,  # Initialize last BN in each block to zero?
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # Number of input channels for the first layer stage
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # MNIST specific adaptation: Start with smaller kernel and stride
        # Standard ResNet: kernel=7, stride=2, padding=3
        # Adapted for MNIST: kernel=3, stride=1, padding=1
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # Standard ResNet uses MaxPool here. Let's keep it for now, but it significantly
        # reduces dimensions quickly on 28x28. Might remove later if needed.
        # Input: N, 64, 28, 28 -> Output: N, 64, 14, 14
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the residual layer stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        # For simplicity on MNIST, we might stop here or add fewer layers than ResNet18/34
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Output size (1, 1) regardless of input size
        # The number of input features to fc is the number of planes in the *last* layer stage
        last_layer_planes = (
            128 * block.expansion
        )  # Use 128 since we stopped after layer2
        self.fc = nn.Linear(last_layer_planes, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,  # Number of output channels for this stage
        blocks: int,  # Number of residual blocks in this stage
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # Check if we need a downsample layer for the skip connection
        # This is needed if stride is not 1 OR if the number of input channels (self.inplanes)
        # doesn't match the number of output channels (planes * block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = (
            planes * block.expansion
        )  # Update inplanes for the next block/layer
        # Add the remaining blocks for the layer
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # N, 64, 14, 14

        x = self.layer1(x)  # N, 64, 14, 14 (assuming stride=1 in _make_layer)
        x = self.layer2(x)  # N, 128, 7, 7 (assuming stride=2 in _make_layer)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)  # N, 128, 1, 1
        x = torch.flatten(x, 1)  # N, 128
        x = self.fc(x)  # N, num_classes

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _resnet_mnist(block: Type[BasicBlock], layers: List[int], **kwargs) -> ResNetMNIST:
    """Helper function to create ResNetMNIST model"""
    return ResNetMNIST(block, layers, **kwargs)


def resnet10_mnist(**kwargs) -> ResNetMNIST:
    """Constructs a ResNet-10 model adapted for MNIST."""
    # Example: 2 blocks in layer1, 2 blocks in layer2 = 1 (conv1) + 2*2 + 2*2 + 1(fc) = 10 layers roughly
    return _resnet_mnist(BasicBlock, [1, 1], **kwargs)  # Very small ResNet


def resnet18_mnist(**kwargs) -> ResNetMNIST:
    """Constructs a ResNet-18 model adapted for MNIST."""
    # Using only first two layer stages from standard ResNet18 [2, 2, 2, 2]
    return _resnet_mnist(BasicBlock, [2, 2], **kwargs)


# --- Test the Model Definition ---
if __name__ == "__main__":
    print("--- Testing ResNetMNIST Definition ---")

    # Create a dummy input tensor representing a batch of 4 grayscale images (28x28)
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 1, 28, 28).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Instantiate the model (using a smaller ResNet-10 version)
    # model = ResNetMNIST(BasicBlock, layers=[1, 1], num_classes=10).to(device) # ResNet-10 like
    model = resnet10_mnist(num_classes=10).to(device)
    # model = resnet18_mnist(num_classes=10).to(device) # Alternative: ResNet-18 like
    print(f"Model instantiated: {model.__class__.__name__}")  # Print class name

    # --- Perform a forward pass ---
    print("--- Performing Forward Pass ---")
    try:
        # Set model to evaluation mode
        model.eval()
        # Disable gradient calculation
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        # Check if output shape is as expected (batch_size, num_classes)
        assert output.shape == (4, 10)
        print("Forward pass successful! Output shape is correct.")

        # --- Print Model Summary (Number of Parameters) ---
        print("--- Model Summary ---")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback

        traceback.print_exc()

    print("--- Test Complete ---")
