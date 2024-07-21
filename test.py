import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNetModel(nn.Module):
    def __init__(self, num_input_channels=6, num_output_features=3):
        super(CustomResNetModel, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 6 input channels instead of 3
        self.resnet.conv1 = nn.Conv2d(
            num_input_channels, 
            self.resnet.conv1.out_channels, 
            kernel_size=self.resnet.conv1.kernel_size, 
            stride=self.resnet.conv1.stride, 
            padding=self.resnet.conv1.padding, 
            bias=False
        )
        
        # Replace the final fully connected layer to output the desired number of features
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_output_features)
    
    def forward(self, x):
        return self.resnet(x)

# Example usage
model = CustomResNetModel(num_input_channels=6, num_output_features=3)
input_tensor = torch.randn(1, 6, 640, 480)  # Batch size of 1, input shape (6, 640, 480)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([1, 3])
