import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Replace the last fully connected layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool,
        )
        self.get_latent_embedding = True
        
    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        output = self.resnet.fc(features)
        ## normalize the feature vector
        # F.normalize(features, dim=1)
        return output, F.normalize(features, dim=1)


    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        # Replace the last fully connected layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool,
        )
        self.get_latent_embedding = True

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        output = self.resnet.fc(features)
        # Normalize the feature vector
        return output, F.normalize(features, dim=1)
        

class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Access the layer before self.resnet.fc
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        ## latent embedding
        self.get_latent_embedding = True
        
        
    def forward(self, x):
        
        if self.get_latent_embedding:
            x = self.features(x)  # Pass input through the feature extractor
            x = x.view(x.size(0), -1)  # Flatten the feature tensor
            return self.resnet.fc(x), x
        elif self.get_latent_embedding == False:
            x = self.features(x)  # Pass input through the feature extractor
            x = x.view(x.size(0), -1)  # Flatten the feature tensor
            x = self.resnet.fc(x)  # Pass the flattened features through the new FC layer
            return x
    

def get_model(model_type):
    
    if model_type == 'resnet18':
        return ResNet18()
    elif model_type == 'resnet34':
        return ResNet34()
    elif model_type == 'ResNet50':
        return ResNet50()
    else:
        raise ValueError('Model not supported')
    
    
## test 
def test_case():
    net = ResNet34()
    net.get_latent_embedding = False
    a,b = net(torch.randn(10, 3, 304, 304))
    print(a.size())

#test_case()