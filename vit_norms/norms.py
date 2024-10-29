from surgeon_pytorch import Extract
import torch
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from PIL import Image




class Norms:
    def __init__(self, model, k=100):
        self.model = model
        self.k = k  # Number of seeds to expand
        self.embed_dim = 768


        # Create the extraction model
        self.conv_out_model = Extract(model, node_out="encoder.layers.encoder_layer_11.self_attention")
        self.conv_out_model.eval()

    def get_norms(self, image):
        with torch.no_grad():
            out = self.conv_out_model(image)

        if isinstance(out, tuple):
            out = out[0]
        out = out[0, 1:, :].view(196, 12, 64) #1,197,768 -> 196,768 -> 196,12,64
        norms = torch.norm(out, dim=2, keepdim=True) #196,12,64 -> 196,12,1
        return norms


    def visualize_heatmap(self, image, norms, figsize=(20, 15)):
        norm_means = norms.mean(dim=1) #196,12,1 -> 196
        heatmap = norm_means.reshape(14, 14).cpu().numpy()
        plt.figure(figsize=(figsize[0], figsize[1]))

        image = image.cpu().numpy() * 0.5 + 0.5
        image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        image = image.transpose(1, 2, 0)

        plt.imshow(image, alpha=1)

        heatmap_resized = cv2.resize(heatmap, (224,224), interpolation=cv2.INTER_NEAREST)
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.65)
        plt.axis('off')
        plt.show()

    def generate_and_visualize(self, image, figsize=(20, 15)):
        norms = self.get_norms(image.unsqueeze(0))
        self.visualize_heatmap(image, norms, figsize)
        return norms

    def generate(self, image):
        return self.get_norms(image.unsqueeze(0))

    def plot_l2_norm_distribution_single(self, image, title="L2 Norm Distribution", bins=50):
        plt.figure(figsize=(5, 4))

        norms = self.generate(image)

        print(norms.shape)
        norms = norms[:,:,0].flatten()
        print(norms)
        # Plot the L2 norm distribution
        plt.hist(norms, bins=bins, log=True, density=True, alpha=1)
        plt.title(title)
        plt.xlabel(r'$L_2$ norm')
        plt.ylabel(r'$log$ scale')

        plt.tight_layout()
        plt.show()



    def tensorize(self, img_path):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(img_path).convert('RGB')
        return preprocess(image)

    def plot_l2_norm_distribution_folder(self, folder_path, title="L2 Norm Distribution", bins=50):
        all_norms = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)

                image = self.tensorize(image_path)
                norms = self.generate(image)[:, :, 0].flatten()

                all_norms.append(norms)

        all_norms = torch.cat(all_norms, dim=0)

        plt.figure(figsize=(5, 4))
        plt.hist(all_norms.cpu().numpy(), bins=bins, log=True, density=True, alpha=1)
        plt.title(title)
        plt.xlabel(r'$L_2$ norm')
        plt.ylabel(r'#')

        plt.tight_layout()
        plt.show()

