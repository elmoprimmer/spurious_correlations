from surgeon_pytorch import Extract
import torch
import cv2
import matplotlib.pyplot as plt


class Norms:
    def __init__(self, model, k=100):
        self.model = model
        self.k = k  # Number of seeds to expand

        # Extract model parameters
        #self_attention_layer = model.encoder.layers[-1].self_attention
        #self.in_proj_weight = self_attention_layer.in_proj_weight
        self.embed_dim = 768
        #self.num_heads = self_attention_layer.num_heads
        #print(self.num_heads)
        #self.head_dim = self.embed_dim // self.num_heads

        # Extract the weight matrix for the keys (Wk) from the combined weight matrix
        #self.W_K = self.in_proj_weight[self.embed_dim:2 * self.embed_dim, :]

        # Create the extraction model
        self.conv_out_model = Extract(model, node_out="encoder.layers.encoder_layer_11.self_attention")
        self.conv_out_model.eval()

    def get_norms(self, image):
        with torch.no_grad():
            out = self.conv_out_model(image)

        if isinstance(out, tuple):
            out = out[0]
        out = out[0, 1:, :].view(196, 12, 64) #1,197,768 -> 196,768 -> 196,12,64

        norms = torch.norm(out, dim=1, keepdim=True)
        print(norms.shape)

        return norms


    def visualize_heatmap(self, image, norms, figsize=(20, 15)):
        norm_means = norms.mean(dim=2)
        print(norm_means.shape)
        heatmap = norm_means.reshape(14, 14).cpu().numpy()

        plt.figure(figsize=(figsize[0], figsize[1]))

        image = image.cpu().numpy() * 0.5 + 0.5
        image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        print(image.min().item(),image.max().item())
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

        norms = self.generate(image)[:,0,:]
        print(norms.shape)
        # Plot the L2 norm distribution
        plt.hist(norms, bins=bins, log=True, density=True, alpha=1)
        plt.title(title)
        plt.xlabel(r'$L_2$ norm')
        plt.ylabel(r'$log$ scale')

        plt.tight_layout()
        plt.show()
