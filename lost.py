from surgeon_pytorch import Extract
import torch
import cv2
import matplotlib.pyplot as plt


class LOSTHeads:
    def __init__(self, model, k=100):
        self.model = model
        self.k = k  # Number of seeds to expand

        # Extract model parameters
        self_attention_layer = model.encoder.layers[-1].self_attention
        self.in_proj_weight = self_attention_layer.in_proj_weight
        self.embed_dim = 768
        self.num_heads = self_attention_layer.num_heads
        print(self.num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        # Extract the weight matrix for the keys (Wk) from the combined weight matrix
        self.W_K = self.in_proj_weight[self.embed_dim:2 * self.embed_dim, :]

        # Create the extraction model
        self.conv_out_model = Extract(model, node_out="encoder.layers.encoder_layer_11.ln_1")

    def get_object_similarity_heatmap(self, image):
        with torch.no_grad():
            features, num_heads = self.extract_last_layer_keys(image)

        heatmaps = []
        for head in range(num_heads):
            head_features = features[head]
            # Compute patch similarities for this head
            similarities = torch.matmul(head_features, head_features.transpose(0, 1))

            # Select the initial seed
            adjacency = (similarities >= 0).float()
            degrees = adjacency.sum(dim=-1)
            initial_seed = degrees.argmin().item()

            # Expand the seed
            _, top_k_indices = torch.topk(degrees, k=min(self.k, degrees.size(0)), largest=False)
            expanded_seeds = top_k_indices[adjacency[initial_seed, top_k_indices] > 0]

            # Compute similarity to the expanded seeds
            seed_similarities = similarities[:, expanded_seeds].mean(dim=1)

            # Reshape to 14x14 for heatmap visualization
            heatmap = seed_similarities.reshape(14, 14).cpu().numpy()
            heatmaps.append(heatmap)

        return heatmaps

    def extract_last_layer_keys(self, image):
        out = self.conv_out_model(image)
        ln_1 = out[:, 1:, :]

        # Compute keys for each head
        k = ln_1 @ self.W_K.T

        # Reshape to separate heads
        k = k.view(1, 196, self.num_heads, self.head_dim)

        # Reshape to (num_heads, 196, head_dim)
        k = k.squeeze(0).permute(1, 0, 2)

        return k, self.num_heads


    def visualize_heatmaps(self, image, heatmaps, figsize=(20, 15)):
        _, axs = plt.subplots(3, 4, figsize=(20, 15))

        image = image.cpu().numpy() * 0.5 + 0.5
        image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        print(image.min().item(),image.max().item())
        image = image.transpose(1, 2, 0)

        for i, ax in enumerate(axs.flat):
            # also plot the image
            ax.imshow(image, alpha=1)

            heatmap = heatmaps[i]

            # resize the heatmap to the image size with bicubic interpolation
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.normalize(heatmap, heatmap, 0, 1, cv2.NORM_MINMAX)
            ax.imshow(heatmap, cmap='jet', alpha=0.65)

            ax.axis('off')
            ax.set_title(f'Head {i + 1}')

        # plt.tight_layout()
        plt.show()

    def generate_and_visualize(self, image, figsize=(20, 15)):
        heatmaps = self.get_object_similarity_heatmap(image.unsqueeze(0))
        self.visualize_heatmaps(image, heatmaps, figsize)
        return heatmaps

    def generate(self, image):
        return self.get_object_similarity_heatmap(image.unsqueeze(0))