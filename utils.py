import torch
import numpy as np

def get_hard_triplets(embeddings, labels):
    '''
    Selects hard positives and negatives based on Euclidean distances.
    '''
    labels = labels.cpu().numpy()
    pairwise_distances = torch.cdist(embeddings, embeddings)  # Compute pairwise distances

    hard_triplets = []
    for i in range(len(labels)):
        anchor_idx = i
        anchor_label = labels[i]

        # Hard positive: Closest with same label
        positive_indices = np.where(labels == anchor_label)[0]
        positive_indices = positive_indices[positive_indices != i]  # Exclude self
        if len(positive_indices) == 0:
            continue
        positive_idx = positive_indices[torch.argmin(pairwise_distances[i, positive_indices])]

        # Hard negative: Furthest with different label
        negative_indices = np.where(labels != anchor_label)[0]
        negative_idx = negative_indices[torch.argmax(pairwise_distances[i, negative_indices])]

        hard_triplets.append((anchor_idx, positive_idx, negative_idx))

    return hard_triplets

def extract_embeddings(model, dataloader, device):
    '''
    Extracts embeddings for images in the dataset.
    '''
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.numpy())

    return torch.cat(all_embeddings), np.array(all_labels)

def retrieve_top_k(embedding, dataset_embeddings, dataset_labels, k=5):
    '''
    Retrieve top-k similar images using Euclidean distance.
    '''
    distances = torch.cdist(embedding.unsqueeze(0), dataset_embeddings)
    top_k_indices = torch.argsort(distances, dim=1)[0][:k]
    return dataset_labels[top_k_indices]