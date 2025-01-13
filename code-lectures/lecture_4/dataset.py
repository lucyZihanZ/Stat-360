from torch.utils.data import Dataset
import torch

class AnimalSoundDataset(Dataset):
    def __init__(self):
        # Make sure all sequences are lowercase
        self.sequences = [
            "the cow says moo",
            "the chicken says cluck cluck",
            "the dog says woof woof",
            "the cat says meow",
            "the sheep says baa",
            "the duck says quack",
            "the pig says oink oink",
            "the mouse says squeak",
            "the horse says neigh",
            "the bird says tweet tweet"
        ]
        
        # Create vocabulary from all words PLUS our OOD tokens
        all_words = set()
        for seq in self.sequences:
            all_words.update(seq.lower().split())
            
        # Add OOD tokens to vocab (also lowercase)
        ood_tokens = {"grad", "student"}
        all_words.update(ood_tokens)
        
        self.vocab = ["<pad>"] + sorted(list(all_words))
        self.word2idx = {token: idx for idx, token in enumerate(self.vocab)}
        
        # Find max sequence length for padding
        self.max_length = max(len(seq.split()) for seq in self.sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        tokens = self.sequences[idx].split()
        # Convert tokens to indices
        indices = [self.word2idx[token] for token in tokens]
        
        # Pad sequence to max_length
        while len(indices) < self.max_length:
            indices.append(self.word2idx['<pad>'])
            
        return torch.tensor(indices)
    
    @staticmethod
    def get_test_prompts():
        return [
            "the cow says",
            "the dog says",
            "the cat says",
            "the duck says",
            "the pig says",
        ]