import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with chess puzzles
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the FEN position, moves, and puzzle number
        fen = self.data.iloc[idx]['FEN']
        moves = self.data.iloc[idx]['Moves']
        puzzle_number = self.data.iloc[idx]['Number']
        
        sample = {
            'fen': fen,
            'moves': moves,
            'puzzle_number': puzzle_number
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# HOW TO USE

# # Step 1: Instantiate the dataset
# dataset = ChessPuzzleDataset(csv_file='games/trainingpuzzles.csv')

# # Step 2: Wrap it with a DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Step 3: Iterate through the dataloader
# for batch_idx, trio in enumerate(dataloader):
#     print(f"Batch {batch_idx}:")
#     print(f'FEN: {trio["fen"]}')
#     print(f'Moves: {trio["moves"]}')
#     print(f'Puzzle Number: {trio["puzzle_number"]}')
#     break
