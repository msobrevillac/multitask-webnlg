import pandas as pd
import torch
from torch.utils.data import Dataset


class WebnlgDataset(Dataset):

	def __init__(self, dataframe, tokenizer, src_max_len, tgt_max_len):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.source_len = src_max_len
		self.target_len = tgt_max_len
		self.source = self.data.source
		if 'target' in dataframe.columns:
			self.target = self.data.target
		else:
			self.target = None

	def __len__(self):
		return len(self.source)


	def __getitem__(self, index):

		str_source = str(self.source[index]).strip()
		if self.pretrained.endswith("t5"):
			source = self.tokenizer.batch_encode_plus([str_source], max_length= self.source_len, 
				pad_to_max_length=True, return_tensors='pt', truncation=True)

		source_ids = source['input_ids'].squeeze()
		source_mask = source['attention_mask'].squeeze()

		if self.target is not None:
			if self.pretrained.endswith("t5"):
				str_target = "<pad> " + str(self.target[index]).strip()
				target = self.tokenizer.batch_encode_plus([str_target], max_length= self.target_len,
					pad_to_max_length=True, return_tensors='pt', truncation=True)

			target_ids = target['input_ids'].squeeze()
			target_mask = target['attention_mask'].squeeze()
			return {
				'source_ids': source_ids.to(dtype=torch.long), 
				'source_mask': source_mask.to(dtype=torch.long), 
				'target_ids': target_ids.to(dtype=torch.long)
				}
		else:
			return {
				'source_ids': source_ids.to(dtype=torch.long), 
				'source_mask': source_mask.to(dtype=torch.long)
				}



def process_data(src_path, tgt_path=None, prefix="Generate from webnlg:"):
	src = []
	with open(src_path, "r") as f:
		src = [prefix + " " + line.strip() for line in f]
  
	tgt = []
	if tgt_path is not None:
		with open(tgt_path, "r") as f:
			tgt = [line.strip() for line in f]
			return pd.DataFrame({'source': src, 'target': tgt})
	else:
		return pd.DataFrame({'source': src})

