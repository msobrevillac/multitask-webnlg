from arguments import get_args
import T5Trainer
import numpy as np
import random

if __name__ == "__main__":
	args = get_args()
	global step

if args.pretrained_model == "t5":
	T5Trainer.main(args)
else:
	print("model does not exist!")
