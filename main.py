from load_data.loadData import loadImages
from train_data.trainData import trainImages


class SolveEquation(object):

    def __init__(self):
        self.dataset_dir = 'dataset'
        self.train_dataset_save_dir = 'train_data'
        self.dataset_dir = 'train_data/'
        self.model_savedir = 'models'
        self.savefile_model_name = 'final_model'


    def load_dataset(self):
        loadImages(
        	dataset_dir = self.dataset_dir, 
        	save_dir = self.train_dataset_save_dir
        )

    def train_data(self):
    	trainImages(
    		dataset_dir = self.dataset_dir, 
    		model_savedir=self.model_savedir, 
    		savefile_model_name = self.savefile_model_name
    	)


    def __del__(self):
        pass


def main():
    solve = SolveEquation()
    # solve.load_dataset()
    solve.train_data()


if __name__ == '__main__':
    main()
