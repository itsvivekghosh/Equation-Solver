import argparse
ap = argparse.ArgumentParser(usage = "--[FILE_PATH]")
ap.add_argument('-i', '--image', help='Path to Image', required=True)
args = vars(ap.parse_args())


from load_data.loadData import loadAndSaveImages
from train_data.trainData import trainImages
from test_data.testData import testImages



class SolveEquation(object):

    def __init__(self):
        self.dataset_dir = 'dataset'
        self.train_dataset_save_dir = 'train_data'
        self.model_savedir = 'models'
        self.savefile_model_name = 'final_model'
        self.image_dir = args['image']
        self.answer, self.prediction = None, None


    def load_and_save_dataset(self):
        loadAndSaveImages(
        	dataset_dir = self.dataset_dir, 
        	save_dir = self.train_dataset_save_dir
        )


    def train_data(self):
    	trainImages(
    		dataset_dir = self.train_dataset_save_dir, 
    		model_savedir=self.model_savedir, 
    		savefile_model_name = self.savefile_model_name
    	)


    def test_data(self):
    	self.prediction, self.answer = testImages(
    		model_dir = self.model_savedir, model_name = self.savefile_model_name, image_dir = self.image_dir
    	)


    def __del__(self):
        print("Predicted Equation: {}".format(self.prediction))
        print("Answer: {}".format(self.answer))



def main():
	solve = SolveEquation()
	# solve.load_and_save_dataset()
	# solve.train_data()
	solve.test_data()

if __name__ == '__main__':
	main()
	
