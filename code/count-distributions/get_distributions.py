import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import SimpleITK as sitk
import os
import sys
import argparse
sys.path.append("..") 
from utils.postprocessing import create_annotations_lipid, create_annotations_calcium

class Get_Distributions:

    def __init__(self, data_path: str, output_filename: str, data_info: str):
        """_summary_

        Args:
            data_path (str): path to the folder with the segmentations you want to count (e.g labelsTr, labelsTs or preds from you model)
            output_filename (str): path to the Excel file you want to generate with the generated distributions data
            data_info (str): path to the Excel file with the patients data (i.e train_test_split_v2.xlsx)
        """        
        
        self.data_path = data_path
        self.output_filename = output_filename
        self.data_info = pd.read_excel('{}.xlsx'.format(data_info))
        self.num_classes = 13


    def get_patient_data(self, file: str) -> str:
        """Processes the name of the file so we retrieve the pullback name

        Args:
            file (str): raw filename of prediction

        Returns:
            str: pullback name processed
        """        

        #Obtain format of pullback name (it's different than in the dataset counting)
        filename = file.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Obtain pullback name
        n_pullback = file.split('_')[1]
        pullback_name = self.data_info[(self.data_info['Nº pullback'] == int(n_pullback)) & (self.data_info['Patient'] == patient_name)]['Pullback'].values[0]

        return pullback_name


    def get_counts(self):

        """Creates Excel file with the label count and lipid and calcium measurements 
            for every frame that contains annotations in the specified predicted folder. You can use this function
            either for the original dataset (i.e labelsTs or labelsTr folder) and the predicted segmentations (i.e in your model predictions)
        """

        counts_per_frame = pd.DataFrame(columns = ['pullback', 'frame', 'background', 'lumen', 'guidewire', 'wall', 'lipid', 'calcium', 
                                    'media', 'catheter', 'sidebranch', 'rthrombus', 'wthrombus', 'dissection',
                                    'rupture', 'lipid_arc', 'cap_thickness', 'calcium_depth', 'calcium_arc', 'calcium_thickness'])


        for file in os.listdir(self.data_path):

            #Check only nifti files
            if file.endswith('nii.gz'):

                pullback_name = self.get_patient_data(file)
                n_frame = file.split('_')[2][5:]

                print('Counting {} ...'.format(file))

                seg_map = sitk.ReadImage(os.path.join(self.data_path, file))
                seg_map_data = sitk.GetArrayFromImage(seg_map)

                #Get count of labels in each frame
                one_hot = np.zeros(self.num_classes)

                unique, _ = np.unique(seg_map_data, return_counts=True)
                unique = unique.astype(int)

                one_hot[[unique[i] for i in range(len(unique))]] = 1

                #Post-processing results
                _, _ , cap_thickness, lipid_arc, _ = create_annotations_lipid(seg_map_data[0], font = 'mine')
                _, _ , calcium_depth, calcium_arc, calcium_thickness, _ = create_annotations_calcium(seg_map_data[0], font = 'mine')

                #Create one hot list with all data
                one_hot_list = one_hot.tolist()
                one_hot_list.insert(0, pullback_name)
                one_hot_list.insert(1, n_frame)
                one_hot_list.append(lipid_arc)
                one_hot_list.append(cap_thickness)
                one_hot_list.append(calcium_depth)
                one_hot_list.append(calcium_arc)
                one_hot_list.append(calcium_thickness)
                counts_per_frame = counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

        counts_per_frame.to_excel('{}.xlsx'.format(self.output_filename))

    def get_class_weights(self):
        """Obtain the class weights for the training. IMPORTANT: data_path must be the path to the labelsTr, otherwise you dont get the true
        distribution of the training set
        """        

        label_counts = np.zeros(self.num_classes)

        for file in os.listdir(self.data_path):

            seg = sitk.ReadImage(os.path.join(self.data_path, file))
            seg_data = sitk.GetArrayFromImage(seg)

            #Count the nº of pixels for a label 
            unique, counts = np.unique(seg_data, return_counts=True)
            label_counts[unique] += counts
                
        #Get class weight in terms of frequency in the dataset
        total_pixels = 704 * 704 * len(os.listdir(self.data_path))
        class_weights = total_pixels / (13 * label_counts)

        print("Class weights:")
        for label, weight in enumerate(class_weights):
            print(f"Label {label}: {weight}")

    
def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_filename', type=str)
    parser.add_argument('--data_info', type=str, default='Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2')
    args, _ = parser.parse_known_args(argv)

    args = parser.parse_args()

    counts = Get_Distributions(args.data_path, args.output_filename, args.data_info)
    counts.get_counts()


if __name__ == "__main__":
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)