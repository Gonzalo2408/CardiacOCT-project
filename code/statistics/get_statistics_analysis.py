import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import pingouin as pg
from scipy import stats
from typing import Tuple
import argparse
import os
import sys
import math
import warnings
from matplotlib.colors import ListedColormap
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

class Statistics:

    def __init__(self, model_id: str, output_filename: str, measures_file: str, png_path: str):
        """Class to obtain the statistics analysis for a specific model

        Args:
            model_id (str): arbitrary ID for a specific model (must be the same name as you put in the measures file!)
            output_filename (str): name of the generated Excel file
            measures_file (str): path to the Excel file that contains all measurements values
            png_path (str): path to save all PNG figues
        """

        self.model_id = model_id
        self.output_filename = output_filename
        self.measures_file = pd.read_excel(measures_file)
        self.png_path = png_path

    def get_data_filtered(self, test_sheet: pd.DataFrame, type_manual: str, type_ai: str, missing: int = -99) -> Tuple[pd.Series, pd.Series, int, int]:        
        """Obtain lipid/calcium values without FP or FN or nulls

        Args:
            test_sheet (pd.DataFrame): dataframe with all the measurements
            type_manual (str): specific column of manual annotation
            type_ai (str): specific column of automatic annotation (must be same type as type_manual)
            missing (int, optional): encoding of missing values. Defaults to -99.

        Returns:
            pd.Series, pd.Series: series for both the manual and predicted measurements
            int, int: values for the FP and FN
        """    

        manual = test_sheet[type_manual]
        ai = test_sheet[type_ai]

        #We look for FP, FN and nan (we only keep TP and TN)
        list_fp = []
        list_fn = []
        list_nulls = []
        list_drops = []
        fp_count = 0
        fn_count = 0

        for value in range(len(manual)):

            if manual[value] == missing and ai[value] != missing:
                list_fp.append(value)
                fp_count += 1

            if manual[value] != missing and ai[value] == missing:
                list_fn.append(value)
                fn_count += 1

            if manual[value] == missing and ai[value] == missing:
                list_nulls.append(value)

            if math.isnan(ai[value]):
                list_nulls.append(value)

        #Create final list that contains all indexes to remove
        list_drops.extend(list_fp)
        list_drops.extend(list_fn)
        list_drops.extend(list_nulls)

        #Drop those cases
        ai.drop(list_drops, inplace=True)
        manual.drop(list_drops, inplace=True)

        return manual, ai, fp_count, fn_count


    def mean_sd(self, manual: pd.Series, automatic: pd.Series) -> Tuple[int, int]:
        """Obtain mean difference and standard deviation

        Args:
            manual (pd.Series): series containing the manual measurements
            automatic (pd.Series): series containing the automated measurements

        Returns:
            int, int: values for mean difference and SD
        """        

        diff = automatic - manual                  
        md = np.mean(diff)                   
        sd = np.std(diff, axis=0) 
        
        return  md, sd


    def corr(self, manual: pd.Series, automatic: pd.Series) -> float:
        """Get correlation

        Args:
            manual (pd.Series): measurements for orginials
            automatic (pd.Series): measurements for predictions

        Returns:
            float: Correlation of manual and automatic measurements
        """    

        return np.corrcoef(manual, automatic)[0][1]


    def scatter_data_save_png(self, manual: pd.Series, automatic: pd.Series, thresh: int, title: str, png_name: str):
        """Creates scatter plot and saves the figure in a folder

        Args:
            manual (pd.Series): measurements for originals
            automatic (pd.Series): measurements for predictions
            thresh (int): critical values for some measurements (e.g 65 for FCT, or 90 for lipid arc)
            title (str): title of the plot
            png_name (str): name of the saved PNG file
        """    
        
        #Depending on the values, we have three cases: 0 and 1 for FP/FN, and 2 for TP/TN
        col = []
        for i, j in zip(manual, automatic):
            if i <= thresh and j >= thresh:
                col.append(0)

            elif i >= thresh and j <= thresh:
                col.append(1)

            else:
                col.append(2)

        fig, ax = plt.subplots()

        # For the case in which there are no FP or FN, we only use one color
        if all(c == 2 for c in col):
            ax.scatter(manual, automatic, c='blue')
        
        #Three colors to separate between different cases
        else:
            ax.scatter(manual, automatic, c=col, cmap=ListedColormap(['black', 'green', 'blue']))
    
        ax.axhline(thresh, color='r')
        ax.axvline(thresh, color='r')
        ax.set_xlabel('Manual')
        ax.set_ylabel('Automatic')
        ax.set_title(title)
        fig.savefig(os.path.join(self.png_path, 'corr-plots', png_name+'.png'), bbox_inches='tight')


    def save_bland_altman(self, manual: pd.Series, ai: pd.Series, region: str, png_name: str):
        """Creates Bland-Altman plot and saves the figure in a folder

        Args:
            manual (pd.Series): original measurements
            ai (pd.Series): predicted measurements
            region (str): type of measurement
            png_name (str): name of the saved PNG file
        """    

        fig, axes = plt.subplots()

        sm.graphics.mean_diff_plot(ai, manual, ax = axes)
        plt.xlabel('Mean {}'.format(region))
        plt.ylabel('{} difference (ai - manual)'.format(region))
        plt.title('{} manual vs automatic'.format(region))
        plt.savefig(os.path.join(self.png_path, 'bland-altman', png_name+'.png'), bbox_inches='tight')



    def calculate_icc(self, manual: pd.Series, ai: pd.Series):
        """Get intra class correlation for automatic and predicted measurements

        Args:
            manual (pd.Series): measuremnts for predictions
            ai (pd.Series): measurements for originals 

        Returns:
            float: ICC(2,1)
        """    

        raters1 = ['Automatic' for _ in range(len(ai))]
        raters2 = ['Manual' for _ in range(len(ai))]
        raters1.extend(raters2)

        exam1 = list(np.arange(0, len(ai)))
        exam2 = list(np.arange(0, len(ai)))
        exam1.extend(exam2)

        all_values = pd.concat([ai, manual])

        icc_df = pd.DataFrame({'exam': exam1, 'raters': raters1, 'all_values': all_values})

        icc = pg.intraclass_corr(icc_df, 'exam', 'raters', 'all_values')

        icc2_1 = icc[icc['Type'] == 'ICC2']['ICC'].values[0]

        return icc2_1


    def find_outliers(self, manual: pd.Series, ai: pd.Series, score: str) -> Tuple[list, dict]:
        """Find automated measurements that are outliers according to a specific method

        Args:
            manual (pd.Series):  measurements for orginials
            ai (pd.Series):  measurements for predictions
            score (str): either 'zscore' or 'tukey'

        Raises:
            ValueError: when the score type is not one of the two mentioned cases

        Returns:
            list, dictionary: list with outlier index and dictionary with the data and the automated and manual values
        """    

        #Find outliers based on difference between automatic and predicted
        differences = ai - manual

        if score == 'zscore':
            #Z-score
            z_scores = stats.zscore(differences)
            z_score_threshold = 3.0
            outliers = np.where(np.abs(z_scores) > z_score_threshold)[0]

        elif score == 'tukey':
            #Tukey's fences
            q1 = np.percentile(differences, 25)
            q3 = np.percentile(differences, 75)
            iqr = q3 - q1
            lower_fence = q1 - (1.5 * iqr)
            upper_fence = q3 + (1.5 * iqr)
            outliers = np.where((differences < lower_fence) | (differences > upper_fence))[0]

        else:
            raise ValueError('Wrong type. Please, choose either "zscore" or "tukey"')

        outliers_data = []

        #Get outlier info
        for outlier in outliers:

            value = ai.index[outlier]    
            outlier_pullback = self.measures_file.iloc[value]['pullback']
            outlier_frame = self.measures_file.iloc[value]['frame']
            outliers_data.append('{}_frame{}'.format(outlier_pullback, outlier_frame))

        return outliers, outliers_data


    def get_stats_file(self):
        """Obtain an Excel file with all the statistics results for a specific model, and save correlation and bland altman plots as png
        """        

        types = {'Depth': -1, 'Arc': 180, 'Thickness': 500, 'FCT': 65, 'Lipid arc': 90}
        analysis_region = pd.DataFrame(columns = ['Type', 'Model', 'FP', 'FN', 'Mean diff', 'SD', 'Correlation', 'ICC(2,1)', 'Nº outliers', 'Outliers'])

        for measure_type in types.keys():

            print('Getting {} data...'.format(measure_type))

            #Get filtered data and FP and FN
            manual, ai, fp, fn = self.get_data_filtered(self.measures_file, '{} test set'.format(measure_type), '{} {}'.format(measure_type, self.model_id))

            #Mean diff, standard deviation, correlation, icc(2,1) and outliers
            md, sd = self.mean_sd(manual, ai)
            corr = self.corr(manual, ai)
            icc = self.calculate_icc(manual, ai)
            _, outliers = self.find_outliers(manual, ai, 'tukey')

            #Save scatter plots and bland altman for every region 
            print('Saving plots')
            self.scatter_data_save_png(manual, ai, types[measure_type], '{} {}'.format(measure_type, self.model_id), '{}_{}'.format(measure_type.lower(), self.model_id.replace(' ','_')))
            self.save_bland_altman(manual, ai, measure_type, '{}_{}'.format(measure_type.lower(), self.model_id))

            #Append all values into a list
            analysis_list = []
            analysis_list.append(measure_type)
            analysis_list.append(self.model_id)
            analysis_list.append(fp)
            analysis_list.append(fn)
            analysis_list.append(md)
            analysis_list.append(sd)
            analysis_list.append(corr)
            analysis_list.append(icc)
            analysis_list.append(len(outliers))
            analysis_list.append(outliers)

            #Save the list into the dataframe
            analysis_region = analysis_region.append(pd.Series(analysis_list, index=analysis_region.columns[:len(analysis_list)]), ignore_index=True)

            print('\n')

        analysis_region.to_excel('Z:/grodriguez/CardiacOCT/info-files/statistics/second_split/{}.xlsx'.format(self.output_filename))
        print('Done! You can find the saved images in ', self.png_path)

        
def main(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, )
    parser.add_argument('--output_filename', type=str)
    parser.add_argument('--measures_file', type=str, default='Z:/grodriguez/CardiacOCT/info-files/statistics/second_split/measures_v2.xlsx')
    parser.add_argument('--png_path')
    args, _ = parser.parse_known_args(argv)

    args = parser.parse_args()

    stats = Statistics(args.model_id, args.output_filename, args.measures_file, args.png_path)
    stats.get_stats_file()

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)

