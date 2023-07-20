import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import pingouin as pg
from scipy import stats
import os
import sys
import math
import warnings
from matplotlib.colors import ListedColormap
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def get_data_filtered(test_sheet, type_manual, type_ai, missing = -99):
    """Obtain lipid/calcium values without FP or FN or nulls

    Args:
        test_sheet (pd.DataFrame): dataframe with all the measurements
        type_manual (string): specific column of manual annotation
        type_ai (_type_): specific column of automatic annotation (must be same type as type_manual)
        missing (int, optional): encoding of missing values. Defaults to -99.

    Returns:
        pd.Series: series for both the manual and predicted measurements
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

    list_drops.extend(list_fp)
    list_drops.extend(list_fn)
    list_drops.extend(list_nulls)

    #Drop those cases
    ai.drop(list_drops, inplace=True)
    manual.drop(list_drops, inplace=True)

    return manual, ai, fp_count, fn_count


def mean_sd(manual, automatic):

    diff = automatic - manual                  
    md = np.mean(diff)                   
    sd = np.std(diff, axis=0) 
    
    return  md, sd


def corr_plot(manual, automatic):
    """Save correlation plot

    Args:
        manual (pd.Series): measurements for orginials
        automatic (pd.Series): measurements for predictions
        thresh (int): critical values for lipid arc, FCT etc (if there are none, just put 0)
        title (string): Title of plot

    Returns:
        float: Correlation of manual and automatic measurements
    """    

    return np.corrcoef(manual, automatic)[0][1]


def scatter_data_save_png(manual, automatic, thresh, title, png_name):

    col = []
    for i, j in zip(manual, automatic):
        if i <= thresh and j >= thresh:
            col.append(0)

        elif i >= thresh and j <= thresh:
            col.append(1)

        else:
            col.append(2)

    fig, ax = plt.subplots()
    if all(c == 2 for c in col):
        ax.scatter(manual, automatic, c='blue')

    else:
        ax.scatter(manual, automatic, c=col, cmap=ListedColormap(['black', 'green', 'blue']))
   
    ax.axhline(thresh, color='r')
    ax.axvline(thresh, color='r')
    ax.set_xlabel('Manual')
    ax.set_ylabel('Automatic')
    ax.set_title(title)
    fig.savefig("Z:/grodriguez/CardiacOCT/info-files/statistics/corr-plots/{}.png".format(png_name), bbox_inches='tight')


def save_bland_altman(manual, ai, region, png_name):


    fig, axes = plt.subplots()

    sm.graphics.mean_diff_plot(ai, manual, ax = axes)
    plt.xlabel('Mean {}'.format(region))
    plt.ylabel('{} difference (ai - manual)'.format(region))
    plt.title('{} manual vs automatic'.format(region))
    plt.savefig("Z:/grodriguez/CardiacOCT/info-files/statistics/bland-altman/{}.png".format(png_name), bbox_inches='tight')



def calculate_icc(manual_values, automatic_values):
    """Get intra class correlation for automatic and predicted measurements

    Args:
        manual_values (pd.Series): measuremtns for predictions
        automatic_values (pd.Series): measurements for originals 

    Returns:
        float: ICC(2,1)
    """    

    raters1 = ['Automatic' for _ in range(len(automatic_values))]
    raters2 = ['Manual' for _ in range(len(automatic_values))]
    raters1.extend(raters2)

    exam1 = list(np.arange(0, len(automatic_values)))
    exam2 = list(np.arange(0, len(automatic_values)))
    exam1.extend(exam2)

    all_values = pd.concat([automatic_values, manual_values])

    icc_df = pd.DataFrame({'exam': exam1, 'raters': raters1, 'all_values': all_values})

    icc = pg.intraclass_corr(icc_df, 'exam', 'raters', 'all_values')

    icc2_1 = icc[icc['Type'] == 'ICC2']['ICC'].values[0]

    return icc2_1


def find_outliers(manual_values, automatic_values, sheet, score):
    """Find automated measurements that are outliers according to a specific method

    Args:
        manual_values (pd.Series):  measurements for orginials
        automatic_values (pd.Series):  measurements for predictions
        sheet (pd.Dataframe): measurements dataframe to get frame and pullback of outlier
        score (string): either 'Z-score' or 'Tukey'

    Raises:
        ValueError: when the score is not one of the two mentioned cases

    Returns:
        list, dictionary: list with outlier index and dictionary with the data and the automated and manual values
    """    

    #Find outliers based on difference between automatic and predicted
    differences = automatic_values - manual_values

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
        raise ValueError('Wrong type')

    outliers_data = []

    #Get outlier info
    for outlier in outliers:

        value = automatic_values.index[outlier]    
        outlier_pullback = sheet.iloc[value]['pullback']
        outlier_frame = sheet.iloc[value]['frame']
        #outlier_ai = automatic_values.tolist()[outlier]
        #outlier_manual = manual_values.tolist()[outlier]

        #outliers_data.append({'pullback': outlier_pullback, 'frame': outlier_frame, 'ai': outlier_ai, 'manual': outlier_manual})
        outliers_data.append('{}_frame{}'.format(outlier_pullback, outlier_frame))

    return outliers, outliers_data


def main(argv):

    excel_name = 'measures_analysis_rgb_final'
    #models = ['model 1', 'model 2', 'model 3', 'model 4', 'model 5', 'model 6', 'model 7', 'model 8', 'model 9']
    models = ['0', '1', '2', '3', '4', 'best', 'last']

    types_cal = ['Depth', 'Arc', 'Thickness']
    thresh_cal = [-1, 180, 500]

    types_lipid = ['FCT', 'Lipid arc']
    thresh_lipid = [65, 90]

    save = False

    measurements_calcium = pd.read_excel(r'Z:\grodriguez\CardiacOCT\info-files\statistics\second_split\model_rgb_script_measures_with_cal.xlsx', sheet_name='Calcium')
    measurements_lipid = pd.read_excel(r'Z:\grodriguez\CardiacOCT\info-files\statistics\second_split\model_rgb_script_measures_with_cal.xlsx', sheet_name='Lipid')

    # #Getting measures in calcium for every model
    for measure_type in types_cal:

        print('Getting {} data'.format(measure_type))

        analysis_region = pd.DataFrame(columns = ['Model', 'FP', 'FN', 'Mean diff', 'SD', 'Correlation', 'ICC(2,1)', 'Nº outliers', 'Outliers'])

        for i in range(len(models)):

            #Get filtered data and FP and FN
            manual, ai, fp, fn = get_data_filtered(measurements_calcium, '{} test set'.format(measure_type), '{} {}'.format(measure_type, models[i]))

            #Mean diff, standard deviation, correlation, icc(2,1) and outliers
            md, sd = mean_sd(manual, ai)
            corr = corr_plot(manual, ai)
            icc = calculate_icc(manual, ai)
            _, outliers = find_outliers(manual, ai, measurements_calcium, 'tukey')

            #Append all values into a list
            analysis_list = []
            analysis_list.append(models[i])
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

        #Create Excel file if it's not created. Otherwise, use a new sheet in the current file to append the new values
        if os.path.exists('Z:/grodriguez/CardiacOCT/info-files/statistics/{}.xlsx'.format(excel_name)) == False:
            analysis_region.to_excel('Z:/grodriguez/CardiacOCT/info-files/statistics/{}.xlsx'.format(excel_name), sheet_name=measure_type)

        else:
            with pd.ExcelWriter('Z:/grodriguez/CardiacOCT/info-files/statistics/{}.xlsx'.format(excel_name), engine='openpyxl', mode='a') as writer:  
                analysis_region.to_excel(writer, sheet_name=measure_type)


    #Same as before but with lipid measurements
    for measure_type in types_lipid:

        print('Getting {} data'.format(measure_type))

        analysis_region = pd.DataFrame(columns = ['Model', 'FP', 'FN', 'Mean diff', 'SD', 'Correlation', 'ICC(2,1)', 'Nº outliers', 'Outliers'])

        for i in range(len(models)):

            #Get filtered data and FP and FN
            manual, ai, fp, fn = get_data_filtered(measurements_lipid, '{} test set'.format(measure_type), '{} {}'.format(measure_type, models[i]))

            #Mean diff, standard deviation, correlation, icc(2,1) and outliers
            md, sd = mean_sd(manual, ai)
            corr = corr_plot(manual, ai)
            icc = calculate_icc(manual, ai)
            _, outliers = find_outliers(manual, ai, measurements_lipid, 'tukey')

            #Append all values into a list
            analysis_list = []
            analysis_list.append(models[i])
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

        #Create Excel file if it's not created. Otherwise, use a new sheet in the current file to append the new values
        if os.path.exists('Z:/grodriguez/CardiacOCT/info-files/statistics/{}.xlsx'.format(excel_name)) == False:
            analysis_region.to_excel('Z:/grodriguez/CardiacOCT/info-files/statistics/{}.xlsx'.format(excel_name), sheet_name=measure_type)

        else:
            with pd.ExcelWriter('Z:/grodriguez/CardiacOCT/info-files/statistics/{}.xlsx'.format(excel_name), engine='openpyxl', mode='a') as writer:  
                analysis_region.to_excel(writer, sheet_name=measure_type)



    if save == True:
        #Save scatter plots and bland altman for every region in calcium and model
        print('Saving plots for calcium')
        for measure_type, thresh in zip(types_cal, thresh_cal):
                
            for i in range(len(models)):
                manual, ai, _, _ = get_data_filtered(measurements_calcium, '{} test set'.format(measure_type), '{} {}'.format(measure_type, models[i]))
                scatter_data_save_png(manual, ai, thresh, '{} {}'.format(measure_type, models[i]), '{}_{}'.format(measure_type.lower(), models[i]))
                save_bland_altman(manual, ai, measure_type, '{}_{}'.format(measure_type.lower(), models[i]))

        
        print('Saving plots for lipid')
        #Save scatter plots and bland altman for every region in calcium and model
        for measure_type, thresh in zip(types_lipid, thresh_lipid):
                
            for i in range(len(models)):
                manual, ai, _, _ = get_data_filtered(measurements_lipid, '{} test set'.format(measure_type), '{} {}'.format(measure_type, models[i]))
                scatter_data_save_png(manual, ai, thresh, '{} {}'.format(measure_type, models[i]), '{}_{}'.format(measure_type.lower(), models[i].replace(' ','_')))
                save_bland_altman(manual, ai, measure_type, '{}_{}'.format(measure_type.lower(), models[i]))
        

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)