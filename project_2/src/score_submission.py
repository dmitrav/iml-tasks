import pandas, numpy
import sklearn.metrics as metrics
from project_2.src.constants import subtask_1_labels, subtask_2_labels, subtask_3_labels
from project_2.src.constants import version

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def get_score(df_true, df_submission):

    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    print(task1, task2, task3)
    return score


def example():

    filename = 'sample.zip'
    df_submission = pd.read_csv(filename)

    # generate a baseline based on sample.zip
    df_true = pd.read_csv(filename)
    for label in TESTS + ['LABEL_Sepsis']:
        # round classification labels
        df_true[label] = np.around(df_true[label].values)

    print('Score of sample.zip with itself as groundtruth', get_score(df_true, df_submission))


def compose_submission():

    path_to_predictions = "/Users/andreidm/ETH/courses/iml-tasks/project_2/res/test/"
    path_to_test_features = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test_features_imputed_engineered_v.0.1.0.csv"

    test_pid = pandas.read_csv(path_to_test_features)['pid']

    all_labels = [*subtask_1_labels, *subtask_2_labels, *subtask_3_labels]

    submission = pandas.DataFrame({'pid': test_pid})

    for label in all_labels:

        path = path_to_predictions + "predictions_" + label + "_v.0.1.0.csv"
        label_prediction = pandas.read_csv(path)

        if label in subtask_1_labels or label in subtask_2_labels:
            submission[label] = label_prediction.iloc[:, 3]
        else:
            submission[label] = label_prediction.iloc[:, 2]

    submission.to_csv(path_to_predictions + 'prediction_' + version + '.zip', index=False, float_format='%.3f',
                      compression='zip')


if __name__ == "__main__":

    compose_submission()