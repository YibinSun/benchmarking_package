from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier, HoeffdingTreeRegressor
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor, StreamingRandomPatchesClassifier
from pandas import DataFrame
from datetime import datetime
from skmf_experiment_runner import skmf_experiment
from skmultiflow.anomaly_detection import HalfSpaceTrees
from sklearn.metrics import roc_auc_score

import os
import pandas as pd
import numpy as np

DT_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MAX_INSTANCES = 100000000
REPETITIONS = 10  # Adjust as needed

RESULT_PATH = f'./repetitions={REPETITIONS}/'
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

def _create_arguments_for_skmf(
        task_type,
        datasets_path,
        skmf_learners,
        skmf_learners_arguments,
        MAX_INSTANCES,
        window_size,
        repetitions,
        show_progress=False,
):
    argument_list = []
    for dataset_path in datasets_path:
        for learner in skmf_learners:
            for learner_arguments in skmf_learners_arguments[learner.__name__]:
                arguments = {
                    'task_type': task_type,
                    'dataset_name': dataset_path.split('/')[-1].split('.')[0],
                    'learner_name': learner.__name__,
                    'learner': learner,
                    'stream_path_csv': dataset_path,
                    'hyperparameters': learner_arguments,
                    'max_instances': MAX_INSTANCES,
                    'window_size': window_size,
                    'repetitions': repetitions,
                    'show_pregress': show_progress,
                }
                argument_list.append(arguments)
    return argument_list

def _skmf_classification_benchmark():
    # Define the datasets
    datasets_path = [
        "./benchmarking_datasets/classification/RTG_2abrupt.csv",
        "./benchmarking_datasets/classification/Hyper100k.csv",
        "./benchmarking_datasets/classification/RBF100k.csv",
        "./benchmarking_datasets/classification/Airlines.csv",
        "./benchmarking_datasets/classification/CoverTypeNorm.csv",
        "./benchmarking_datasets/classification/ElecNormNew.csv",
    ]

    # Define the learners
    skmf_learners = [
        NaiveBayes,
        HoeffdingTreeClassifier,
        ExtremelyFastDecisionTreeClassifier,
        AdaptiveRandomForestClassifier,
        StreamingRandomPatchesClassifier,
        SRPClassifier,
        KNNClassifier,
    ]

    # Define the arguments
    skmf_learners_arguments = {
        'NaiveBayes': [
            {},
        ],
        'ExtremelyFastDecisionTreeClassifier': [
            {'grace_period': 200, 'split_confidence': 1e-7}
        ],
        'HoeffdingTreeClassifier': [
            {'grace_period': 200, 'split_confidence': 1e-7}
        ],
        'KNNClassifier': [
            {'n_neighbors': 5},
        ],
       'AdaptiveRandomForestClassifier': [
            {'n_estimators': 5},
            {'n_estimators': 10},
            {'n_estimators': 30},
            {'n_estimators': 100},
       ],
       'StreamingRandomPatchesClassifier': [
            {'n_estimators': 5},
            {'n_estimators': 10},
            {'n_estimators': 30},
            {'n_estimators': 100},
       ]
    }

    df_results = DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_skmf(
            task_type='classification',
            datasets_path=datasets_path,
            skmf_learners=skmf_learners,
            skmf_learners_arguments=skmf_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True,
    ):
        results, raw_results = skmf_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}/full_skmf_classification_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_skmf_classification_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_skmf_classification_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_skmf_classification_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)

def _skmf_regression_benchmark():
    # Define the datasets
    datasets_path = [
        "./benchmarking_datasets/regression/bike.csv",
        "./benchmarking_datasets/regression/elevators.csv",
        "./benchmarking_datasets/regression/FriedmanLea.csv",
        "./benchmarking_datasets/regression/hyperA.csv",
        "./benchmarking_datasets/regression/MetroTraffic.csv",
        "./benchmarking_datasets/regression/superconductivity.csv",
    ]

    # Define the learners
    skmf_learners = [
        KNNRegressor,
        HoeffdingTreeRegressor,
        AdaptiveRandomForestRegressor
    ]

    # Define the arguments
    skmf_learners_arguments = {
        'KNNRegressor': [
            {'n_neighbors': 10},
        ],
        'HoeffdingTreeRegressor': [
            {}
        ],
        'AdaptiveRandomForestRegressor': [
            {'n_estimators': 5},
            {'n_estimators': 10},
            {'n_estimators': 30},
            {'n_estimators': 100},
        ]
    }

    df_results = DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_skmf(
            task_type='regression',
            datasets_path=datasets_path,
            skmf_learners=skmf_learners,
            skmf_learners_arguments=skmf_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True
    ):
        results, raw_results = skmf_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_skmf_regression_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_skmf_regression_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_skmf_regression_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_skmf_regression_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)


def _skmf_anomaly_benchmark():
    datasets_path = [
        "./benchmarking_datasets/anomaly/Donors.csv",
        "./benchmarking_datasets/anomaly/Http.csv",
        "./benchmarking_datasets/anomaly/Insects.csv",
        "./benchmarking_datasets/anomaly/Mulcross.csv",
        "./benchmarking_datasets/anomaly/Shuttle.csv",
        "./benchmarking_datasets/anomaly/Smtp.csv",
    ]

    # Define anomaly learners
    skmf_learners = [
        HalfSpaceTrees,
    ]

    # Define hyperparameters for each learner
    skmf_learners_arguments = {
        'HalfSpaceTrees': [
            {'n_estimators': 25, 'window_size': 250, 'depth': 15},
        ],
    }

    df_results = DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_skmf(
            task_type='anomaly_detection',
            datasets_path=datasets_path,
            skmf_learners=skmf_learners,
            skmf_learners_arguments=skmf_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True,
    ):
        results, raw_results = skmf_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_skmf_anomaly_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_skmf_anomaly_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_skmf_anomaly_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_skmf_anomaly_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)

if __name__ == "__main__":
    _skmf_classification_benchmark()
    print("Finished scikit-multiflow classification benchmark")
    _skmf_regression_benchmark()
    print("Finished scikit-multiflow regression benchmark")
    _skmf_anomaly_benchmark()
    print("Finished scikit-multiflow anomaly benchmark")
    exit()
