from capymoa.classifier import (
    KNN, HoeffdingTree, EFDT,
    NaiveBayes as NB,
    AdaptiveRandomForestClassifier as ARF,
    StreamingRandomPatches as SRP,
)
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier
from river.neighbors import KNNClassifier
from river.forest import ARFClassifier
from river.ensemble import SRPClassifier

from capymoa.regressor import (
    KNNRegressor,
    PassiveAggressiveRegressor,
    AdaptiveRandomForestRegressor,
)
from river.neighbors import KNNRegressor as river_KNNRegressor
from river.linear_model import PARegressor as river_PARegressor
from river.forest import ARFRegressor as river_ARFRegressor
from river.tree import HoeffdingTreeRegressor as river_HoeffdingTreeRegressor
from river.anomaly import HalfSpaceTrees

from capymoa.stream import stream_from_file
from benchmark_river import river_experiment

import pandas as pd
from datetime import datetime
import os


DT_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MAX_INSTANCES = 100000000
REPETITIONS = 10 # 30 # 100

RESULT_PATH = f'./repetitions={REPETITIONS}/'
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

def _create_arguments_for_capymoa(
        datasets_path,
        capymoa_learners,
        capymoa_learners_arguments,
        MAX_INSTANCES,
        window_size,
        repetitions,
        show_progress=True,
):
    argument_list = []
    for dataset_path in datasets_path:
        for learner in capymoa_learners:
            for learner_arguments in capymoa_learners_arguments[learner.__name__]:
                arguments = {
                    'dataset_name': dataset_path.split('/')[-1].split('.')[0],
                    'learner_name': learner.__name__,
                    'learner': learner,
                    'stream': stream_from_file(dataset_path),
                    'hyperparameters': learner_arguments,
                    'max_instances': MAX_INSTANCES,
                    'window_size': window_size,
                    'repetitions': repetitions,
                    'show_pregress': show_progress,
                }
                argument_list.append(arguments)

    return argument_list


def _create_arguments_for_river(
        task_type,
        datasets_path,
        river_learners,
        river_learners_arguments,
        MAX_INSTANCES,
        window_size,
        repetitions,
        show_progress=True,
):
    argument_list = []
    for dataset_path in datasets_path:
        for learner in river_learners:
            for learner_arguments in river_learners_arguments[learner.__name__]:
                arguments = {
                    'task_type': task_type,  # 'classification', 'regression', 'anomaly_detection
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


def _river_classification_benchmark():
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
    river_learners = [
        GaussianNB,
        HoeffdingTreeClassifier,
        ExtremelyFastDecisionTreeClassifier,
        ARFClassifier,
       SRPClassifier,
        KNNClassifier,
    ]

    # Define the arguments
    river_learners_arguments = {
        'GaussianNB': [
            {},
        ],
        'ExtremelyFastDecisionTreeClassifier': [
            {'grace_period': 200, 'delta': 1e-7},
        ],
        'HoeffdingTreeClassifier': [
            {'grace_period': 200, 'delta': 1e-7}
        ],
        'KNNClassifier': [
            {'n_neighbors': 5},
            {}
        ],
        'ARFClassifier': [
            {'n_models': 5},
            {'n_models': 10},
            {'n_models': 30},
            {'n_models': 100},
        ],
          'SRPClassifier': [
            {'n_models': 5},
            {'n_models': 10},
            {'n_models': 30},
            {'n_models': 100},
       ]
    }

    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_river(
            task_type='classification',
            datasets_path=datasets_path,
            river_learners=river_learners,
            river_learners_arguments=river_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True,
    ):
        results, raw_results = river_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}/full_river_classification_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_river_classification_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_river_classification_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_river_classification_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)

def _river_regression_benchmark():
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
    river_learners = [
        river_KNNRegressor,
        river_HoeffdingTreeRegressor,
        river_ARFRegressor
    ]

    # Define the arguments
    river_learners_arguments = {
        'KNNRegressor': [
            {'n_neighbors': 10}
        ],
        'HoeffdingTreeRegressor':
        [
            {'grace_period': 200, 'tau': 0.05}    
        ],
        'ARFRegressor': [
            {'n_models': 5},
            {'n_models': 10},
            {'n_models': 30},
            {'n_models': 100},
        ]
    }


    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_river(
        task_type='regression',
        datasets_path=datasets_path,
        river_learners=river_learners,
        river_learners_arguments=river_learners_arguments,
        MAX_INSTANCES=MAX_INSTANCES,
        window_size=1000,
        repetitions=REPETITIONS,
        show_progress=True
    ):
        results, raw_results = river_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_river_regression_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_river_regression_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_river_regression_raw/')
        df.to_csv(f'{RESULT_PATH}benchmark_river_regression_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv', index=False)


def _river_anomaly_detection_benchmark():
    # Define the datasets
    datasets_path = [
        "./benchmarking_datasets/anomaly/Donors.csv",
        "./benchmarking_datasets/anomaly/Http.csv",
        "./benchmarking_datasets/anomaly/Insects.csv",
        "./benchmarking_datasets/anomaly/Mulcross.csv",
        "./benchmarking_datasets/anomaly/Shuttle.csv",
        "./benchmarking_datasets/anomaly/Smtp.csv",
    ]

    # Define the learners
    river_learners = [
        HalfSpaceTrees,
    ]

    # Define the arguments
    river_learners_arguments = {
        'HalfSpaceTrees': [
            {'n_trees': 25, 'height': 15, 'window_size': 250},
        ],
    }

    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_river(
        task_type='anomaly_detection',
        datasets_path=datasets_path,
        river_learners=river_learners,
        river_learners_arguments=river_learners_arguments,
        MAX_INSTANCES=MAX_INSTANCES,
        window_size=1000,
        repetitions=REPETITIONS,
        show_progress=True
    ):
        results, raw_results = river_experiment(**argus)

        # Extend column coverage
        all_columns = df_results.columns.tolist() + [col for col in results.columns if col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_river_anomaly_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_river_anomaly_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_river_anomaly_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_river_anomaly_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)


if __name__ == "__main__":
    _capymoa_classification_benchmark()
    print("Finished capymoa classification benchmark")
    _river_classification_benchmark()
    print("Finished river classification benchmark")
    _river_anomaly_detection_benchmark()
    print("Finished river anomaly benchmark")
    exit()
