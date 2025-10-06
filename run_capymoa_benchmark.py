from capymoa.classifier import (
    KNN, HoeffdingTree, EFDT,
    NaiveBayes as NB,
    AdaptiveRandomForestClassifier as ARF,
    StreamingRandomPatches as SRP,
)

from capymoa.regressor import (
    KNNRegressor,
    PassiveAggressiveRegressor,
    AdaptiveRandomForestRegressor,
)
from capymoa.anomaly import HalfSpaceTrees

from capymoa.stream import stream_from_file
from benchmark_capymoa import capymoa_experiment
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

def _capymoa_classification_benchmark():
# Define the datasets
    datasets_path = [
        "./benchmarking_datasets/classification/RTG_2abrupt.arff",
        "./benchmarking_datasets/classification/Hyper100k.arff",
        "./benchmarking_datasets/classification/RBF100k.arff",
        "./benchmarking_datasets/classification/Airlines.arff",
        "./benchmarking_datasets/classification/CoverTypeNorm.arff",
        "./benchmarking_datasets/classification/ElecNormNew.arff",
    ]

    # Define the learners
    capymoa_learners = [
        NB,
        HoeffdingTree,
        EFDT,
        ARF,
        SRP,
        KNN,
    ]

    # Define the arguments
    capymoa_learners_arguments = {
        'NaiveBayes': [
           {},
        ],
        'EFDT': [
            {'min_samples_reevaluate': 2000,'confidence': 1e-7}
        ],
        'HoeffdingTree': [
            {'confidence': 1e-7}
        ],
        'KNN': [
            {'k': 5},
        ],
        'AdaptiveRandomForestClassifier': [
           {'ensemble_size': 5},
            {'ensemble_size': 10},
            {'ensemble_size': 30},
            {'ensemble_size': 100},
            # {'ensemble_size': 100, 'number_of_jobs':8},
            # {'ensemble_size': 100, 'number_of_jobs':8, 'minibatch_size':25},
        ],
        'StreamingRandomPatches': [
           {'ensemble_size': 5},
            {'ensemble_size': 10},
            {'ensemble_size': 30},
            {'ensemble_size': 100},
        ]
    }


    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_capymoa(
        datasets_path=datasets_path,
        capymoa_learners=capymoa_learners,
        capymoa_learners_arguments=capymoa_learners_arguments,
        MAX_INSTANCES=MAX_INSTANCES,
        window_size=1000,
        repetitions=REPETITIONS,
        show_progress=True
    ):
        results, raw_results = capymoa_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns
                                                     if col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_capymoa_classification_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_capymoa_classification_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_capymoa_classification_raw/')
        df.to_csv(f'{RESULT_PATH}benchmark_capymoa_classification_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv', index=False)


def _capymoa_regression_benchmark():
# Define the datasets
    datasets_path = [
        "./benchmarking_datasets/regression/bike.arff",
        "./benchmarking_datasets/regression/elevators.arff",
        "./benchmarking_datasets/regression/FriedmanLea.arff",
        "./benchmarking_datasets/regression/hyperA.arff",
        "./benchmarking_datasets/regression/MetroTraffic.arff",
        "./benchmarking_datasets/regression/superconductivity.arff",
    ]

    # Define the learners
    capymoa_learners = [
        KNNRegressor,
        FIMTDD,
        AdaptiveRandomForestRegressor,
    ]

    # Define the arguments
    capymoa_learners_arguments = {
        'KNNRegressor': [
            {'k': 10},
            # {'k': 11}
        ],
        'FIMTDD': [
            {}
        ],
        'AdaptiveRandomForestRegressor': [
            {'ensemble_size': 5},
            {'ensemble_size': 10},
            {'ensemble_size': 30},
            {'ensemble_size': 100},
        ]
    }


    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_capymoa(
        datasets_path=datasets_path,
        capymoa_learners=capymoa_learners,
        capymoa_learners_arguments=capymoa_learners_arguments,
        MAX_INSTANCES=MAX_INSTANCES,
        window_size=1000,
        repetitions=REPETITIONS,
        show_progress=True
    ):
        results, raw_results = capymoa_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_capymoa_regression_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_capymoa_regression_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_capymoa_regression_raw/')
        df.to_csv(f'{RESULT_PATH}benchmark_capymoa_regression_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv', index=False)


def _capymoa_anomaly_benchmark():
    # Define the datasets
    datasets_path = [
        "./benchmarking_datasets/anomaly/Donors.arff",
        "./benchmarking_datasets/anomaly/Http.arff",
        "./benchmarking_datasets/anomaly/Insects.arff",
        "./benchmarking_datasets/anomaly/Mulcross.arff",
        "./benchmarking_datasets/anomaly/Shuttle.arff",
        "./benchmarking_datasets/anomaly/Smtp.arff",
    ]

    # Define the anomaly detection learners
    capymoa_learners = [
        HalfSpaceTrees,
    ]

    # Define the hyperparameter grid
    capymoa_learners_arguments = {
        'HalfSpaceTrees': [
            {'number_of_trees': 25, 'max_depth': 15, 'window_size': 250},
        ],
    }

    df_results = pd.DataFrame()
    df_raw_results = []

    for argus in _create_arguments_for_capymoa(
        datasets_path=datasets_path,
        capymoa_learners=capymoa_learners,
        capymoa_learners_arguments=capymoa_learners_arguments,
        MAX_INSTANCES=MAX_INSTANCES,
        window_size=1000,
        repetitions=REPETITIONS,
        show_progress=True,
    ):
        results, raw_results = capymoa_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_capymoa_anomaly_results_{DT_stamp}.csv', index=False)
    if not os.path.exists(f'{RESULT_PATH}benchmark_capymoa_anomaly_raw/'):
        os.mkdir(f'{RESULT_PATH}benchmark_capymoa_anomaly_raw/')
    for df in df_raw_results:
        df.to_csv(
            f'{RESULT_PATH}benchmark_capymoa_anomaly_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)


if __name__ == "__main__":
    _capymoa_classification_benchmark()
    print("Finished capymoa classification benchmark")
    _capymoa_regression_benchmark()
    print("Finished capymoa regression benchmark")
    _capymoa_anomaly_benchmark()
    print("Finished capymoa anomaly detection benchmark")
    exit()
