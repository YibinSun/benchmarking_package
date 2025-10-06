from benchmark_moa import moa_experiment
import pandas as pd
from datetime import datetime
import os

DT_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MAX_INSTANCES = 100000000
REPETITIONS = 10  # 30 # 100

RESULT_PATH = f'./repetitions={REPETITIONS}/'

_randomizable = {
    'kNN': False,
    'NaiveBayes': False,
    'HoeffdingTree': False,
    'EFDT': False,
    'AdaptiveRandomForest': True,
    'StreamingRandomPatches': True,
    'PassiveAggressiveRegressor': True,
    'AdaptiveRandomForestRegressor': True,
    'HSTrees': True,
}


if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


def _create_arguments_for_moa(
        datasets_path,
        moa_learners_arguments,
        MAX_INSTANCES,
        window_size,
        repetitions,
        task_type,
        show_progress=True,
):
    argument_list = []
    for dataset_path in datasets_path:
        for learner in moa_learners_arguments.keys():
            for arguments in moa_learners_arguments[learner]:

                learner_arg = learner

                if len(arguments) > 0:

                    for k, v in arguments.items():
                        learner_arg += ' -' + str(k) + ' ' + str(v) + ' '
                    learner_arg.strip()
                if task_type == 'classification':
                    command_line = f'java -Xmx16g -Xms50m -Xss1g -jar /home/spencer/Documents/why/benchmark/moa.jar "EvaluateInterleavedTestThenTrain -e (BasicClassificationPerformanceEvaluator -o) -i {MAX_INSTANCES} -f 1000000000 -q 1000000000 -s (ArffFileStream -f {dataset_path}) -l ({learner_arg})'
                elif task_type == 'regression':
                    if _randomizable[learner.split('.')[-1]]:
                        command_line = f'java -Xmx16g -Xms50m -Xss1g -jar ./moa.jar "EvaluatePrequentialRegression -e BasicRegressionPerformanceEvaluator -i {MAX_INSTANCES} -f 1000000000 -q 1000000000 -s (ArffFileStream -f {dataset_path}) -l (meta.AdaptiveRandomForestRegressor -l (ARFFIMTDD -k 5 -s VarianceReductionSplitCriterion -g 200 -c 0.0000001) -s {arguments["s"]} -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))'
                    else:
                        command_line = f'java -Xmx16g -Xms50m -Xss1g -jar ./moa.jar "EvaluatePrequentialRegression -e BasicRegressionPerformanceEvaluator -i {MAX_INSTANCES} -f 1000000000 -q 1000000000 -s (ArffFileStream -f {dataset_path}) -l ({learner_arg})'

                elif task_type == 'anomaly':
                    command_line = f'java -Xmx16g -Xms50m -Xss1g -jar ./moa.jar "EvaluateInterleavedTestThenTrain -e (BasicAUCImbalancedPerformanceEvaluator -a) -i {MAX_INSTANCES} -f 1000000000 -q 1000000000 -s (ArffFileStream -f {dataset_path}) -l ({learner_arg})'


                if _randomizable[learner.split('.')[-1]]:
                    if task_type == 'regression':
                        command_line = command_line[:-1] +' -r seed_value' + command_line[-1:]+'"'
                    else:
                        command_line += f' -r seed_value"'
                else:
                    command_line += '"'


                arguments = {
                        'dataset_name': dataset_path.split('/')[-1].split('.')[0],
                        'learner_name': learner.split('.')[-1],
                        'command_line': command_line,
                        'repetitions': repetitions,
                        'show_progress': show_progress,
                }
                argument_list.append(arguments)

    return argument_list

def _moa_classification_benchmark():
    # Define the datasets
    datasets_path = [
        "./benchmarking_datasets/classification/RTG_2abrupt.arff",
        "./benchmarking_datasets/classification/Hyper100k.arff",
        "./benchmarking_datasets/classification/RBF100k.arff",
        "./benchmarking_datasets/classification/Airlines.arff",
        "./benchmarking_datasets/classification/CoverTypeNorm.arff",
        "./benchmarking_datasets/classification/ElecNormNew.arff",
    ]

    # Define the arguments
    moa_learners_arguments = {
        'bayes.NaiveBayes': [
            {},
        ],
        'trees.EFDT': [
            # {'g': 50, 'c': 0.01},
        #     {'g': 200, 'c': 1e-7}
            {}
        ],
        'trees.HoeffdingTree': [
            # {'g': 50, 'c': 0.01},
        #     {'g': 200, 'c': 1e-7}
            {}
        ],
        'lazy.kNN': [
            # {'k': 3},
        #     {'k': 11}
            {}
        ],
        'meta.AdaptiveRandomForest': [
            {'s': 5},
            {'s': 10},
            {'s': 30},
            {'s': 100},
            # {'ensemble_size': 100, 'number_of_jobs':8},
            # {'ensemble_size': 100, 'number_of_jobs':8, 'minibatch_size':25},
        ],
        'meta.StreamingRandomPatches': [
            {'s': 5},
            {'s': 10},
            {'s': 30},
            {'s': 100},
        ]
    }

    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_moa(
            task_type='classification',
            datasets_path=datasets_path,
            moa_learners_arguments=moa_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True
    ):
        results, raw_results = moa_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns
                                                     if col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_moa_classification_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_moa_classification_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_moa_classification_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_moa_classification_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)


def _moa_regression_benchmark():
    # Define the datasets
    datasets_path = [
        "./benchmarking_datasets/regression/bike.arff",
        "./benchmarking_datasets/regression/elevators.arff",
        "./benchmarking_datasets/regression/FriedmanLea.arff",
        "./benchmarking_datasets/regression/hyperA.arff",
        "./benchmarking_datasets/regression/MetroTraffic.arff",
        "./benchmarking_datasets/regression/superconductivity.arff",
    ]

    # Define the arguments
    moa_learners_arguments = {
        'lazy.kNN': [
            {'k': 10},
        ],
        'trees.FIMTDD': [
            {}
        ],
        'meta.AdaptiveRandomForestRegressor': [
            {'s': 5},
            {'s': 10},
            {'s': 30},
            {'s': 100},
        ]
    }

    df_results = pd.DataFrame()
    df_raw_results = []
    for argus in _create_arguments_for_moa(
            task_type='regression',
            datasets_path=datasets_path,
            moa_learners_arguments=moa_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True
    ):
        results, raw_results = moa_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns if
                                                     col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_moa_regression_results_{DT_stamp}.csv', index=False)
    for df in df_raw_results:
        if not os.path.exists(f'{RESULT_PATH}benchmark_moa_regression_raw/'):
            os.mkdir(f'{RESULT_PATH}benchmark_moa_regression_raw/')
        df.to_csv(
            f'{RESULT_PATH}benchmark_moa_regression_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)

def _moa_anomaly_benchmark():
    datasets_path = [
        "./benchmarking_datasets/anomaly/Donors.arff",
        "./benchmarking_datasets/anomaly/Http.arff",
        "./benchmarking_datasets/anomaly/Insects.arff",
        "./benchmarking_datasets/anomaly/Mulcross.arff",
        "./benchmarking_datasets/anomaly/Shuttle.arff",
        "./benchmarking_datasets/anomaly/Smtp.arff",
    ]

    moa_learners_arguments = {
        'oneclass.HSTrees': [
            {'t': 25, 'h': 15, 'p': 250},
        ],
    }

    df_results = pd.DataFrame()
    df_raw_results = []

    for argus in _create_arguments_for_moa(
            task_type='anomaly',
            datasets_path=datasets_path,
            moa_learners_arguments=moa_learners_arguments,
            MAX_INSTANCES=MAX_INSTANCES,
            window_size=1000,
            repetitions=REPETITIONS,
            show_progress=True
    ):
        results, raw_results = moa_experiment(**argus)

        all_columns = df_results.columns.tolist() + [col for col in results.columns
                                                     if col not in df_results.columns.tolist()]
        df_results = df_results.reindex(columns=all_columns, fill_value='N/A')
        results = results.reindex(columns=all_columns, fill_value='N/A')
        df_results = pd.concat([df_results, results], ignore_index=True)

        df_raw_results.append(raw_results)

    df_results.to_csv(f'{RESULT_PATH}full_moa_anomaly_results_{DT_stamp}.csv', index=False)
    if not os.path.exists(f'{RESULT_PATH}benchmark_moa_anomaly_raw/'):
        os.mkdir(f'{RESULT_PATH}benchmark_moa_anomaly_raw/')
    for df in df_raw_results:
        df.to_csv(
            f'{RESULT_PATH}benchmark_moa_anomaly_raw/{str(df.iloc[0, 2])}_{str(df.iloc[0, 3])}_{str(df.iloc[0, 4])}_{DT_stamp}.csv',
            index=False)


if __name__ == "__main__":
    _moa_classification_benchmark()
    print("Finished moa classification benchmark")
    _moa_regression_benchmark()
    print("Finished moa regression benchmark")
    _moa_anomaly_benchmark()
    print("Finished moa anomaly benchmark")
    exit()
