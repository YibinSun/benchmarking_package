import pandas as pd
from datetime import datetime
import numpy as np
import inspect
import time, math

# scikit-multiflow imports
from skmultiflow.data import DataStream, FileStream
# from skmultiflow.metrics import ClassificationPerformanceEvaluator, RegressionPerformanceEvaluator
from skmultiflow.evaluation import EvaluatePrequential

# Metric name mappings
_cla_metric_names = ['accuracy', 'kappa', 'recall', 'precision', 'f1']
_reg_metric_names = ['mean_square_error', 'mean_absolute_error']
# _anomaly_metric_names = ['auc']

_capymoa_scikit_name_converter_dict = {
    "accuracy": "accuracy",
    "kappa": "kappa",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "mean_square_error": "mean_square_error",
    "mean_absolute_error": "mean_absolute_error",
    "r2_score": "r2_score",
    "auc": "auc"
}


def _test_then_train_skmf(
        task_type,
        stream_data,
        model,
        max_instances=1000000,
):
    stream = stream_data

    metric_names = (
        _cla_metric_names if task_type == "classification" else
        _reg_metric_names if task_type == "regression" else
        None
    )

    evaluator = EvaluatePrequential(
        n_wait=1,
        max_samples=max_instances,
        pretrain_size=0,
        metrics=metric_names,
        show_plot=False
    )

    evaluator.evaluate(stream=stream, model=model)

    return evaluator


def skmf_experiment(
        task_type,  # 'classification', 'regression'
        dataset_name,
        learner_name,
        stream_path_csv,
        learner,
        hyperparameters={},
        repetitions=1,
        max_instances=1000000,
        show_progress=True,
        **kwargs,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    if show_progress:
        print(f"[{date_time_stamp}][skmf] Executing {learner_name} on {dataset_name}")

    raw_results = []  # Store raw results for each repetition

    metric_names = (
        _cla_metric_names if task_type == "classification" else
        _reg_metric_names if task_type == "regression" else
        None
    )

    repetition = 1
    for _ in range(repetitions):
        if show_progress:
            print(f"[{date_time_stamp}][skmf]\trepetition {repetition}")
        stream_data = FileStream(filepath=stream_path_csv, target_idx=-1)

        # if 'random_state' not in list(hyperparameters.keys()):
        if 'random_state' in inspect.signature(learner).parameters:
            hyperparameters['random_state'] = repetition
        model_instance = learner(**hyperparameters)

        start_wallclock = time.time()
        start_cpu = time.process_time()

        evaluator = _test_then_train_skmf(
            stream_data=stream_data,
            model=model_instance,
            max_instances=max_instances,
            task_type=task_type,
        )

        # Collect the evaluation results
        result = evaluator.get_mean_measurements()[0]

        end_wallclock = time.time()
        end_cpu = time.process_time()

        wallclock = end_wallclock - start_wallclock
        cpu_time = end_cpu - start_cpu

        # Append raw result to list
        raw_result_dict = {
            "library": "scikit-multiflow",
            "repetition": repetition,
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters).replace("\n", ""),
            "wallclock": wallclock,
            "cpu_time": cpu_time
        }

        result_d = {
            "accuracy": result.accuracy_score(),
            "kappa": result.kappa_score(),
            "recall": result.recall_score(),
            "precision": result.precision_score(),
            "f1": result.f1_score()
        } if task_type == 'classification' else {
            "mean_absolute_error": result.get_average_error(),
            "mean_square_error": result.get_mean_square_error(),
        }

        for name, value in result_d.items():
            raw_result_dict[_capymoa_scikit_name_converter_dict[name]] = value
        raw_results.append(raw_result_dict)
        repetition += 1
        time.sleep(1)
    # Calculate average and std for metrics, wallclock, and cpu_time
    avg_metrics = {name: 'avg_' + _capymoa_scikit_name_converter_dict[name] for name in metric_names}
    std_metrics = {name: 'std_' + _capymoa_scikit_name_converter_dict[name] for name in metric_names}
    for name in metric_names:
        avg_metrics[name] = pd.Series(
            [result[_capymoa_scikit_name_converter_dict[name]] for result in raw_results]).mean()
        std_metrics[name] = pd.Series(
            [result[_capymoa_scikit_name_converter_dict[name]] for result in raw_results]).std()
    avg_wallclock = sum(result['wallclock'] for result in raw_results) / repetitions
    std_wallclock = pd.Series([result['wallclock'] for result in raw_results]).std()
    avg_cpu_time = sum(result['cpu_time'] for result in raw_results) / repetitions
    std_cpu_time = pd.Series([result['cpu_time'] for result in raw_results]).std()

    # Remove the random seed hyperparameter for the aggregated results
    hyperparameters.pop('random_state', hyperparameters)

    # Create DataFrame for aggregated results
    aggregated_result_dict = {
        "library": "scikit-multiflow",
        "dataset": dataset_name,
        "learner": learner_name,
        "hyperparameters": str(hyperparameters),
        "repetitions": repetitions,
        "avg_wallclock": avg_wallclock,
        "std_wallclock": std_wallclock,
        "avg_cpu_time": avg_cpu_time,
        "std_cpu_time": std_cpu_time
    }
    for name in metric_names:
        aggregated_result_dict[f"avg_{_capymoa_scikit_name_converter_dict[name]}"] = avg_metrics[name]
        aggregated_result_dict[f"std_{_capymoa_scikit_name_converter_dict[name]}"] = std_metrics[name]

    df_aggregated = pd.DataFrame(aggregated_result_dict, index=[0])  # Single row

    # Create DataFrame for raw results
    df_raw = pd.DataFrame(raw_results)

    return df_aggregated, df_raw
