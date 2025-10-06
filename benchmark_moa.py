import subprocess
import pandas as pd
from io import StringIO
from datetime import datetime
import time
from capymoa.evaluation.results import PrequentialResults



_capymoa_moa_name_converter_dict = {
    "classifications correct (percent)": "accuracy",
    "Kappa Statistic (percent)": "kappa",
    "Recall (percent)": "recall",
    "Precision (percent)": "precision",
    "F1 Score (percent)": "f1_score",
    "root mean squared error": "rmse",
    "mean absolute error": "mae",
    "coefficient of determination": "r2",
    "AUC": "auc"
}

def moa_experiment(
		dataset_name,
		learner_name,
		command_line,
        repetitions,
		show_progress=True,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")

    if show_progress:
        print(f"[{date_time_stamp}][moa] Executing {learner_name} on {dataset_name}")


    results = []
    raw_results = []  # Store raw results for each repetition

    repetition = 1

    _randomizable = False
    for _ in range(repetitions):
        if show_progress:
            print(f"[{date_time_stamp}][moa]\trepetition {repetition}")

        if 'seed_value' in command_line:
            _randomizable = True
            command_line = command_line.replace('seed_value', str(repetition))

        if _randomizable:
            command_line = command_line.replace(f'-r {repetition-1}', f'-r {repetition}')

        start_wallclock = time.time()
        # start_cpu = time.process_time()

        output = subprocess.run(command_line, shell=True, capture_output=True, text=True)

        end_wallclock = time.time()
        # end_cpu = time.process_time()
        wallclock = end_wallclock - start_wallclock
        # cpu_time = end_cpu - start_cpu
        result = pd.read_csv(StringIO(output.stdout))
        # result.drop(result.columns[0], axis=1, inplace=True)
        time.sleep(1)
        results.append(result)

        metric_names = result.columns

        raw_result_dict = {
	    	"library":"moa",
	    	"repetition":repetition,
	    	"dataset":dataset_name,
	    	"learner":learner_name,
	    	"hyperparameters": str(command_line.split('-l')[1].split(')')[0].split(learner_name)[-1]),
	    	"wallclock":wallclock,
	    	"cpu_time": result['evaluation time (cpu seconds)'].iloc[-1],
	    }

        for name in metric_names:
            if name in _capymoa_moa_name_converter_dict:
                raw_result_dict[_capymoa_moa_name_converter_dict[name]] = result.iloc[-1][name]
        raw_results.append(raw_result_dict)
        repetition += 1

        time.sleep(1)

    classified_instance = results[0]['classified instances']

    # Calculate average and std for accuracy, wallclock, and cpu_time
    avg_metrics = {name: 'avg' + _capymoa_moa_name_converter_dict[name] for name in metric_names if name in _capymoa_moa_name_converter_dict}
    std_metrics = {name: 'std' + _capymoa_moa_name_converter_dict[name] for name in metric_names if name in _capymoa_moa_name_converter_dict}
    for name in metric_names:
        if name in _capymoa_moa_name_converter_dict:
            avg_metrics[name] = pd.Series(
                [result[_capymoa_moa_name_converter_dict[name]] for result in raw_results]).mean()
            std_metrics[name] = pd.Series(
                [result[_capymoa_moa_name_converter_dict[name]] for result in raw_results]).std()
    avg_wallclock = sum(result['wallclock'] for result in raw_results) / repetitions
    std_wallclock = pd.Series([result['wallclock'] for result in raw_results]).std()
    avg_cpu_time = sum(result['cpu_time'] for result in raw_results) / repetitions
    std_cpu_time = pd.Series([result['cpu_time'] for result in raw_results]).std()

    # avg_wallclock = sum(result['wallclock'] for result in results) / repetitions
    # std_wallclock = pd.Series([result['wallclock'] for result in results]).std()
    # avg_cpu_time = sum(result['cpu_time'] for result in results) / repetitions
    # std_cpu_time = pd.Series([result['cpu_time'] for result in results]).std()

    # avg_metrics = {name: 'avg'+name for name in metric_names if name in _capymoa_moa_name_converter_dict}
    # std_metrics = {name: 'std'+name for name in metric_names if name in _capymoa_moa_name_converter_dict}

    # for name in metric_names:
    #     if name in _capymoa_moa_name_converter_dict:
    #         avg_metrics[name] = pd.Series([getattr(result, name)() for result in results]).mean()
    #         std_metrics[name] = pd.Series([getattr(result, name)() for result in results]).std()


    aggregated_result_dict = {
        "library": "moa",
        "dataset": dataset_name,
        "learner": learner_name,
        "hyperparameters": str(command_line.split('-l')[1].split(')')[0].split(learner_name)[-1]).replace("-r seed_value", ""),
        "repetitions": repetitions,
        "instances": classified_instance,
        "avg_wallclock": avg_wallclock,
        "std_wallclock": std_wallclock,
        "avg_cpu_time": avg_cpu_time,
        "std_cpu_time": std_cpu_time,
    }

    for name in metric_names:
        if name in _capymoa_moa_name_converter_dict:
            aggregated_result_dict[f"avg_{name}"] = avg_metrics[name]
            aggregated_result_dict[f"std_{name}"] = std_metrics[name]

    df = pd.DataFrame(
        aggregated_result_dict
    , index=[0])  # Single row

    # Create DataFrame for raw results
    raw_df = pd.DataFrame(raw_results)

    return df, raw_df


