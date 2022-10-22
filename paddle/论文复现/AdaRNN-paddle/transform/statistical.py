from dataset.data_weather import get_dataset_statistic


def get_data_statistic(df, station, start_time, end_time):
    mean_train, std_train = get_dataset_statistic(
        df, station, start_time, end_time)
    return mean_train, std_train
