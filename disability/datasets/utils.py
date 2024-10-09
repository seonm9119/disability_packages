import os
import pandas as pd

def make_datasets(config, mode='audio'):

    classes_to_idx = {}
    classes = []

    csv_filename = os.path.join(config.data_dir, config.annotation)
    file = pd.read_csv(csv_filename, header=None, encoding='utf-8')

    # 딕셔너리 및 리스트 초기화
    classes_to_idx = pd.Series(file[0].values, index=file[1]).to_dict()
    classes = file[1].tolist()

    dataframes = []
    for class_folder in classes:
        class_path = os.path.join(config.data_dir, class_folder, mode)
        file_list = os.listdir(class_path)

        df = pd.DataFrame({'file_path': file_list})
        df['file_path'] = df['file_path'].apply(lambda x: os.path.join(class_path, x))
        df['label'] = classes_to_idx[class_folder]

        dataframes.append(df)


    df = pd.concat(dataframes, ignore_index=True)
    print(f"Saving merged training data to CSV file at: {config.csv_file}")
    df.to_csv(f'{config.data_dir}/{config.csv_file}', index=False)

    idx_to_classes = {value: key for key, value in classes_to_idx.items()}
    class_counts = df.groupby('label').size().reset_index(name='count')
    class_counts = class_counts[['count', 'label']]
    class_counts['label'] = class_counts['label'].apply(lambda x: str(idx_to_classes[x]).ljust(10))
    print(f"{class_counts}\n")


def load_datasets(config, file_path=None, ratio=0.1, mode='audio'):

    if file_path is None:
        make_datasets(config, mode)
        file_path = os.path.join(config.data_dir, config.csv_file)
        return load_datasets(config, file_path=file_path, ratio=ratio, mode=mode)
    


    df = pd.read_csv(file_path)
    shuffled_df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    val_size = int(len(shuffled_df) * ratio)
    val_df = shuffled_df[:val_size].reset_index(drop=True)
    train_df = shuffled_df[val_size:].reset_index(drop=True)

    print(f"Total number of samples: {len(shuffled_df)}")
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")
    
    return {'train': train_df, 'val': val_df}