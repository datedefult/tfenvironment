from dataset.DatasetReturn import data_loader

if __name__ == '__main__':
    dataset = data_loader()
    dataset.to_csv('../joy_data/test_data.csv')