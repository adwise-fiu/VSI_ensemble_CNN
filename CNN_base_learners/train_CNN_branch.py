import argparse, os
from prepare_dataset_for_network import DataSetGeneratorForBranchCNN
from cnn_network import BranchCNNModel

parser = argparse.ArgumentParser(
    description='Train the constrained_net',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ds_path', type=str, required=True, help="Absolute path to the folder containing all quadrants")
parser.add_argument('--model_path', type=str, required=True, help="Absolute path to the folder where model needs to be saved")
parser.add_argument('--tensor_flow_path', type=str, required=True, help="Absolute path to the folder where tensor information needs to be saved")
parser.add_argument('--quadrant', type=str, required=True, help="Quadrant number. quadrant_1, quadrant_2, quadrant_3, or quadrant_4")
if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.ds_path
    model_path = args.model_path
    tensor_flow_path = args.tensor_flow_path
    quadrant = args.quadrant

    dataset_path = os.path.join(dataset_path, quadrant)
    data_factory = DataSetGeneratorForBranchCNN(input_dir_patchs=dataset_path)

    num_classes = len(data_factory.get_class_names())
    
    train_dataset_dict = data_factory.create_train_dataset()
    print(f'Train dataset contains {len(train_dataset_dict)} samples')

    valid_dataset_dict = data_factory.create_validation_dataset()
    print(f'Validation dataset contains {len(valid_dataset_dict)} samples')
    
    constr_net = BranchCNNModel(quadrant=quadrant, model_files_path=model_path, tensorflow_files_path=tensor_flow_path)
    
    constr_net.create_model(num_classes)
    
    print(f'This model will be trained for {num_classes} classes')
    constr_net.print_model_summary()
    
    history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict, num_classes=num_classes)