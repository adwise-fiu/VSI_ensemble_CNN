import argparse

from ensemble_CNN import EnsembleCNN

parser = argparse.ArgumentParser(
    description='Run Ensemble CNN',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--quadrant_1_model_path', type=str, required=True, help='Path to the model for quadrant 1')
parser.add_argument('--quadrant_2_model_path', type=str, required=True, help='Path to the model for quadrant 2')
parser.add_argument('--quadrant_3_model_path', type=str, required=True, help='Path to the model for quadrant 3')
parser.add_argument('--quadrant_4_model_path', type=str, required=True, help='Path to the model for quadrant 4')
parser.add_argument('--ds_path_quadrant_1', type=str, required=True, help='Path to the dataset for quadrant 1')
parser.add_argument('--ds_path_quadrant_2', type=str, required=True, help='Path to the dataset for quadrant 2')
parser.add_argument('--ds_path_quadrant_3', type=str, required=True, help='Path to the dataset for quadrant 3')
parser.add_argument('--ds_path_quadrant_4', type=str, required=True, help='Path to the dataset for quadrant 4')
parser.add_argument('--path_to_directory', type=str, required=True, help='Absolute path to the cnn_ensemble_vsi')

if __name__ == "__main__":
    args = parser.parse_args()
    quadrant_1_model_path = args.quadrant_1_model_path
    quadrant_2_model_path = args.quadrant_2_model_path
    quadrant_3_model_path = args.quadrant_3_model_path
    quadrant_4_model_path = args.quadrant_4_model_path
    ds_path_quadrant_1 = args.ds_path_quadrant_1
    ds_path_quadrant_2 = args.ds_path_quadrant_2
    ds_path_quadrant_3 = args.ds_path_quadrant_3
    ds_path_quadrant_4 = args.ds_path_quadrant_4
    path_to_directory = args.path_to_directory

    ensembleCNN = EnsembleCNN(quadrant_1_model_path, quadrant_2_model_path, quadrant_3_model_path, quadrant_4_model_path,
                    ds_path_quadrant_1, ds_path_quadrant_2, ds_path_quadrant_3, ds_path_quadrant_4,
                    path_to_directory)

    ensembleCNN.run_ensemble()
