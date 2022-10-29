import os
import logging
from parse_args import parse_arguments
from load_data import build_splits_baseline
from experiments.baseline import BaselineExperiment

def setup_experiment(opt):
    
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment()
        train_loader, validation_loader, test_loader = build_splits_baseline(opt)

    else:
        raise ValueError('Experiment not yet supported.')
    
    return experiment, train_loader, validation_loader, test_loader

def main(opt):
    experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)

    iteration = 0
    best_accuracy = 0
    total_train_loss = 0

    # Restore last checkpoint
    if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
        iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')

    # Train loop
    while iteration < opt['max_iterations']:
        for data in train_loader:

            total_train_loss += experiment.train_iteration(data)

            if iteration % opt['print_every'] == 0:
                logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
            
            if iteration % opt['validate_every'] == 0:
                # Run validation
                val_accuracy, val_loss = experiment.validate(validation_loader)
                logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                if val_accuracy > best_accuracy:
                    experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
                experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')

            iteration += 1
            if iteration >= opt['max_iterations']:
                break
    
    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    experiment.validate(test_loader)

if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='w')
    logging.info(opt)

    main(opt)