import csv
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from mil_nets.dataset import df_to_dataset
from mil_nets.metrics import bag_accuracy, bag_loss, bag_mse
from mil_nets.utils import convertToBatch, feature_sets
from myargs import args
from network import AMINN
from network import train_eval, test_eval, predict_eval
from sklearn.metrics import roc_auc_score


def AMINN_experiment(dataset):
    """ train and evaluate AMINN model
    # Arguments:
        dataset: bag dataset prepared using function: df_to_dataset
    """
    train_bags = dataset['train']
    test_bags = dataset['test']
    train_set = convertToBatch(train_bags)
    test_set = convertToBatch(test_bags)

    model = AMINN.build(train_set)

    # define losses, including:
    #   1. bag BCE loss calculated from MINN output 'fp' and mortality
    #   2. bag MSE loss calculated from AE output 'recon' and input features
    losses = {
        "fp": bag_loss,
        "recon": bag_mse,
    }
    lossWeights = {
        "fp": args.fp_coef,
        "recon": args.recon_coef,
    }

    adam = Adam(lr=args.lr, decay=args.lr / args.epoch)
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam, metrics=[bag_accuracy])

    # training, testing and predicting
    for epoch in range(args.epoch):
        train_loss, train_acc, fp_loss, mse_loss = train_eval(model, train_set)
        test_loss, test_acc = test_eval(model, test_set)
        if epoch == args.epoch - 1:
            y_true, y_pred = predict_eval(model, test_set)
        print('epoch=', epoch, '   train_loss = {:3f}'.format(train_loss), '   bag_loss = {:3f}'.format(fp_loss),
              '   mse_loss = {:3f}'.format(mse_loss), 'train_acc={:3f}'.format(train_acc),
              '   test_loss = {:3f}'.format(test_loss), '   test_acc={:3f}'.format(test_acc))
    return test_acc, y_true, y_pred


if __name__ == "__main__":
    # read dataframe from csv and convert dataframe to dataset
    df = pd.read_csv('./data/input.csv')
    # normalize=True if applying two-step normalization
    features, labels = feature_sets(df, normalize=True)
    dataset = df_to_dataset(features, labels, nfolds=args.folds, shuffle=True)

    acc = np.zeros((args.runs, args.folds), dtype=float)
    sum_auc = 0
    auc = np.zeros(args.runs, dtype=float)

    # 10 repeated runs of 3-fold cross-validation
    for irun in range(args.runs):
        for ifold in range(args.folds):
            acc[irun][ifold], ytrue, ypred = AMINN_experiment(dataset[ifold])
            print(f'Ground truth: {ytrue}, Predicted: {ypred}')
            print(f'Runs: {irun}, Training fold:{ifold}')
            test_auc = roc_auc_score(ytrue, ypred)
            auc[irun] = test_auc
            print('AUC = ', test_auc)
            sum_auc += test_auc

    avg_auc = sum_auc / args.runs / args.folds
    print('Mean accuracy = ', np.mean(acc))
    print('Acc std = ', np.std(acc))
    print('Mean AUC = ', avg_auc)
    print('AUC std = ', np.std(auc))

    # write model performance and corresponding hyper-parameters to a csv file
    csv_columns = ['recon_coef', 'do_rate', 'lr', 'pooling', 'epoch', 'acc', 'acc_std', 'auc', 'auc_std', 'pretrain',
                   'runs']
    dict_data = [{'recon_coef': args.recon_coef, 'do_rate': args.do, 'lr': args.lr, 'pooling': args.pooling,
                  'epoch': args.epoch, 'acc': np.mean(acc), 'acc_std': np.std(acc), 'auc': avg_auc,
                  'auc_std': np.std(auc), 'runs': args.runs}]

    with open(args.out_csv, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
