import argparse
from chainer import backend
from chainer import training
from chainer.datasets import get_cifar10
from chainer.training import extensions
from chainer.training import triggers
import chainer
import chainer.links as L

from net import ParallelMLP, MLP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g0', default=0, type=int,
                        help='First GPU ID')
    parser.add_argument('--gpu1', '-g1', default=1, type=int,
                        help='Second GPU ID')
    parser.add_argument('--out', '-o', default='result_model_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', default=1000, type=int,
                        help='Number of units')
    args = parser.parse_args()

    print(f'GPU: {args.gpu0}, {args.gpu1}')
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # model = L.Classifier(MLP(args.unit, 10))
    model = L.Classifier(ParallelMLP(args.unit, 10, args.gpu0, args.gpu1))
    chainer.backends.cuda.get_device_from_id(args.gpu0).use()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = get_cifar10()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu0)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu0))
    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
