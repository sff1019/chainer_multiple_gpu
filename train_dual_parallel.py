import argparse
import argparse
from chainer import backend
from chainer import training
from chainer.datasets import get_cifar10
from chainer.training import extensions
from chainer.training import triggers
import chainer
import chainer.links as L
import chainermn
import chainermn.datasets
import chainermn.functions

from parallel_net import ParallelMLP, MLP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    parser.add_argument('--epoch', '-e', default=20, type=int)
    parser.add_argument('--gpu0', '-g0', default=0, type=int)
    parser.add_argument('--gpu1', '-g1', default=1, type=int)
    parser.add_argument('--out', '-o', default='result_model_parallel')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', default=1000, type=int,
                        help='Number of units')
    parser.add_argument('--parallel_data', '-pd', type=bool, default=False)
    args = parser.parse_args()

    model = L.Classifier(MLP(args.unit, 10))

    # Create communicator
    comm = chainermn.create_communicator()
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()

    # Create a multi-node optimzer
    optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.Adam(), comm)
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
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
