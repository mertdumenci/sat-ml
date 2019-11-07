
import os

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ignite import engine, metrics
from tqdm import tqdm


def train_model(model, train_loader, val_loader_make, epochs, log_interval, checkpoint_dir):
    writer = SummaryWriter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    m = {'accuracy': metrics.Accuracy(), 'loss': metrics.Loss(loss_fn)}
    trainer = engine.create_supervised_trainer(
        model, optimizer, loss_fn,
        output_transform=lambda x, y, y_pred, loss: (loss.item(), y, y_pred), device=device
    )
    evaluator = engine.create_supervised_evaluator(model, metrics=m, device=device)

    acc_metric = metrics.RunningAverage(metrics.Accuracy(output_transform=lambda x: (x[2], x[1])))
    acc_metric.attach(trainer, 'running_avg_accuracy')
    avg_loss = metrics.RunningAverage(output_transform=lambda x: x[0])
    avg_loss.attach(trainer, 'running_avg_loss')

    desc = "Loss: {:.7f}, Accuracy: {:.7f}"
    train_batches = len(train_loader)

    pbar = tqdm(
        initial=0, leave=False, total=train_batches,
        desc=desc.format(0, 0)
    )

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % train_batches + 1

        avg_loss = engine.state.metrics['running_avg_loss']
        avg_accuracy = engine.state.metrics['running_avg_accuracy']

        if iter % log_interval == 0:
            writer.add_scalar('Loss/train', avg_loss, engine.state.iteration - 1)
            writer.add_scalar('Accuracy/train', avg_accuracy, engine.state.iteration - 1)
            pbar.desc = desc.format(avg_loss, avg_accuracy)
            pbar.update(log_interval)

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader_make())
    
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']

        writer.add_scalar('Accuracy/val', avg_accuracy, engine.state.iteration - 1)
        writer.add_scalar('Loss/val', avg_loss, engine.state.iteration - 1)

        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.6f} Avg loss: {:.6f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))

        pbar.n = pbar.last_print_n = 0

        # Checkpoint
        model_name = type(model).__name__
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}-{engine.state.epoch}epoch-{avg_accuracy}valacc.pt")
        torch.save(model.state_dict(), checkpoint_path)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
