
import torch
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from ignite import engine, metrics
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs, log_interval):
    writer = SummaryWriter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    m = {'accuracy': metrics.Accuracy(), 'loss': metrics.Loss(loss_fn)}

    trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = engine.create_supervised_evaluator(model, metrics=m, device=device)

    desc = "Loss: {:.2f}"
    train_batches = len(train_loader)

    pbar = tqdm(
        initial=0, leave=False, total=train_batches,
        desc=desc.format(0)
    )

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % train_batches + 1

        writer.add_scalar('Loss/train', engine.state.output, engine.state.iteration - 1)
        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)

        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        
        writer.add_scalar('Accuracy/train', avg_accuracy, engine.state.iteration - 1)

        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.6f} Avg loss: {:.6f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss)
        )

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']

        writer.add_scalar('Accuracy/val', avg_accuracy, engine.state.iteration - 1)
        writer.add_scalar('Loss/val', avg_loss, engine.state.iteration - 1)

        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.6f} Avg loss: {:.6f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
