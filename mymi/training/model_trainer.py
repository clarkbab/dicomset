from datetime import datetime
import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from mymi import loaders
from mymi import utils
from mymi.metrics import dice as dice_metric

data_dir = os.environ['MYMI_DATA']
TENSORBOARD_DIR = os.path.join(data_dir, 'tensorboard')
CHECKPOINT_DIR = os.path.join(data_dir, 'checkpoints')

class ModelTrainer:
    def __init__(self, train_loader, validation_loader, optimiser, loss_fn, visual_loader, 
        max_epochs=100, run_name=None, metrics=('dice'), device=torch.device('cpu'), print_interval='epoch', 
        record_interval='epoch', validation_interval='epoch', print_format='.10f', num_validation_images=3, 
        is_reporter=False, half_precision=True, log_info=logging.info):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.visual_loader = visual_loader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.device = device
        self.print_interval = print_interval
        self.record_interval = record_interval
        self.validation_interval = validation_interval
        self.num_validation_images = num_validation_images
        self.print_format = print_format
        self.best_loss = None
        self.max_epochs_without_improvement = 5
        self.num_epochs_without_improvement = 0
        self.run_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') if run_name is None else run_name
        self.is_reporter = is_reporter
        self.log_info = log_info
        if is_reporter:
            self.writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, self.run_name))
        self.running_scores = {
            'print': {
                'loss': 0
            },
            'record': {
                'loss': 0
            },
            'validation-print': {
                'loss': 0
            },
            'validation-record': {
                'loss': 0
            }
        }
        if 'dice' in self.metrics: 
            self.running_scores['print']['dice'] = 0
            self.running_scores['record']['dice'] = 0
            self.running_scores['validation-print']['dice'] = 0
            self.running_scores['validation-record']['dice'] = 0

    def __call__(self, model):
        """
        effect: updates the model parameters.
        model: the model to train.
        """
        # Convert model to half precision if necessary.
        # Put in training mode.
        model.train()

        for epoch in range(self.max_epochs):
            for batch, (input, mask) in enumerate(self.train_loader):
                input, mask = input.float(), mask.long()
                input = input.unsqueeze(1)
                input, mask = input.to(self.device), mask.to(self.device)
                if self.is_reporter and epoch == 0 and batch == 0:
                    self.writer.add_graph(model, input)

                # Perform forward/backward pass.
                self.optimiser.zero_grad()
                pred = model(input)
                loss = self.loss_fn(pred, mask)
                loss.backward()
                self.optimiser.step()
                self.running_scores['print']['loss'] += loss.item()
                self.running_scores['record']['loss'] += loss.item()

                # Calculate other metrics.
                if 'dice' in self.metrics:
                    dice = dice_metric(pred, mask)
                    self.running_scores['print']['dice'] += dice.item()
                    self.running_scores['record']['dice'] += dice.item()

                # Record training info to Tensorboard.
                iteration = epoch * len(self.train_loader) + batch
                if self.is_reporter and self.should_record(iteration):
                    self.record_training_results(epoch, iteration)
                    self.reset_record_scores()
                
                # Print results.
                if self.should_print(iteration, len(self.train_loader)):
                    self.print_training_results(epoch, iteration)
                    self.reset_print_scores()

                # Perform validation and checkpointing.
                if self.is_reporter and self.should_validate(iteration):
                    self.validate_model(model, epoch, iteration)

                # Check early stopping.
                if self.num_epochs_without_improvement >= self.max_epochs_without_improvement:
                    self.log_info(f"Stopping early due to {self.num_epochs_without_improvement} epochs without improved validation score.")
                    return

        self.log_info(f"Maximum epochs ({self.max_epochs} reached.")

    def validate_model(self, model, epoch, iteration):
        model.eval()

        # Plot validation images for visual indication of improvement.
        # for batch, (input, mask) in enumerate(self.visual_loader):
        #     input, mask = input.float(), mask.long()
        #     input = input.unsqueeze(1)
        #     input, mask = input.to(self.device), mask.to(self.device)

        #     # Perform forward pass.
        #     pred = model(input)
        #     loss = self.loss_fn(pred, mask)

        #     # Add image data.
        #     image_data = utils.image_data(input, mask, pred)
        #     tag = f"Visual validation batch {batch}"
        #     self.writer.add_images(tag, image_data, dataformats='NCWH', global_step=iteration)

        # Calculate validation score.
        for val_batch, (input, mask) in enumerate(self.validation_loader):
            input, mask = input.float(), mask.long()
            input = input.unsqueeze(1)
            input, mask = input.to(self.device), mask.to(self.device)

            # Perform forward pass.
            pred = model(input)
            loss = self.loss_fn(pred, mask)
            self.running_scores['validation-record']['loss'] += loss.item()
            self.running_scores['validation-print']['loss'] += loss.item()

            if 'dice' in self.metrics:
                dice = dice_metric(pred, mask)
                self.running_scores['validation-record']['dice'] += dice.item()
                self.running_scores['validation-print']['dice'] += dice.item()

            # Print results.
            if self.should_print(val_batch, len(self.validation_loader)):
                self.print_validation_results(epoch, val_batch)
                self.reset_validation_print_scores()
        
        # Record validation results on Tensorboard.
        interval = len(self.validation_loader)
        loss = self.running_scores['validation-record']['loss'] / interval
        
        kwargs = {}
        if 'dice' in self.metrics:
            kwargs['dice'] = self.running_scores['validation-record']['dice'] / interval

        self.record_validation_results(loss, iteration, **kwargs)
        self.reset_validation_record_scores()

        # Check for validation improvement.
        if self.best_loss is None or loss < self.best_loss:
            self.save_model(model, iteration, loss)
            self.num_epochs_without_improvement = 0
        else:
            self.num_epochs_without_improvement += 1

        model.train()

    def save_model(self, model, iteration, loss):
        self.log_info(f"Saving model at iteration {iteration}, achieved best loss: {loss:{self.print_format}}")
        filepath = os.path.join(CHECKPOINT_DIR, self.run_name, 'best.pt')
        info = {
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(info, filepath)
        
    def should_print(self, iteration, loader_length):
        if ((self.print_interval == 'epoch' and iteration % loader_length == loader_length - 1) or
            (self.print_interval != 'epoch' and iteration % self.print_interval == 0 and iteration != 0)):
            return True
        else:
            return False

    def should_record(self, iteration):
        if ((self.record_interval == 'epoch' and iteration % len(self.train_loader) == len(self.train_loader) - 1) or
            (self.record_interval != 'epoch' and iteration % self.record_interval == 0 and iteration != 0)):
            return True
        else:
            return False

    def should_validate(self, iteration):
        if ((self.validation_interval == 'epoch' and iteration % len(self.train_loader) == len(self.train_loader) - 1) or 
            (self.validation_interval != 'epoch' and iteration % self.validation_interval == 0 and iteration != 0)):
            return True
        else:
            return False

    def print_training_results(self, epoch, batch):
        """
        effect: logs an update to STDOUT.
        """
        print_interval = len(self.train_loader) if self.print_interval == 'epoch' else self.print_interval
        message = f"[{epoch}, {batch}] Loss: {self.running_scores['print']['loss'] / print_interval:{self.print_format}}"

        if 'dice' in self.metrics:
            message += f", Dice: {self.running_scores['print']['dice'] / print_interval:{self.print_format}}"

        self.log_info(message)
        
    def print_validation_results(self, epoch, batch):
        """
        effect: logs an update to STDOUT.
        """
        print_interval = len(self.validation_loader) if self.print_interval == 'epoch' else self.print_interval
        message = f"Validation - [{epoch}, {batch}] Loss: {self.running_scores['validation-print']['loss'] / print_interval:{self.print_format}}"

        if 'dice' in self.metrics:
            message += f", Dice: {self.running_scores['validation-print']['dice'] / print_interval:{self.print_format}}"

        self.log_info(message)

    def record_training_results(self, epoch, iteration):
        """
        returns: True if results recorded, false otherwise.
        """
        record_interval = len(self.train_loader) if self.record_interval == 'epoch' else self.record_interval
        avg_loss = self.running_scores['record']['loss'] / record_interval
        self.writer.add_scalar('Loss/train', avg_loss, iteration)
        
        if 'dice' in self.metrics:
            avg_dice = self.running_scores['record']['dice'] / record_interval
            self.writer.add_scalar('Dice/train', avg_dice, iteration)

    def record_validation_results(self, *args, **kwargs):
        loss = args[0]
        iteration = args[1]
        dice = kwargs['dice'] if 'dice' in kwargs else None
        self.writer.add_scalar('Loss/validation', loss, iteration)

        if dice is not None:
            self.writer.add_scalar('Dice/validation', dice, iteration)

    def reset_print_scores(self):
        self.running_scores['print']['loss'] = 0
        self.running_scores['print']['dice'] = 0
    
    def reset_record_scores(self):
        self.running_scores['record']['loss'] = 0
        self.running_scores['record']['dice'] = 0

    def reset_validation_print_scores(self):
        self.running_scores['validation-print']['loss'] = 0
        self.running_scores['validation-print']['dice'] = 0
        
    def reset_validation_record_scores(self):
        self.running_scores['validation-record']['loss'] = 0
        self.running_scores['validation-record']['dice'] = 0

