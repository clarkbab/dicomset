from collections.abc import Callable, Iterable
import logging
import numpy as np
import os
import torch
from torch.autograd import profiler
from torch.cuda.amp import autocast, GradScaler
# from torch.autograd.profiler import profile, tensorboard_trace_handler
from torch.autograd.profiler import profile
from torch.utils.tensorboard import SummaryWriter
from typing import *

from mymi import checkpoint
from mymi import config
from mymi import loaders
from mymi import plotter
from mymi.postprocessing import batch_largest_connected_component
from mymi.reporting import WandbReporter
from mymi import utils
from mymi.metrics import batch_dice, sitk_batch_hausdorff_distance

PRINT_DP = '.10f'

class ModelTrainer:
    def __init__(
        self,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        project_name: str,
        run_name: str,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        visual_validation_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu'),
        early_stopping: bool = False,
        hausdorff_delay: int = 200,
        is_primary: bool = False,
        log_info: Callable[[str], None] = logging.info,
        max_epochs: int = 500,
        mixed_precision: bool = True,
        metrics: Iterable[str] = ('dice', 'hausdorff'),
        print_interval: Union[str, int] = 'epoch',
        report: bool = True,
        report_interval: Union[str, int] = 'epoch',
        spacing: Optional[Iterable[float]] = None,
        validation_interval: Union[str, int] ='epoch') -> None:
        """
        effect: sets the initial trainer values.
        args:
            loss_fn: objective function of the training.
            optimiser: updates the model parameters in response to gradients.
            run_name: the name of the run to show in reporting.
            train_loader: provides the training input and label batches.
            validation_loader: provides the validation input and label batches.
            visual_validation_loader: provides the visual validation input and label batches.
        kwargs:
            device: the device to train on.
            early_stopping: if the training should use early stopping or not.
            hausdorff_delay: calculate HD after this many steps due to HD expense.
            is_primary: is this process the primary in the pool, i.e. responsible for validation and reporting.
            log_info: the logging function. Allows us to include multi-process info if required.
            max_epochs: the maximum number of epochs to run training.
            mixed_precision: run the training using PyTorch mixed precision training.
            metrics: the metrics to print and report during training.
            print_interval: how often to print results during training.
            report: turns reporting on and off.
            report_interval: how often to report results during training.
            spacing: the voxel spacing. Required for calculating Hausdorff distance.
            validation_interval: how often to run the validation.
        """
        self.device = device
        self.early_stopping = early_stopping
        self.hausdorff_delay = hausdorff_delay
        self.is_primary = is_primary
        if is_primary and report:
            # Create tensorboard writer.
            self.reporter = WandbReporter(project_name, run_name)

            # Add hyperparameters.
            hparams = {
                'run-name': run_name,
                'loss-function': str(loss_fn),
                'max-epochs': max_epochs,
                'mixed-precision': mixed_precision,
                'optimiser': str(optimiser),
                'transform': str(train_loader.dataset.transform),
            }
            self.reporter.add_hyperparameters(hparams)
        self.log_info = log_info
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.max_epochs_since_improvement = 20
        self.metrics = metrics
        self.min_validation_loss = np.inf
        self.mixed_precision = mixed_precision
        self.num_epochs_since_improvement = 0
        self.optimiser = optimiser
        self.report = report
        self.run_name = run_name
        self.scaler = GradScaler(enabled=mixed_precision)
        self.spacing = spacing
        if 'hausdorff' in metrics:
            assert spacing is not None, 'Voxel spacing must be provided when calculating Hausdorff distance.'
        self.train_loader = train_loader
        self.train_print_interval = len(train_loader) if print_interval == 'epoch' else print_interval
        self.train_report_interval = len(train_loader) if report_interval == 'epoch' else report_interval
        self.validation_interval = len(train_loader) if validation_interval == 'epoch' else validation_interval
        self.validation_loader = validation_loader
        self.validation_print_interval = len(validation_loader) if print_interval == 'epoch' else print_interval
        self.visual_validation_loader = visual_validation_loader

        # Initialise running scores.
        self.running_scores = {}
        keys = ['print', 'report', 'validation-print', 'validation-report']
        for key in keys:
            self.running_scores[key] = {}
            self.reset_running_scores(key)

    def __call__(
        self,
        model: torch.nn.Module) -> None:
        """
        effect: performs training to update model parameters whilst validating model performance.
        args:
            model: the model to train.
        """
        # Put in training mode.
        model.train()

        for epoch in range(self.max_epochs):
            for batch, (input, label) in enumerate(self.train_loader):
                # Calculate training step.
                step = epoch * len(self.train_loader) + batch

                # Convert input and label.
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self.device), label.to(self.device)

                # Add model structure.
                if self.is_primary and epoch == 0 and batch == 0:
                    # Error when adding graph with 'mixed-precision' training.
                    if not self.mixed_precision:
                        self.reporter.add_model_graph(model, input)

                # Perform forward/backward pass.
                with autocast(enabled=self.mixed_precision):
                    pred = model(input)
                    loss = self.loss_fn(pred, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad()
                self.running_scores['print']['loss'] += [loss.item()]
                self.running_scores['report']['loss'] += [loss.item()]

                # Convert to binary prediction.
                pred = pred.argmax(axis=1)

                # Move data to CPU for metric calculations.
                pred, label = pred.cpu(), label.cpu()

                # Calculate other metrics.
                if 'dice' in self.metrics:
                    dice = batch_dice(pred, label)
                    self.running_scores['print']['dice'] += [dice.item()]
                    self.running_scores['report']['dice'] += [dice.item()]

                if 'hausdorff' in self.metrics and step > self.hausdorff_delay:
                    # Can't calculate HD if prediction is empty.
                    if pred.sum() > 0:
                        hausdorff = sitk_batch_hausdorff_distance(pred, label, spacing=self.spacing)
                        self.running_scores['print']['hausdorff'] += [hausdorff.item()]
                        self.running_scores['report']['hausdorff'] += [hausdorff.item()]
                
                # Print results.
                if self.should_print(self.train_print_interval, step):
                    self.print_training_results(epoch, batch, step)
                    self.reset_running_scores('print')

                # Report results.
                if self.is_primary and self.should_report(step):
                    self.report_training_results(step)
                    self.reset_running_scores('report')

                # Perform validation and checkpointing.
                if self.is_primary and self.should_validate(step):
                    self.validate_model(model, epoch, batch, step)

                # Check early stopping.
                if self.early_stopping:
                    if self.num_epochs_since_improvement >= self.max_epochs_since_improvement:
                        self.log_info(f"Stopping early due to {self.num_epochs_since_improvement} epochs without improved validation score.")
                        return

        self.log_info(f"Maximum epochs ({self.max_epochs} reached.")

    def validate_model(
        self,
        model: torch.nn.Module,
        epoch: int,
        batch: int,
        step: int) -> None:
        """
        effect: evaluates the model on the validation and visual validation sets.
        args:
            model: the model to evaluate.
            epoch: the current training epoch.
            batch: the current training batch.
            step: the current training step.
        """
        model.eval()

        # Calculate validation score.
        for val_batch, (input, label) in enumerate(self.validation_loader):
            # Convert input data.
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            with autocast(enabled=self.mixed_precision):
                pred = model(input)
                loss = self.loss_fn(pred, label)
            self.running_scores['validation-report']['loss'] += [loss.item()]
            self.running_scores['validation-print']['loss'] += [loss.item()]

            # Convert to binary prediction.
            pred = pred.argmax(axis=1)

            # Move data to CPU for metric calculations.
            pred, label = pred.cpu(), label.cpu()

            if 'dice' in self.metrics:
                dice = batch_dice(pred, label)
                self.running_scores['validation-report']['dice'] += [dice.item()]
                self.running_scores['validation-print']['dice'] += [dice.item()]

            if 'hausdorff' in self.metrics and step > self.hausdorff_delay:
                # Can't calculate HD if prediction is empty.
                if pred.sum() > 0:
                    hausdorff = sitk_batch_hausdorff_distance(pred, label, spacing=self.spacing)
                    self.running_scores['print']['hausdorff'] += [hausdorff.item()]
                    self.running_scores['report']['hausdorff'] += [hausdorff.item()]

            # Print results.
            if self.should_print(self.validation_print_interval, val_batch):
                self.print_validation_results(epoch, batch, step, val_batch)
                self.reset_running_scores('validation-print')

        # Check for validation loss improvement.
        loss = np.mean(self.running_scores['validation-report']['loss'])
        if loss < self.min_validation_loss:
            # Save model checkpoint.
            info = {
                'training-epoch': epoch,
                'training-batch': batch,
                'training-step': step,
                'validation-loss': loss
            }
            checkpoint.save(model, self.run_name, self.optimiser, info=info)
            self.min_validation_loss = loss
            self.num_epochs_since_improvement = 0
        else:
            self.num_epochs_since_improvement += 1
        
        # Report validation results.
        if self.report:
            self.report_validation_results(step)
            self.reset_running_scores('validation-report')

        # Plot validation images for visual indication of improvement.
        if self.report:
            for batch, (input, label) in enumerate(self.visual_validation_loader):
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self.device), label.to(self.device)

                # Perform forward pass.
                with autocast(enabled=self.mixed_precision):
                    pred = model(input)

                # Convert prediction to binary values.
                pred = pred.argmax(axis=1)

                # Move data to CPU for calculations.
                input, pred, label = input.squeeze(1).cpu(), pred.cpu(), label.cpu()

                # Loop through batch.
                for sample_idx in range(len(pred)):
                    # Load the label centroid.
                    label_centroid = np.round(np.argwhere(label[sample_idx] == 1).sum(1) / label[sample_idx].sum()).long()

                    # Report centroid slices for each view. 
                    for j, c in enumerate(label_centroid):
                        # Create index.
                        index = [slice(None), slice(None), slice(None)]
                        index[j] = c.item()

                        # Add figure.
                        class_labels = { 1: 'Parotid-Left' }
                        input_data = input[sample_idx][index]
                        label_data = label[sample_idx][index]
                        pred_data = pred[sample_idx][index]

                        # Rotate data and get axis name.
                        if j == 0:
                            input_data = input_data.rot90()
                            label_data = label_data.rot90()
                            pred_data = pred_data.rot90()
                            axis = 'sagittal'
                        elif j == 1:
                            input_data = input_data.rot90()
                            label_data = label_data.rot90()
                            pred_data = pred_data.rot90()
                            axis = 'coronal'
                        elif j == 2:
                            input_data = input_data.rot90(-1)
                            label_data = label_data.rot90(-1)
                            pred_data = pred_data.rot90(-1)
                            axis = 'axial'

                        self.reporter.add_figure(input_data, label_data, pred_data, batch, sample_idx, axis, step, class_labels)

        model.train()
        
    def should_print(
        self,
        interval: int,
        step: int) -> bool:
        """
        returns: whether the training or validation score should be printed.
        args:
            interval: the interval between printing.
            step: the current training or validation step.
        """
        if (step + 1) % interval == 0:
            return True
        else:
            return False

    def should_report(
        self,
        step: int) -> bool:
        """
        returns: whether the training score should be reported.
        args:
            step: the current training step.
        """
        if not self.report:
            return False
        elif (step + 1) % self.train_report_interval == 0:
            return True
        else:
            return False

    def should_validate(
        self,
        step: int) -> bool:
        """
        returns: whether the validation should be performed.
        args:
            step: the current training step.
        """
        if ((self.validation_interval == 'epoch' and (step + 1) % len(self.train_loader) == 0) or
            (self.validation_interval != 'epoch' and (step + 1) % self.validation_interval == 0)):
            return True
        else:
            return False

    def print_training_results(
        self,
        epoch: int,
        batch: int,
        step: int) -> None:
        """
        effect: logs averaged training results over the last print interval.
        args:
            epoch: the current training epoch.
            batch: the current training batch.
            step: the current training step.
        """
        # Get average training loss.
        mean_loss = np.mean(self.running_scores['print']['loss'])
        message = f"[E:{epoch}, B:{batch}, I:{step}] Loss: {mean_loss:{PRINT_DP}}"

        # Get additional metrics.
        if 'dice' in self.metrics:
            mean_dice = np.mean(self.running_scores['print']['dice'])
            message += f", Dice: {mean_dice:{PRINT_DP}}"

        if 'hausdorff' in self.metrics and step > self.hausdorff_delay:
            mean_hausdorff = np.mean(self.running_scores['print']['hausdorff'])
            message += f", Hausdorff: {mean_hausdorff:{PRINT_DP}}"

        self.log_info(message)
        
    def print_validation_results(
        self,
        epoch: int,
        batch: int,
        step: int,
        validation_batch: int) -> None:
        """
        effect: logs the averaged validation results.
        args:
            epoch: the current training epoch.
            batch: the current training batch.
            step: the current training step.
            validation_batch: the current validation batch.
        """
        # Get average validation loss.
        mean_loss = np.mean(self.running_scores['validation-print']['loss'])
        message = f"Validation - [E:{epoch}, B:{batch}, I:{step}, VB:{validation_batch}] Loss: {mean_loss:{PRINT_DP}}"

        # Get additional metrics.
        if 'dice' in self.metrics:
            mean_dice = np.mean(self.running_scores['validation-print']['dice'])
            message += f", Dice: {mean_dice:{PRINT_DP}}"

        if 'hausdorff' in self.metrics and step > self.hausdorff_delay:
            mean_hausdorff = np.mean(self.running_scores['validation-print']['hausdorff'])
            message += f", Hausdorff: {mean_hausdorff:{PRINT_DP}}"

        self.log_info(message)

    def report_training_results(
        self,
        step: int) -> None:
        """
        effect: reports averaged training results.
        args:
            step: the current training step.
        """
        mean_loss = np.mean(self.running_scores['report']['loss'])
        self.reporter.add_metric('Loss/train', mean_loss, step)
        
        if 'dice' in self.metrics:
            mean_dice = np.mean(self.running_scores['report']['dice'])
            self.reporter.add_metric('Dice/train', mean_dice, step)

        if 'hausdorff' in self.metrics and step > self.hausdorff_delay:
            mean_hausdorff = np.mean(self.running_scores['report']['hausdorff'])
            self.reporter.add_metric('Hausdorff/train', mean_hausdorff, step)

    def report_validation_results(
        self,
        step: int) -> None:
        """
        effect: reports averaged validation results.
        args:
            step: the current training step. 
        """
        mean_loss = np.mean(self.running_scores['validation-report']['loss'])
        self.reporter.add_metric('Loss/validation', mean_loss, step)

        if 'dice' in self.metrics:
            mean_dice = np.mean(self.running_scores['validation-report']['dice'])
            self.reporter.add_metric('Dice/validation', mean_dice, step)

        if 'hausdorff' in self.metrics and step > self.hausdorff_delay:
            mean_hausdorff = np.mean(self.running_scores['report']['hausdorff'])
            self.reporter.add_metric('Hausdorff/train', mean_hausdorff, step)

    def reset_running_scores(
        self,
        key: str) -> None:
        """
        effect: initialises the metrics under the key namespace.
        args:
            key: the metric namespace, e.g. print, report, etc.
        """
        self.running_scores[key]['loss'] = []
        if 'dice' in self.metrics:
            self.running_scores[key]['dice'] = []
        if 'hausdorff' in self.metrics:
            self.running_scores[key]['hausdorff'] = []

    def get_batch_centroids(
        self,
        label_b: torch.Tensor,
        plane: str) -> Iterable[int]:
        """
        returns: the centroid location of the label along the plane axis, for each
            image in the batch.
        args:
            label_b: the batch of labels.
            plane: the plane along which to find the centroid.
        """
        assert plane in ('axial', 'coronal', 'sagittal')

        # Move data to CPU.
        label_b = label_b.cpu()

        # Determine axes to sum over.
        if plane == 'axial':
            axes = (0, 1)
        elif plane == 'coronal':
            axes = (0, 2)
        elif plane == 'sagittal':
            axes = (1, 2)

        centroids = np.array([], dtype=np.int)

        # Loop through batch and get centroid for each label.
        for label_i in label_b:
            # Get weighting along 'plane' axis.
            weights = label_i.sum(axes)

            # Get average weighted sum.
            indices = np.arange(len(weights))
            avg_weighted_sum = (weights * indices).sum() /  weights.sum()

            # Get centroid index.
            centroid = np.round(avg_weighted_sum).long()
            centroids = np.append(centroids, centroid)

        return centroids
