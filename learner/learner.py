"""Define the learner."""

import pathlib
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from learner.saver import Saver
from learner.metric import PSNR, SSIM, PSNR_, SSIM_
from util import common_util, constant_util, logger

# for pruning, weight clustering and quantization
import os
import sys
import zipfile
import tempfile
import keras
import tensorflow_model_optimization as tfmot

from tensorflow.keras.layers import Input
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude
from tensorflow_model_optimization.python.core.clustering.keras.experimental import (
    cluster,
)

import pdb

class StandardLearner():
    """Implement the standard learner.

    Attributes:
        config: A `dict` contains the configuration of the learner.
        model: A list of `tf.keras.Model` objects which generate predictions.
        dataset: A dataset `dict` contains dataloader for different split.
        step: An `int` represents the current step. Initialize to 0.
        optimizer: A `tf.keras.optimizers` is used to optimize the model. Initialize to None.
        lr_scheduler: A `tf.keras.optimizers.schedules.LearningRateSchedule` is used to schedule
            the leaning rate. Initialize to None.
        metric_functions: A `dict` contains one or multiple functions which are used to
            metric the results. Initialize to {}.
        saver: A `Saver` is used to save checkpoints. Initialize to None.
        summary: A `TensorboardSummary` is used to save eventfiles. Initialize to None.
        log_dir: A `str` represents the directory which records experiments.
        steps: An `int` represents the number of train steps.
        log_train_info_steps: An `int` represents frequency of logging training information.
        keep_ckpt_steps: An `int` represents frequency of saving checkpoint.
        valid_steps: An `int` represents frequency  of validation.
    """

    def __init__(self, config, model, dataset, log_dir):
        """Initialize the learner and attributes.

        Args:
            config: Please refer to Attributes.
            model: Please refer to Attributes.
            dataset: Please refer to Attributes.
            log_dir: Please refer to Attributes.
        """
        super().__init__()
        with common_util.check_config(config['general']) as cfg:
            self.total_steps = cfg.pop('total_steps', constant_util.MAXIMUM_TRAIN_STEPS)
            self.log_train_info_steps = cfg.pop(
                'log_train_info_steps', constant_util.LOG_TRAIN_INFO_STEPS
            )
            self.keep_ckpt_steps = cfg.pop('keep_ckpt_steps', constant_util.KEEP_CKPT_STEPS)
            self.valid_steps = cfg.pop('valid_steps', constant_util.VALID_STEPS)

        self.config = config
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir

        self.step = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.metric_functions = {}
        self.saver = None
        self.summary = None

        # set random seed
        random.seed(2454)
        np.random.seed(2454)
        tf.random.set_seed(2454)

    def register_training(self):
        """Prepare for training."""
        # prepare learning rate scheduler for training
        lr_config = self.config['lr_scheduler'] if 'lr_scheduler' in self.config else {}
        module = getattr(tf.keras.optimizers.schedules, lr_config.pop('name'))
        self.lr_scheduler = module(**lr_config)

        # prepare optimizer for training
        opt_config = self.config['optimizer'] if 'optimizer' in self.config else {}
        module = getattr(tf.keras.optimizers, opt_config.pop('name'))
        self.optimizer = module(learning_rate=self.lr_scheduler, **opt_config)

        # prepare saver to save and load checkpoints
        saver_config = self.config['saver'] if 'saver' in self.config else {}
        self.saver = Saver(saver_config, self, is_train=True, log_dir=self.log_dir)

        # prepare metric functions
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def register_evaluation(self):
        """Prepare for evaluation."""
        # prepare saver to save and load checkpoints
        saver_config = self.config['saver'] if 'saver' in self.config else {}
        self.saver = Saver(saver_config, self, is_train=False, log_dir=self.log_dir)

        # prepare metric functions
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def register_finetune(self):
        """Prepare for prune/weight cluster."""
        # prepare learning rate scheduler for finetuning
        lr_config = self.config['lr_scheduler'] if 'lr_scheduler' in self.config else {}
        module = getattr(tf.keras.optimizers.schedules, lr_config.pop('name'))
        self.lr_scheduler = module(**lr_config)

        # prepare optimizer for finetuning
        opt_config = self.config['optimizer'] if 'optimizer' in self.config else {}
        module = getattr(tf.keras.optimizers, opt_config.pop('name'))
        self.optimizer = module(learning_rate=self.lr_scheduler, **opt_config)

        # prepare saver to save and load checkpoints
        saver_config = self.config['saver'] if 'saver' in self.config else {}
        self.saver = Saver(saver_config, self, is_train=True, log_dir=self.log_dir)

        # prepare metrics for model.fit()
        self.metric_list = [PSNR_(data_range=255), SSIM_(data_range=255)]
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def loss_fn(self, pred_tensor, target_tensor):
        """Define the objective function and prepare loss for backward.

        Args:
            pred_tensor: A `torch.Tensor` represents the prediction.
            target_tensor: A `torch.Tensor` represents the target.
        """
        # l1 charbonnier loss
        epsilon = 1e-6
        diff = pred_tensor - target_tensor
        loss = tf.math.sqrt(diff * diff + epsilon)
        return tf.reduce_mean(loss)

    def log_metric(self, prefix=''):
        """Log the metric values."""
        metric_dict = {}
        with self.summary.as_default(step=self.step):
            for metric_name in self.metric_functions:
                value = self.metric_functions[metric_name].get_result().numpy()
                self.metric_functions[metric_name].reset()

                tf.summary.scalar(prefix + metric_name, value)
                metric_dict[metric_name] = value

        logger.info(f'Step: {self.step}, {prefix}Metric: {metric_dict}')
        self.summary.flush()

    @tf.function
    def train_step(self, data):
        """Define one training step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        l1_norm_loss = 0
        with tf.GradientTape() as tape:
            for i in range(recurrent_steps):
                if i == 0:
                    b, _, h, w, _ = input_tensors.shape.as_list()
                    input_tensor = tf.concat(
                        [input_tensors[:, 0, ...], input_tensors[:, 0, ...], input_tensors[:, 1, ...]], axis=-1
                    )

                    pred_tensor = self.model(input_tensor, training=True)
                elif i == recurrent_steps - 1:
                    b, _, h, w, _ = input_tensors.shape.as_list()
                    input_tensor = tf.concat(
                        [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i, ...]], axis=-1
                    )
                    pred_tensor = self.model(input_tensor, training=True)
                else:
                    input_tensor = tf.concat(
                        [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i + 1, ...]], axis=-1
                    )
                    pred_tensor = self.model(input_tensor, training=True)
                l1_norm_loss += self.loss_fn(pred_tensor, target_tensors[:, i, ...])
        # Calculate gradients and update.
        gradients = tape.gradient(l1_norm_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return pred_tensor, l1_norm_loss

    def test_step(self, data):
        """Define one testing step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T
        pred_tensors = []
        for i in range(recurrent_steps):
            if i == 0:
                b, _, h, w, _ = input_tensors.shape.as_list()
                input_tensor = tf.concat(
                    [input_tensors[:, 0, ...], input_tensors[:, 0, ...], input_tensors[:, 1, ...]], axis=-1
                )
                pred_tensor = self.model(input_tensor, training=False)
            elif i == recurrent_steps - 1:
                b, _, h, w, _ = input_tensors.shape.as_list()
                input_tensor = tf.concat(
                    [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i, ...]], axis=-1
                )
                pred_tensor = self.model(input_tensor, training=False)
            else:
                input_tensor = tf.concat(
                    [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i + 1, ...]], axis=-1
                )
                pred_tensor = self.model(input_tensor, training=False)
                

            for metric_name in self.metric_functions:
                self.metric_functions[metric_name].update(pred_tensor, target_tensors[:, i, ...])

            pred_tensors.append(pred_tensor)

        return pred_tensors

    def train(self):
        """Train the model."""
        self.register_training()
        self.summary = tf.summary.create_file_writer(self.log_dir)

        # restore checkpoint
        if self.saver.restore_ckpt:
            logger.info(f'Restore from {self.saver.restore_ckpt}')
            self.saver.load_checkpoint()
        else:
            logger.info('Train from scratch')

        train_loader = self.dataset['train']
        train_iterator = iter(train_loader)
        val_loader = self.dataset['val']

        # train loop
        while self.step < self.total_steps:
            try:
                data_pair = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                data_pair = next(train_iterator)
            # training
            pred, loss = self.train_step(data_pair)
            self.step = self.optimizer.iterations.numpy()

            # log the training information every n steps
            if self.step % self.log_train_info_steps == 0:
                with self.summary.as_default(step=self.step):
                    logger.info(f'Step {self.step} train loss: {loss}')
                    tf.summary.scalar('train_loss', loss)
                    tf.summary.scalar('learning_rate', self.optimizer.lr.read_value())
                    tf.summary.image('pred', [pred[0] / 255.0])
                    tf.summary.image(
                        'input',
                        [data_pair[0][0, -2, ...] / 255.0, data_pair[0][0, -1, ...] / 255.0]
                    )
                    tf.summary.image('target', [data_pair[1][0, -1, ...] / 255.0])
                self.summary.flush()

            # save checkpoint every n steps
            if self.step % self.keep_ckpt_steps == 0:
                self.saver.save_checkpoint()

            # validation and log the validation results n steps
            if self.step % self.valid_steps == 0:
                for metric_name in self.metric_functions:
                    self.metric_functions[metric_name].reset()

                for data_pair in val_loader:
                    self.test_step(data_pair)
                    break

                # log the validation results
                self.log_metric('Val_')

        # save the checkpoint after finishing training
        self.saver.save_checkpoint()

    def test(self):
        """Evaluate the model."""
        self.register_evaluation()
        self.summary = tf.summary.create_file_writer(self.log_dir)

        # restore checkpoint
        logger.info(f'Restore from {self.saver.restore_ckpt}')
        self.saver.load_checkpoint()

        val_loader = self.dataset['val']
        save_path = pathlib.Path(self.log_dir) / 'output'
        save_path.mkdir(exist_ok=True)
        for i, data_pair in tqdm(enumerate(val_loader)):
            pred_tensors = self.test_step(data_pair)
            '''
            for j, pred_tensor in enumerate(pred_tensors):
                tf.keras.utils.save_img(
                    save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_tensor[0]
                )
            # break
            '''
        self.model.summary()
        # log the evaluation results
        self.log_metric('Test_')

    def print_model_weights_sparsity(self, model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Wrapper):
                weights = layer.trainable_weights
            else:
                weights = layer.weights
            for weight in weights:
                if "kernel" not in weight.name or "centroid" in weight.name:
                    continue
                weight_size = weight.numpy().size
                zero_num = np.count_nonzero(weight == 0)
                print(
                    f"{weight.name}: {zero_num/weight_size:.2%} sparsity ",
                    f"({zero_num}/{weight_size})",
                )

    def print_model_weight_clusters(self, model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Wrapper):
                weights = layer.trainable_weights
            else:
                weights = layer.weights
            for weight in weights:
                # ignore auxiliary quantization weights
                if "quantize_layer" in weight.name:
                    continue
                if "kernel" in weight.name:
                    unique_count = len(np.unique(weight))
                    print(
                        f"{layer.name}/{weight.name}: {unique_count} clusters "
                    )    

    def get_gzipped_model_size(self, file):
        # It returns the size of the gzipped model in kilobytes.

        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(file)

        return os.path.getsize(zipped_file)/1000

    @tf.function
    def finetune_train_step(self, data):
        """Define one training step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        if self.finetune_cfg == 'prune':
                model = self.pruned_model
        elif self.finetune_cfg == 'cluster':
                model = self.sparsity_clustered_model
        elif self.finetune_cfg == 'quantize':
                model = self.qat_model

        l1_norm_loss = 0
        with tf.GradientTape() as tape:
            for i in range(recurrent_steps):
                if i == 0:
                    b, _, h, w, _ = input_tensors.shape.as_list()
                    input_tensor = tf.concat(
                        [input_tensors[:, 0, ...], input_tensors[:, 0, ...], input_tensors[:, 1, ...]], axis=-1
                    )
                    hidden_state = tf.zeros([b, h, w, 32])
                    pred_tensor, hidden_state = model([input_tensor, hidden_state], training=True)
                elif i == recurrent_steps - 1:
                    b, _, h, w, _ = input_tensors.shape.as_list()
                    input_tensor = tf.concat(
                        [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i, ...]], axis=-1
                    )
                    pred_tensor, hidden_state = model([input_tensor, hidden_state], training=True)
                else:
                    input_tensor = tf.concat(
                        [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i + 1, ...]], axis=-1
                    )
                    pred_tensor, hidden_state = model([input_tensor, hidden_state], training=True)
                l1_norm_loss += self.loss_fn(pred_tensor, target_tensors[:, i, ...])
        # Calculate gradients and update.
        gradients = tape.gradient(l1_norm_loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return {'l1_norm_loss': l1_norm_loss}
    
    def finetune_test_step(self, data):
        """Define one testing step.

        Args:
            data: A `tuple` contains input and target tensor.
        """

        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        if self.finetune_cfg == 'prune':
                model = self.pruned_model
        elif self.finetune_cfg == 'cluster':
                model = self.sparsity_clustered_model
        elif self.finetune_cfg == 'quantize':
                model = self.qat_model

        pred_tensors = []
        for i in range(recurrent_steps):
            if i == 0:
                b, _, h, w, _ = input_tensors.shape.as_list()
                input_tensor = tf.concat(
                    [input_tensors[:, 0, ...], input_tensors[:, 0, ...], input_tensors[:, 1, ...]], axis=-1
                )
                hidden_state = tf.zeros([b, h, w, 32])
                pred_tensor, hidden_state = model([input_tensor, hidden_state], training=False)
            elif i == recurrent_steps - 1:
                b, _, h, w, _ = input_tensors.shape.as_list()
                input_tensor = tf.concat(
                    [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i, ...]], axis=-1
                )
                pred_tensor, hidden_state = model([input_tensor, hidden_state], training=False)
            else:
                input_tensor = tf.concat(
                    [input_tensors[:, i - 1, ...], input_tensors[:, i, ...], input_tensors[:, i + 1, ...]], axis=-1
                )
                pred_tensor, hidden_state = model([input_tensor, hidden_state], training=False)
                
            for metric_name in model.compiled_metrics.metrics:
                metric_fn = model.compiled_metrics.get_metric(metric_name)
                metric_fn.update_state(target_tensors[:, i, ...], pred_tensor)

            pred_tensors.append(pred_tensor)

        return {m.name: m.result() for m in model.compiled_metrics.metrics}
    
    def prune_cluster(self):
        # Configuration register
        self.register_finetune()
        self.summary = tf.summary.create_file_writer(self.log_dir)

        # Restore checkpoint
        if self.saver.restore_ckpt:
            logger.info(f'Restore from {self.saver.restore_ckpt}')
            self.saver.load_checkpoint()
        else:
            logger.info('Restore failed from null!')

        # Define the Functional Model and apply the sparsity API
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
        }

        def apply_pruning_to_conv(layer):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer

        # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_conv` to the layers of the model.
        self.pruned_model = tf.keras.models.clone_model(
            self.model,
            clone_function=apply_pruning_to_conv,
        )
        self.finetune_cfg = 'prune'

        # Use smaller learning rate for fine-tuning
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

        self.pruned_model.compile(
            loss=lambda y_true, y_pred: self.loss_fn(y_pred, y_true),
            optimizer=opt,
            metrics=self.metric_list)

        # Override pruned_model's train_step/test_step method
        self.pruned_model.train_step = self.finetune_train_step
        self.pruned_model.test_step = self.finetune_test_step

        # Pruning through training the model
        train_loader = self.dataset['train']
        val_loader = self.dataset['val']
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep()
        ]
        print('Train pruning model:')
        self.pruned_model.fit(
            train_loader,
            epochs=16,
            steps_per_epoch=1365,
            validation_data=val_loader,
            verbose=1,
            callbacks=callbacks,)

        # Strip the pruning wrapper first, then check that the model kernels were correctly pruned. 
        self.stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(self.pruned_model)

        self.print_model_weights_sparsity(self.stripped_pruned_model)     

        # Apply sparsity preserving clustering and check its effect on model sparsity in both cases    
        cluster_weights = tfmot.clustering.keras.cluster_weights

        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        #cluster_weights = cluster.cluster_weights

        clustering_params = {
            'number_of_clusters': 10,
            'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
            'preserve_sparsity': True
        }

        def apply_clustering_to_conv(layer):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return cluster_weights(layer, **clustering_params)
            return layer

        # Use `tf.keras.models.clone_model` to apply `apply_clustering_to_conv` to the layers of the model.
        self.sparsity_clustered_model = tf.keras.models.clone_model(
            self.model,
            clone_function=apply_clustering_to_conv,
        )
        self.finetune_cfg = 'cluster'

        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

        self.sparsity_clustered_model.compile(
                    optimizer=opt,
                    loss=lambda y_true, y_pred: self.loss_fn(y_pred, y_true),
                    metrics=self.metric_list)

        # Override sparsity_clustered_model's train_step/test_step method
        self.sparsity_clustered_model.train_step = self.finetune_train_step
        self.sparsity_clustered_model.test_step = self.finetune_test_step

        # Clustering through training the model
        print('Train sparsity preserving clustering model:')
        self.sparsity_clustered_model.fit(train_loader,
                                     epochs=16,
                                     steps_per_epoch=1365,
                                     validation_data=val_loader,
                                     verbose=1,)

        # Strip the clustering wrapper first, then check that the model is correctly pruned and clustered.
        self.stripped_clustered_model = tfmot.clustering.keras.strip_clustering(self.sparsity_clustered_model)

        print("Model sparsity:\n")
        self.print_model_weights_sparsity(self.stripped_clustered_model)

        print("\nModel clusters:\n")
        self.print_model_weight_clusters(self.stripped_clustered_model)

        # Save model as 'SavedModel' format to facilitate qat/pcqat 
        self.stripped_clustered_model.save('prune_cluster_pd')

    def quantize(self):
        # Configuration register
        self.register_finetune()
        self.summary = tf.summary.create_file_writer(self.log_dir)
        self.finetune_cfg = 'quantize'

        # Load pruned and clustered model (Prune cluster procedure was deemed to have already been executed in advance)
        print(f'current work dir:{os.getcwd()}\n')
        if os.path.exists('prune_cluster_pd'):
            print("SavedModel format model file path(prune_cluster_pd) exists!\n")
            pass
        else:
            print("Need to execute prune_cluster method in ahead!\n")
            sys.exit()
            
        self.stripped_clustered_model = tf.keras.models.load_model('prune_cluster_pd')
        '''
        # Apply QAT        
        def apply_quantization_to_conv(layer):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            return layer

        # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_conv` 
        # to the layers of the model.

        self.quant_aware_annotate_model = tf.keras.models.clone_model(
            self.stripped_clustered_model,
            clone_function=apply_quantization_to_conv,
        )
           
        self.qat_model = tfmot.quantization.keras.quantize_apply(
            self.quant_aware_annotate_model
        )
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        self.qat_model.compile(optimizer=opt,
                               metrics=self.metric_list,
                               loss=lambda y_true, y_pred: self.loss_fn(y_pred, y_true),)

        train_loader = self.dataset['train']
        val_loader = self.dataset['val']

        # Override qat_model's train_step/test_step method
        self.qat_model.train_step = self.finetune_train_step
        self.qat_model.test_step = self.finetune_test_step


        print('Train qat model:')
        self.qat_model.fit(train_loader,
                           epochs=1,
                           steps_per_epoch=1365,
                           validation_data=val_loader,
                           verbose=1,)
        
                    
        print("\nQAT Model clusters:")
        self.print_model_weight_clusters(self.qat_model)
        print("\nQAT Model sparsity:")
        self.print_model_weights_sparsity(self.qat_model)
        '''


        # Apply PCQAT        
        def apply_quantization_to_conv(layer):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            return layer

        # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_conv` 
        # to the layers of the model.

        self.quant_aware_annotate_model = tf.keras.models.clone_model(
            self.stripped_clustered_model,
            clone_function=apply_quantization_to_conv,
        )
           
        self.qat_model = tfmot.quantization.keras.quantize_apply(
            self.quant_aware_annotate_model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True)
        )
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        self.qat_model.compile(optimizer=opt,
                               metrics=self.metric_list,
                               loss=lambda y_true, y_pred: self.loss_fn(y_pred, y_true),)

        train_loader = self.dataset['train']
        val_loader = self.dataset['val']

        # Override qat_model's train_step/test_step method
        self.qat_model.train_step = self.finetune_train_step
        self.qat_model.test_step = self.finetune_test_step


        print('Train qat model:')
        self.qat_model.fit(train_loader,
                           epochs=1,
                           steps_per_epoch=1365,
                           validation_data=val_loader,
                           verbose=1,)
        print("\nPCQAT Model clusters:")
        self.print_model_weight_clusters(self.qat_model)
        print("\nPCQAT Model sparsity:")
        self.print_model_weights_sparsity(self.qat_model)
        #self.qat_model.save('prune_cluster_qat_pd')


LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}