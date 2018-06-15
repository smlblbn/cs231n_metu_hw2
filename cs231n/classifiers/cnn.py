import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def init_cnn(weight_scale=1e-3, bias_scale=0, input_shape=(1, 1, 1000), num_classes=2, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model


def regresion_loss(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss = np.sum(np.square(scores - y)) / 2
  dscores = scores -y

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def train(X, y, X_val, y_val,
          model, loss_function,
          reg=0.0,
          learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
          update='momentum', sample_batches=True,
          num_epochs=30, batch_size=100, acc_frequency=None,
          verbose=False):
  """
  Optimize the parameters of a model to minimize a loss function. We use
  training data X and y to compute the loss and gradients, and periodically
  check the accuracy on the validation set.

  Inputs:
  - X: Array of training data; each X[i] is a training sample.
  - y: Vector of training labels; y[i] gives the label for X[i].
  - X_val: Array of validation data
  - y_val: Vector of validation labels
  - model: Dictionary that maps parameter names to parameter values. Each
    parameter value is a numpy array.
  - loss_function: A function that can be called in the following ways:
    scores = loss_function(X, model, reg=reg)
    loss, grads = loss_function(X, model, y, reg=reg)
  - reg: Regularization strength. This will be passed to the loss function.
  - learning_rate: Initial learning rate to use.
  - momentum: Parameter to use for momentum updates.
  - learning_rate_decay: The learning rate is multiplied by this after each
    epoch.
  - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
  - sample_batches: If True, use a minibatch of data for each parameter update
    (stochastic gradient descent); if False, use the entire training set for
    each parameter update (gradient descent).
  - num_epochs: The number of epochs to take over the training data.
  - batch_size: The number of training samples to use at each iteration.
  - acc_frequency: If set to an integer, we compute the training and
    validation set error after every acc_frequency iterations.
  - verbose: If True, print status after each epoch.

  Returns a tuple of:
  - best_model: The model that got the highest validation accuracy during
    training.
  - loss_history: List containing the value of the loss function at each
    iteration.
  - train_acc_history: List storing the training set accuracy at each epoch.
  - val_acc_history: List storing the validation set accuracy at each epoch.
  """
  step_cache = {}
  N = X.shape[0]

  if sample_batches:
    iterations_per_epoch = N / batch_size  # using SGD
  else:
    iterations_per_epoch = 1  # using GD
  num_iters = num_epochs * iterations_per_epoch
  epoch = 0
  best_val_loss = float('inf')
  best_model = {}
  train_loss_history = []
  val_loss_history = []
  for it in xrange(num_iters):
    if it % 10 == 0:  print 'starting iteration ', it

    # get batch of data
    if sample_batches:
      batch_mask = np.random.choice(N, batch_size)
      X_batch = X[batch_mask]
      y_batch = y[batch_mask]
    else:
      # no SGD used, full gradient descent
      X_batch = X
      y_batch = y

    # evaluate cost and gradient
    train_loss, grads = loss_function(X_batch, model, y_batch, reg)
    train_loss_history.append(train_loss)

    val_scores = loss_function(X_val, model)
    val_loss = np.sum(np.square(val_scores - y_val)) / 2
    val_loss_history.append(val_loss)

    cache = 0

    # perform a parameter update
    for p in model:
      # compute the parameter step
      if update == 'sgd':
        dx = -learning_rate * grads[p]

      elif update == 'momentum':
        if not p in step_cache:
          step_cache[p] = np.zeros(grads[p].shape)

        #####################################################################
        # Momentum                                                          #
        #####################################################################
        step_cache[p] = momentum * step_cache[p] - learning_rate * grads[p]
        dx = step_cache[p]

      elif update == 'rmsprop':
        decay_rate = 0.99  # you could also make this an option TODO
        if not p in step_cache:
          step_cache[p] = np.zeros(grads[p].shape)
        dx = np.zeros_like(grads[p])  # you can remove this after
        #####################################################################
        # RMSProp                                                           #
        #####################################################################
        step_cache[p] = decay_rate * step_cache[p] + (1 - decay_rate) * grads[p] ** 2
        dx = - learning_rate * grads[p] / np.sqrt(step_cache[p] + 1e-8)

      else:
        raise ValueError('Unrecognized update type "%s"' % update)

      # update the parameters
      model[p] += dx

    # every epoch perform an evaluation on the validation set
    first_it = (it == 0)
    epoch_end = (it + 1) % iterations_per_epoch == 0
    acc_check = (acc_frequency is not None and it % acc_frequency == 0)
    if first_it or epoch_end or acc_check:
      if it > 0 and epoch_end:
        # decay the learning rate
        learning_rate *= learning_rate_decay
        epoch += 1

      # keep track of the best model based on validation loss
      if val_loss < best_val_loss:
        # make a copy of the model
        best_val_loss = val_loss
        best_model = {}
        for p in model:
          best_model[p] = model[p].copy()

      # print progress if needed
      if verbose:
        print ('Finished epoch %d / %d: train loss %f validation loss %f'
               % (epoch, num_epochs, train_loss, val_loss))

  if verbose:
    print 'finished optimization. best validation loss: %f' % (best_val_loss)
  # return the best model and the training history statistics
  return best_model, train_loss_history, val_loss_history
