# Attack experiments against the privacy of the model parameters.

The client tries to learn the model parameters of the server, for the scaling-based protocol.

Information leaked by the protocol: the sign of each intermediate value.

## Layer-wise attack method

Attack method: start from the first layer, the attacker train the parameters using the sign information layer by layer. The training target is to make the sign of each dimension of the intermediate result to be the same as the leaked sign.
We can train it using the binary cross entropy (BCE) loss or hinge loss. In this experiment, we use the BCE loss as follows, where $sgn(\cdot)$ is the sign function, $f_i(x_i, W_i)$ is the client's view for the output of layer $i$ using input $x_i$ and model parameter $W_i$, and $s_{i+1}$ is the sign leaked by the protocol.

$$ L(W_i; x_i, s_{i+1}) = BCE(sgn(f_i(x_i, W_i)), s_{i+1} ) $$

We can perform the training layer by layer. Once the model parameter $W_i$ is trained, we can use it to compute the scaling factor $m_i$. Then, we can use them to attack the next layer.

We show the attack for the first layer of MiniONN model on CIFAR-10 dataset.

## Model-wise attack method

We can augment the end-to-end *retrain attack* method of the basic prediction API attack with the sign information.

The client knows (x, y) and the sign of every intermediate layer. 
So, we change the training target to not only minimizing the prediction loss but also minimizing the difference between the sign of the intermediate layer and the leaked sign.
Therefore, the loss function for model parameters $W=[W_1, \cdots, W_L]$ over data sample $(x, y)$ is as follows, where $S=[s_1, \cdots, s_L]$ is the leaked signs for each layer, and $x_i$ is the client's view of layer $i$'s output.

$$ L(W; x, y, S) = CELoss(f(x, W), y) + \frac{\theta}{L} \sum_i MSE(sgn(x_i), s_i) $$

The first part is the prediction loss using cross entropy loss.
The second part is the sign loss using mean square error loss. The hyperparameter $\theta$ is used to balance the two parts.
