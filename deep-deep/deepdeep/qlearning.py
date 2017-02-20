# -*- coding: utf-8 -*-
"""
Q-Learning for Web Crawling
===========================

Q-learning estimator with linear function approximation,
experience replay and double learning.

State-action value :math:`Q(s, a)` function is used. This function
predicts "return" :math:`R` - a discounted sum of all future rewards after
following action :math:`a` from state :math:`s`.

.. math::

    R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + ... = \\
          r_{t+1} + \gamma R_{t+1}

    0 \leq \gamma < 1


Q function parameters are learned from "training examples":

* :math:`a_t` action taken (i.e. a feature vector for a link followed);
* :math:`r_{t+1}` observed reward (scalar value, e.g. whether a form is
  found or is a page on-topic);
* a set of actions :math:`A_{t+1}` (i.e. a feature matrix of links) available
  at this page; next action :math:`a_{t+1} \in A_{t+1}` used for TD updates
  is chosen from this set. In Q-learning it is a link with the highest
  :math:`Q(a_{t+1})` score. We need to store all available actions in
  experience replay memory because Q function changes over time.
* :math:`s_t` state (feature vector for the page a link is extracted from);
* :math:`s_{t+1}` state (feature vector for the current page, i.e. a page
  the link leads to);

With this data we can train a regression model for :math:`Q(s,a)` function
using any machine learning method. The trick is that instead of true
return values (which are unknown) in Q learning (and TD methods in general)
current estimates are used to train the model. Recall that Q function
is an approximation of R.

.. math::

    R_{predicted} = Q(s_t, a_t)

    R_{observed} = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}),

The regression model is trained on samples from experience replay memory;
currently it is not trained online. Experience replay provides several
benefits:

* data is used more efficiently;
* training is more stable - pure online training introduces strong biases
  because examples don't come in random order;
* a less obvious benefit is that even though we use a first-order
  approximation :math:`R_{observed} = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})`,
  credit can be assigned to states and actions from several steps back if
  we repeatedly sample from the replay memory. So initially it works
  like :math:`TD(0)`, but over time, as we keep sampling, it moves towards
  :math:`TD(1)`. The effect is similar to :math:`TD(\lambda)`.

To stabilize training instead of a single :math:`Q(s, a)` function
two functions are used:

* target :math:`Q(s, a)` - this function is used for predictions, to define
  which action to follow; it doesn't change for a specified number of steps;
* online :math:`Q(s, a)` - this function is being trained using samples from
  experience replay memory; each N steps parameters of online Q function
  are copied to the target Q function.

For efficiency reasons instead of two (s, a) vectors a single vector is used,
with all features joined. It requires ~2x RAM because multiple
`s` copies are stored in memory, but the scipy-based implementation becomes
10x faster.
"""
from __future__ import absolute_import
import random
from typing import Callable, List, Tuple, Any, Optional

import numpy as np
from scipy import sparse
import sklearn.base
from sklearn.linear_model import SGDRegressor

from deepdeep.utils import log_time, csr_nbytes


class QLearner:
    """
    This class represents :math:`Q(s, a)` function approximated with
    Linear Regression and knows how to train it using Q-Learning algorithm
    with Experience Replay and Double Learning.

    Parameters
    ----------
    double_learning : bool
        Whether to use Double Learning (default: True).
        Currently a simple variant of Double Learning is implemented
        (http://arxiv.org/abs/1509.06461): instead of using two totally
        separate Q functions it uses online and target Q functions
        which we need anyways to stabilize Experience Replay training.
    steps_before_switch : int
        Parameters of online Q function are copied to target Q function
        every `steps_before_switch` steps (default: 100).
    gamma : float
        Discounting factor, ``0 <= gamma < 1`` (default: 0.4).
        Lower values make spider focus on immediate reward::

            gamma     % of credit assigned to n-th previous step  effective steps
            -----     ------------------------------------------  ---------------
            0.00      100   0   0   0   0   0   0   0   0   0     1
            0.05      100   5   0   0   0   0   0   0   0   0     2
            0.10      100  10   1   0   0   0   0   0   0   0     2
            0.15      100  15   2   0   0   0   0   0   0   0     2
            0.20      100  20   4   0   0   0   0   0   0   0     2
            0.25      100  25   6   1   0   0   0   0   0   0     3
            0.30      100  30   9   2   0   0   0   0   0   0     3
            0.35      100  35  12   4   1   0   0   0   0   0     3
            0.40      100  40  16   6   2   1   0   0   0   0     4
            0.45      100  45  20   9   4   1   0   0   0   0     4
            0.50      100  50  25  12   6   3   1   0   0   0     5
            0.55      100  55  30  16   9   5   2   1   0   0     6
            0.60      100  60  36  21  12   7   4   2   1   1     6
            0.65      100  65  42  27  17  11   7   4   3   2     7
            0.70      100  70  48  34  24  16  11   8   5   4     9
            0.75      100  75  56  42  31  23  17  13  10   7     10
            0.80      100  80  64  51  40  32  26  20  16  13     10+
            0.85      100  85  72  61  52  44  37  32  27  23     10+
            0.90      100  90  81  72  65  59  53  47  43  38     10+
            0.95      100  95  90  85  81  77  73  69  66  63     10+

    initial_predictions : float
        Default :math:`Q(s, a)` value when the model is not fit yet
        (default : 0.05).
    replay_sample_size : int
        How many examples from replay memory to use on each time step
        (default: 300). At each time step (i.e. with each new experience)
        ``replay_sample_size`` random examples are fetched from the
        replay memory, and online :math:`Q(s, a)` function is trained on
        these examples.
    fit_interval : int
        How often to update online :math:`Q(s, a)` function (default: 1).
        By default, it is updated on each time step; set ``fit_interval``
        to a higher value to update on each fit_interval-th time step.
    on_model_changed: callable, optional
        Function to call when target :math:`Q(s, a)` function is changed.
    pickle_memory: bool
        When True (default), experience replay memory is pickled along
        with the model; it allows to resume training after unpickling.
        Set it to False if model is going to be used only for predictions
        after unpickling; it can save a huge amount of memory.
    dummy: bool
        When True, don't learn anything. Default is False.
    er_maxsize: int, optional
        Max size of experience replay memory. None (default) means there
        is no limit.
    er_maxlinks: int, optional
        Max number of links in experience replay memory.
        None (default) means there is no limit.
    """
    def __init__(self, *,
                 double_learning: bool = True,
                 steps_before_switch: int = 100,
                 gamma: float = 0.4,
                 initial_predictions: float = 0.05,
                 replay_sample_size: int = 300,
                 fit_interval: int = 1,
                 on_model_changed: Optional[Callable[[], None]]=None,
                 pickle_memory: bool = True,
                 dummy: bool = False,
                 er_maxsize: Optional[int] = None,
                 er_maxlinks: Optional[int] = None,
                 clf_penalty: str='l2',
                 clf_alpha: float=1e-6
                 ) -> None:
        assert 0 <= gamma < 1
        self.double_learning = double_learning
        self.steps_before_switch = steps_before_switch
        self.gamma = gamma
        self.initial_predictions = initial_predictions
        self.replay_sample_size = replay_sample_size
        self.on_model_changed = on_model_changed
        self.fit_interval = fit_interval
        self.pickle_memory = pickle_memory
        self.dummy = dummy

        self.clf_online = SGDRegressor(
            penalty=clf_penalty,
            average=False,
            n_iter=1,
            learning_rate='constant',
            # loss='epsilon_insensitive',
            alpha=clf_alpha,
            eta0=0.1,
        )

        self.clf_target = sklearn.base.clone(self.clf_online)  # type: SGDRegressor
        self.memory = ExperienceMemory(maxsize=er_maxsize, maxlinks=er_maxlinks)
        self.t_ = 0

    @classmethod
    def join_As(cls,
                A: sparse.spmatrix,
                s: Optional[sparse.spmatrix]) -> sparse.csr_matrix:
        """
        Append vector ``s`` to each row of matrix ``A``.

        For efficiency reasons state vector should be appended to each
        action vector. It requires ~2x RAM, but is ~10x faster in the end.
        """
        if A is not None and s is not None:
            n_rows = A.shape[0]
            S = sparse.vstack([sparse.coo_matrix(s)] * n_rows)
            return sparse.hstack([A, S]).tocsr()
        else:
            return A

    @classmethod
    def join_as(cls,
                a: sparse.spmatrix,
                s: Optional[sparse.spmatrix]) -> sparse.csr_matrix:
        """ Append sparse vector ``s`` to sparse vector ``a``. """
        return sparse.hstack([a, s]).tocsr() if s is not None else a

    def add_experience(self, as_t, AS_t1, r_t1) -> None:
        """
        Tell QLearner about the observed experience. QLearner stores it
        to the experience replay memory and updates Q functions.
        """
        self.t_ += 1
        if not self.dummy:
            self.memory.add(as_t=as_t, AS_t1=AS_t1, r_t1=r_t1)

            if (self.t_ % self.fit_interval) == 0:
                self.fit_iteration(self.replay_sample_size)

        if (self.t_ % self.steps_before_switch) == 0:
            self._update_target_clf()
            if self.on_model_changed is not None:
                self.on_model_changed()

    def predict(self, AS: sparse.csr_matrix, online: bool=False) -> np.ndarray:
        """
        Compute Q(s, a) function for all state-action pairs.

        Parameters
        ----------

        AS : csr_matrix, shape (n_rows, n_action_features + n_state_features)
             Feature matrix for actions. If state features are used, state
             feature vector should be appended to each action feature row.

             See also: :meth:`join_As`.

        online : bool
            Whether to use online Q function (default is False, meaning
            target Q function is used for predictions).

        Returns
        -------
        y : array-like, shape (n_rows,)
            A vector of :math:`Q(s, a)` values.

        """
        clf = self.clf_target if not online else self.clf_online
        if clf.coef_ is None:
            return np.ones(AS.shape[0]) * self.initial_predictions
        return clf.predict(AS)

    def predict_one(self, as_, online=False) -> float:
        """
        Compute Q(s, a) function.

        If you have many (s, a) pairs it is better to use :meth:`predict`
        method because it is faster.

        Parameters
        ----------

        ``as_`` : array-like, shape (n_action_features + n_state_features,)
            Feature vector for action. If state features are used, state
            feature vector should be appended to the action feature vector.

            See also: :meth:`join_as`.

        online : bool
            Whether to use online Q function (default is False, meaning
            target Q function is used for predictions).

        Returns
        -------
        y : float
            :math:`Q(s, a)` value

        """
        return self.predict(sparse.vstack([as_]), online=online)[0]

    @log_time
    def fit_iteration(self, sample_size: int) -> None:
        """
        Update online Q function using random examples from the experience
        replay memory.
        """
        sample = self.memory.sample(sample_size)
        as_t_list, AS_t1_list, r_t1_list = zip(*sample)
        rewards = np.asarray(r_t1_list)
        X = sparse.vstack(as_t_list)
        Q_t1_vector = self._get_Q_t1_values(rewards.shape, AS_t1_list)
        y = rewards + self.gamma * Q_t1_vector
        self.clf_online.partial_fit(X, y)

    def _get_Q_t1_values(self,
                         shape: Tuple,
                         AS_t1_list: List[sparse.csr_matrix],
                         ):
        Q_t1_values = np.zeros(shape)

        # XXX: An alternative way to implement it would be to
        # stack all AS_t1 matrices into one big matrix, predict
        # all scores at once, and then slice the result according
        # to shapes of original matrices to fill Q_t1_values.
        # The implementation can be found here:
        # https://gist.github.com/kmike/c0d3fa1822cd6ddcdbca9b067ee3e94a;
        # it turns out to be ~4x slower.

        for idx, AS_t1 in enumerate(AS_t1_list):
            if AS_t1 is not None and AS_t1.shape[0] > 0:
                scores = self.predict(AS_t1, online=True)
                if self.double_learning:
                    # This is a simple variant of double learning
                    # used in http://arxiv.org/abs/1509.06461.
                    # Instead of using totally separate Q functions
                    # action is chosen by online Q function, but the score
                    # is estimated using target Q function.
                    best_idx = scores.argmax()
                    as_t1 = AS_t1[best_idx]
                    Q_t1_values[idx] = self.predict_one(as_t1, online=False)
                else:
                    Q_t1_values[idx] = scores.max()  # vanilla Q-learning
        # print('Q_t1_values shape:', Q_t1_values.shape)
        # print('Total links: ', sum(_A.shape[0] for _A in AS_t1_list if _A is not None))
        return Q_t1_values

    def _update_target_clf(self):
        trained_params = [
            't_',
            'coef_',
            'intercept_',
            'average_coef_',
            'average_intercept_',
            'standard_coef_',
            'standard_intercept_',
        ]
        for attr in trained_params:
            if not hasattr(self.clf_online, attr):
                continue
            data = getattr(self.clf_online, attr)
            if hasattr(data, 'copy'):
                data = data.copy()
            setattr(self.clf_target, attr, data)

    def coef_norm(self, online: bool=True) -> float:
        """ Return L2 norm of classifier weights """
        clf = self.clf_target if not online else self.clf_online
        if clf.coef_ is None:
            return 0
        return np.sqrt((clf.coef_ ** 2).sum())

    def __getstate__(self):
        dct = self.__dict__.copy()
        try:
            del dct['on_model_changed']
        except KeyError:
            pass
        if not self.pickle_memory:
            dct['memory'] = ExperienceMemory()
        return dct


class ExperienceMemory:
    """
    Experience replay memory.

    Parameters
    ----------
    maxsize : int, optional
        When maxsize is passed, replay memory size is limited.
        When memory size is over limit a new observation replaces
        a random observation from the memory. By default there is no
        limit.

        Random replaces mean that experience replay memory contains
        both old and new examples, and that over time average age of
        observations stored in memory converges to ``maxsize``.

        Ring buffer would have been more aggressive pruning old observations;
        average age would have been ``maxsize/2`` with a ring buffer.

    maxlinks : int, optional
        Has the same logic as maxsize, but controls maximum number of links:
        each observation can contain multiple links. This is useful to
        control in case of running separate spiders for each domain,
        as different domains have different average number of links.
    """
    def __init__(self,
                 maxsize: Optional[int]=None,
                 maxlinks: Optional[int]=None,
                 ) -> None:
        self.data = []  # type: List[Tuple[Any, Any, Any]]
        self.maxsize = maxsize
        self.maxlinks = maxlinks
        self._n_links = 0

    def add(self, as_t, AS_t1, r_t1) -> None:
        """
        Add an example to the replay memory.

        If memory is full and maxsize is enabled, a random example from
        memory is replaced with a passed example.
        """
        # TODO: In AS matrix rows of S columns usually contains the same data;
        # delta-compress them?
        item = (as_t, AS_t1, r_t1)
        too_large = False
        if self.maxsize and len(self.data) >= self.maxsize:
            too_large = True
        elif self.maxlinks and self._n_links >= self.maxlinks:
            too_large = True
        self._n_links += AS_t1.shape[0] if AS_t1 is not None else 0
        if not too_large:
            self.data.append(item)
        else:
            idx = random.randint(0, len(self.data)-1)
            removed = self.data[idx]
            if removed[1] is not None:
                self._n_links -= removed[1].shape[0]
            self.data[idx] = item

    def sample(self, k: int) -> List[Tuple[Any, Any, Any]]:
        """
        Return no more than ``k`` random examples from the memory.
        """
        assert k >= 0
        k = min(k, len(self.data))
        return random.sample(self.data, k)

    def clear(self) -> None:
        self.data.clear()
        self._n_links = 0

    def __len__(self) -> int:
        return len(self.data)

    def nbytes(self) -> int:
        """
        Memory taken by sparse matrices in self.data.
        """
        return sum(csr_nbytes(as_t) + csr_nbytes(AS_t1)
                   for as_t, AS_t1, _ in self.data)
