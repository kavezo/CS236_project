# Copyright (c) 2018 Rui Shu
import numpy as np
import os
import shutil
import sys
import torch
import io
import re
# import tensorflow as tf
from codebase.models.gmvae import GMVAE
from codebase.models.ssvae import SSVAE
from codebase.models.vae import VAE
from codebase.models.conv_vae import ConvVAE
from torch.nn import functional as F
from torchvision import datasets, transforms

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

################################################################################
# Please familiarize yourself with the code below.
#
# Note that the notation is
# argument: argument_type: argument_shape
#
# Furthermore, the expected argument_shape is only a guideline. You're free to
# pass in inputs that violate the expected argument_shape provided you know
# what you're doing
################################################################################

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z
    ################################################################################
    batch = m.size()[0]
    dim = m.size()[1]
    eps = torch.distributions.normal.Normal(torch.zeros(dim), 1).sample((batch,))
    z = m + v.sqrt()*eps
    ################################################################################
    # End of code modification
    ################################################################################
    return z


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################

    inside = (x - m) / v.sqrt()
    inside = -0.5 * inside * inside
    log_prob = inside - (np.sqrt(2*np.pi) * v.sqrt()).log()
    log_prob = log_prob.sum(-1)

    # log_prob = torch.distributions.normal.Normal(m,v).log_prob(x).sum(-1)

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################
    z_mix = z.unsqueeze(1).expand(-1, m.shape[1], -1)
    log_prob = log_mean_exp(log_normal(z_mix, m, v), -1)
    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob


def kl_cat(q, log_q, log_p):
    """
    Computes the KL divergence between two categorical distributions

    Args:
        q: tensor: (batch, dim): Categorical distribution parameters
        log_q: tensor: (batch, dim): Log of q
        log_p: tensor: (batch, dim): Log of p

    Return:
        kl: tensor: (batch,) kl between each sample
    """
    element_wise = (q * (log_q - log_p))
    kl = element_wise.sum(-1)
    return kl


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def duplicate(x, rep):
    """
    Duplicates x along dim=0

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def load_model_by_name(model, global_step, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join('checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


################################################################################
# No need to read/understand code beyond this point. Unless you want to.
# But do you tho ¯\_(ツ)_/¯
################################################################################


def evaluate_lower_bound(model, labeled_test_subset, run_iwae=True):
    check_model = isinstance(model, VAE) or isinstance(model, GMVAE) or isinstance(model, ConvVAE)
    assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    xl, _ = labeled_test_subset
    torch.manual_seed(0)
    xl = torch.bernoulli(xl)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec = detach_torch_tuple(fn(xl))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
    print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

    if run_iwae:
        for iw in [1, 10, 100, 1000]:
            repeat = max(100 // iw, 1) # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec = compute_metrics(fn, repeat)
            print("Negative IWAE-{}: {}".format(iw, niwae))

def evaluate_lower_bound_continuous(model, labeled_test_subset, run_iwae=True):
    check_model = isinstance(model, VAE) or isinstance(model, GMVAE) or isinstance(model, ConvVAE)
    assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    xl, _ = labeled_test_subset
    torch.manual_seed(0)
    xl = torch.from_numpy(xl)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec = detach_torch_tuple(fn(xl))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
    print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))


def evaluate_classifier(model, test_set):
    check_model = isinstance(model, SSVAE)
    assert check_model, "This function is only intended for SSVAE"

    print('*' * 80)
    print("CLASSIFICATION EVALUATION ON ENTIRE TEST SET")
    print('*' * 80)

    X, y = test_set
    pred = model.cls.classify(X)
    accuracy = (pred.argmax(1) == y).float().mean()
    print("Test set classification accuracy: {}".format(accuracy))


def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        delete_existing(log_dir)
        delete_existing(save_dir)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer


def log_summaries(writer, summaries, global_step):
    pass # Sad :<
    # for tag in summaries:
    #     val = summaries[tag]
    #     tf_summary = tf.Summary.Value(tag=tag, simple_value=val)
    #     writer.add_summary(tf.Summary(value=[tf_summary]), global_step)
    # writer.flush()


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)


def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass

class HiyoriDataset(torch.utils.data.Dataset):
    def __init__(self, otopath = 'data/hiyori/train.ini'):
        self.otopath = otopath
        self.vbpath = os.path.dirname(otopath)
        self.read_oto()
        countdict = {}
        self.labels = []
        for filename in self.eventdict:
            for line in self.eventdict[filename]:
                if len(line) == 0 or line[0] == 'E': break
                seq = line[0]
                countdict[seq] = countdict.get(seq, 0) + 1
                suffix = str(countdict[seq]) if countdict[seq] > 1 else ''
                self.labels.append(seq + suffix)

        self.reverse_labels = {label: i for i, label in enumerate(set(self.labels))}

        loadnpy = lambda seq: np.load(self.vbpath+'/mel/'+seq+'.npy')
        self.data = [loadnpy(seq).real for seq in self.labels]

        min_len = 157 #min([mel.shape[1] for mel in self.data]) # hardcode...

        self.data = [mel[:, :min_len] for mel in self.data] # crop to smallest size
        self.data = [(mel-np.min(mel))/(np.max(mel)-np.min(mel)) for mel in self.data] # put in [0, 1] range
        self.data = [np.expand_dims(data, 0) for data in self.data] # add channel dim
        self.data = [mel.astype('float32') for mel in self.data] # for PyTorch
        self._feat_shape = (1, self.data[0].shape[0], min_len)


    def read_oto(self):
        self.linepat = re.compile(r'(.+\.wav)=(.+),([\-.\d]+),([\-.\d]+),([\-.\d]+),([\-.\d]+),([\-.\d]+)')

        with io.open(self.otopath, 'r', encoding='utf-8') as oto:
            lines = oto.readlines()
        lines = [line.strip() for line in lines]

        self.eventdict = {}
        processed_names = set()
        for line in lines:
            try:
                name = self.linepat.match(line).group(1)
            except:
                continue
            if not name in processed_names:
                processed_names.add(name)
                currlines = [l for l in lines if name in l]
                phonemes = self.process_oto_lines(currlines)
                if not len(phonemes) == 0:
                    self.eventdict[name] = phonemes

    def process_oto_lines(self, lines):
        triphones = []

        # remove labels that are the same except for the alias
        i = 0
        end_set = []
        while 1:
            end = lines[i][lines[i].find(','):]
            if end in end_set:
                del lines[i]
            else:
                end_set.append(end)
                i += 1
            if i >= len(lines):
                break

        for line in lines:
            matches = self.linepat.match(line)
            if not matches:  # for breaths and stuff that don't match the pattern
                return []
            seq = matches.group(2)
            timings = [int(float(t)) for t in
                       matches.group(3, 4, 5, 6, 7)]  # offset, consonant, cutoff, preutterance, overlap

            # For a VCV:
            # --V1-----C2-V2---------C_n
            #   of   ov  p  co   cu
            sylstart = timings[0] + timings[4]

            triphones.append((seq, sylstart))

            if (timings[2] < 0):
                lasttime = timings[0] - timings[2]  # "cutoff"
                triphones.append(('E', lasttime))

        return triphones

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)

def get_hiyori_data(device, use_test_subset=True):
    train_loader = torch.utils.data.DataLoader(HiyoriDataset(), batch_size=32, shuffle=True)
    test_set = HiyoriDataset('data/hiyori/test.ini')
    X_test = test_set.data
    y_test = test_set.labels

    test = (np.array(X_test), np.array(y_test))

    return train_loader, test, test


def get_mnist_data(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=97,  # Using a weird batch size to prevent students from hard-coding
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=97,
        shuffle=True)

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    # Create supervised subset (deterministically chosen)
    # This subset will serve dual purpose of log-likelihood evaluation and
    # semi-supervised learning. Pretty hacky. Don't judge :<
    X = X_test if use_test_subset else X_train
    y = y_test if use_test_subset else y_train

    xl, yl = [], []
    for i in range(10):
        idx = y == i
        idx_choice = get_mnist_index(i, test=use_test_subset)
        xl += [X[idx][idx_choice]]
        yl += [y[idx][idx_choice]]

    if use_test_subset:
        xl = static_binarize(torch.cat(xl)).to(device)
    else:
        xl = torch.cat(xl).to(device)

    yl = torch.cat(yl).to(device)
    yl = yl.new(np.eye(10)[yl.cpu()]).to(device)
    labeled_subset = (xl, yl)
    return train_loader, labeled_subset, (X_test, y_test)


def static_binarize(x):
    # torch.bernoulli seeding behavior is different on CPU v GPU
    # so we'll convert to numpy array and use binomial to sample static x
    with FixedSeed(0):
        x = np.random.binomial(1, x.cpu().numpy())
        x = torch.FloatTensor(x)
    return x


def get_mnist_index(i, test=True):
    # Obviously *hand*-coded
    train_idx = np.array([[2732,2607,1653,3264,4931,4859,5827,1033,4373,5874],
                          [5924,3468,6458,705,2599,2135,2222,2897,1701,537],
                          [2893,2163,5072,4851,2046,1871,2496,99,2008,755],
                          [797,659,3219,423,3337,2745,4735,544,714,2292],
                          [151,2723,3531,2930,1207,802,2176,2176,1956,3622],
                          [3560,756,4369,4484,1641,3114,4984,4353,4071,4009],
                          [2105,3942,3191,430,4187,2446,2659,1589,2956,2681],
                          [4180,2251,4420,4870,1071,4735,6132,5251,5068,1204],
                          [3918,1167,1684,3299,2767,2957,4469,560,5425,1605],
                          [5795,1472,3678,256,3762,5412,1954,816,2435,1634]])

    test_idx = np.array([[684,559,629,192,835,763,707,359,9,723],
                         [277,599,1094,600,314,705,551,87,174,849],
                         [537,845,72,777,115,976,755,448,850,99],
                         [984,177,755,797,659,147,910,423,288,961],
                         [265,697,639,544,543,714,244,151,675,510],
                         [459,882,183,28,802,128,128,53,550,488],
                         [756,273,335,388,617,42,442,543,888,257],
                         [57,291,779,430,91,398,611,908,633,84],
                         [203,324,774,964,47,639,131,972,868,180],
                         [1000,846,143,660,227,954,791,719,909,373]])

    if test:
        return test_idx[i]

    else:
        return train_idx[i]


def get_svhn_data(device):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split='extra', download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    return train_loader, (None, None), (None, None)


def gumbel_softmax(logits, tau, eps=1e-8):
    U = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel
    y = F.softmax(y / tau, dim=1)
    return y


class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)


# System verifier
if sys.version_info[0] < 3:
    raise Exception("Detected unpermitted Python version: Python{}. You should use Python3."
                    .format(sys.version_info[0]))
