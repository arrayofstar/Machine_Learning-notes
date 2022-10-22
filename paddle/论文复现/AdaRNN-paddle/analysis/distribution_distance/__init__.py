from analysis.distribution_distance.adv_loss import adv
from analysis.distribution_distance.coral import CORAL
from analysis.distribution_distance.cos import cosine
from analysis.distribution_distance.kl_js import kl_div, js
from analysis.distribution_distance.mmd import MMD_loss
from analysis.distribution_distance.mutual_info import Mine
from analysis.distribution_distance.pair_dist import pairwise_dist

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]