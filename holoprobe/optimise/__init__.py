from .cavi_online_spike_and_slab import cavi_online_spike_and_slab
from .cavi_offline_spike_and_slab import cavi_offline_spike_and_slab

from jax.config import config
config.update("jax_enable_x64", True)