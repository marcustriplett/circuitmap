import numpy as np
import functools
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy as jsp
import circuitmap as cm

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

jax.config.update('jax_platform_name', 'cpu')

def _photocurrent_shape(
    O_inf, R_inf, tau_o, tau_r, g,  # shape params
    t_on, t_off,  # timing params
    linear_onset, # boolean, whether or not we use a linear onset
    t=None,
    time_zero_idx=None,
    O_0=0.0, R_0=1.0,
    window_len=900,
    msecs_per_sample=0.05,
    conv_window_len=25,
):

    # In order to correctly handle stim times which start at t < 0,
    # we need to work with a larger window and then crop later.
    # We also pad the window by conv_window_len to avoid zero
    # padding issues during the convolution step.
    # left_bound = jnp.minimum(0, t_on / msecs_per_sample)
    # right_bound = jnp.abs(left_bound) + window_len + conv_window_len
    # t = jnp.arange(left_bound, right_bound) * msecs_per_sample

    # # get the index where t=0 occurs. This is the beginning of the
    # # window we'll return to the user.
    # time_zero_idx = int(-jnp.minimum(t_on / msecs_per_sample, 0))

    mask_stim_on = jnp.where((t >= t_on) & (t <= t_off), 1, 0)
    mask_stim_off = jnp.where((t > t_off), 1, 0)

    # get index where stim is off
    index_t_off = time_zero_idx + jnp.array(t_off // msecs_per_sample, dtype=int)    

    O_on = mask_stim_on * (O_inf - (O_inf - O_0) *
                           jnp.exp(- (t - t_on)/(tau_o)))
    O_off = mask_stim_off * O_on[index_t_off] * jnp.exp(-(t - t_off)/tau_o)

    R_on = mask_stim_on * (R_inf - (R_inf - R_0) * jnp.exp(-(t - t_on)/tau_r))
    R_off = mask_stim_off * \
        (1 - (1 - R_on[index_t_off]) * jnp.exp(-(t - t_off)/tau_r))

    # form photocurrent from each part
    i_photo = g * (O_on + O_off) * (R_on + R_off)


    # if linear_onset=True, use a different version of i_photo with the rising
    # period replaced.
    i_photo_linear = jnp.copy(i_photo)
    stim_off_val = i_photo_linear[index_t_off]
    # zero out the current during the stim
    i_photo_linear = i_photo_linear - (i_photo * mask_stim_on)
    # add linear onset back in
    i_photo_linear = i_photo_linear + ((t - t_on) / (t - t_on)[index_t_off] * stim_off_val) * mask_stim_on

    # conditionally replace i_photo
    i_photo = jax.lax.cond(
        linear_onset,
        lambda _:i_photo_linear,
        lambda _:i_photo,
        None,
    )
        
    # convolve with gaussian to smooth
    x = jnp.linspace(-3, 3, conv_window_len)
    window = jsp.stats.norm.pdf(x, scale=25)
    i_photo = jsp.signal.convolve(i_photo, window, mode='same')
    i_photo /= (jnp.max(i_photo) + 1e-3)

    return (i_photo[time_zero_idx:time_zero_idx + window_len],
            O_on[time_zero_idx:time_zero_idx + window_len],
            O_off[time_zero_idx:time_zero_idx + window_len],
            R_on[time_zero_idx:time_zero_idx + window_len],
            R_off[time_zero_idx:time_zero_idx + window_len])

photocurrent_shape = jax.jit(_photocurrent_shape, static_argnames=('time_zero_idx'))

def _sample_photocurrent_params(key,
    t_on_min=5.0,
    t_on_max=7.0,
    t_off_min=10.0,
    t_off_max=11.0,
    O_inf_min=0.3,
    O_inf_max=1.0,
    R_inf_min=0.3,
    R_inf_max=1.0,
    tau_o_min=5,
    tau_o_max=14,
    tau_r_min=25,
    tau_r_max=30,):
    keys = jrand.split(key, num=6)

    t_on  = jrand.uniform(keys[0], minval=t_on_min, maxval=t_on_max)
    t_off  = jrand.uniform(keys[1], minval=t_off_min, maxval=t_off_max)
    O_inf = jrand.uniform(keys[2], minval=O_inf_min, maxval=O_inf_max)
    R_inf = jrand.uniform(keys[3], minval=R_inf_min, maxval=R_inf_max)
    tau_o = jrand.uniform(keys[4], minval=tau_o_min, maxval=tau_o_max)
    tau_r = jrand.uniform(keys[5], minval=tau_r_min, maxval=tau_r_max)
    g = 1.0

    return O_inf, R_inf, tau_o, tau_r, g, t_on, t_off,


def _sample_scales(key, min_pc_fraction, max_pc_fraction,
                   num_traces, min_pc_scale, max_pc_scale):
    # sample scale values for photocurrents.
    # Randomly set some traces to have no photocurrent
    # according to pc_fraction.

    pc_fraction = jrand.uniform(key, minval=min_pc_fraction,
                                maxval=max_pc_fraction)
    key = jrand.fold_in(key, 0)
    pc_mask = jnp.where(
        jrand.uniform(key, shape=(num_traces,)) <= pc_fraction,
        1.0,
        0.0)
    key = jrand.fold_in(key, 0)
    pc_scales = jrand.uniform(key, shape=(num_traces,),
                              minval=min_pc_scale, maxval=max_pc_scale)
    pc_scales *= pc_mask
    return pc_scales


def _sample_gp(key, pcs, gp_lengthscale=25, gp_scale=0.01, ):
    n_samples, trial_dur = pcs.shape
    # creates a distance matrix between indices,
    # much faster than a loop
    D = jnp.broadcast_to(jnp.arange(trial_dur), (trial_dur, trial_dur))
    D -= jnp.arange(trial_dur)[:, None]
    D = jnp.array(D, dtype=jnp.float64)
    K = jnp.exp(-D**2/(2 * gp_lengthscale**2)) + 1e-4 * jnp.eye(trial_dur)
    mean = jnp.zeros(trial_dur, dtype=jnp.float64)
    return gp_scale * jrand.multivariate_normal(key, mean=mean, cov=K, shape=(n_samples,))
sample_gp = jax.jit(_sample_gp)


@jax.jit
def _sample_experiment_noise_and_scales(
    key,
    cur_pc_template,
    prev_pc_template,
    next_pc_template,
    psc_background,
    min_pc_scale,
    max_pc_scale,
    min_pc_fraction,
    max_pc_fraction,
    prev_pc_fraction,
    gp_lengthscale,
    gp_scale,
    iid_noise_scale,
):
    num_traces, trial_dur = psc_background.shape

    keys = jrand.split(key, num=4)
    prev_pcs = _sample_scales(keys[0],
                              prev_pc_fraction, prev_pc_fraction,
                              num_traces, min_pc_scale, max_pc_scale)[:, None] * prev_pc_template

    cur_pcs = _sample_scales(keys[1],
                             min_pc_fraction, max_pc_fraction,
                             num_traces, min_pc_scale, max_pc_scale)[:, None] * cur_pc_template

    next_pcs = _sample_scales(keys[2],
                              min_pc_fraction, max_pc_fraction,
                              num_traces, min_pc_scale, max_pc_scale)[:, None] * next_pc_template

    # TODO: add GP and IID noise
    gp_noise = sample_gp(
        keys[3],
        cur_pcs,
        gp_lengthscale=gp_lengthscale,
        gp_scale=gp_scale,
    )

    iid_noise = iid_noise_scale * \
        jrand.normal(keys[3], shape=(num_traces, trial_dur))
    targets = cur_pcs
    observations = prev_pcs + cur_pcs + next_pcs + iid_noise + gp_noise + psc_background

    return observations, targets

def _default_pc_shape_params():
    return dict(
        # shape params
        O_inf_min=0.3,
        O_inf_max=1.0,
        R_inf_min=0.3,
        R_inf_max=1.0,
        tau_o_min=3,
        tau_o_max=8,
        tau_r_min=25,
        tau_r_max=30,
    )


def _exp_func(t, a, b, c):
    ''' Exponential function
    '''
    return a * np.exp(b * t) + c


def _fit_exponential_tail(trace, t, a0, b0, c0):
    '''
    Fit exponentials to the provided traces.
    params:
        traces: N x T array
    returns:
        a, b, c: length N arrays of parameters, such that the exponential
                 a[i] * exp(b[i] * t) + c[i] ~= traces[i] for i = 1,...,N
    '''
    popt, pcov = curve_fit(
        _exp_func,
        t, trace,
        p0=(a0, b0, c0)
    )
    return popt


def _extend_traces(
    traces,
    msecs_per_sample,
    num_samples_to_add,
    fit_start_idx=400,
    replace_start_idx=600,
    a0=0.1,
    b0=-1.0/20.0,
    c0=0.5
    ):
    N, window_len = traces.shape
    t_fit = np.arange(fit_start_idx, window_len) * msecs_per_sample
    params = [_fit_exponential_tail(trace, t_fit, a0, b0, c0)
        for trace in traces[:, fit_start_idx:]]
    
    # create decaying exponentials of length num_samples_to_add
    t_new = np.arange(replace_start_idx, window_len + num_samples_to_add) \
         * msecs_per_sample
    extensions = np.array([_exp_func(t_new, *popt) for popt in params])

    # concatenate traces with estimated tails
    out = np.zeros((N, window_len + 2 * num_samples_to_add))
    out[:, num_samples_to_add:num_samples_to_add + replace_start_idx] = traces[:, 0:replace_start_idx]
    out[:, num_samples_to_add + replace_start_idx:] = extensions

    # extend beginning of trace by appending constant
    out[:, 0:num_samples_to_add] = traces[:,0:1]

    return out


def sample_from_templates(
    templates,
    key,
    size=100,
    jitter_ms=0.5,
    window_len=900,
    smoothing_sigma=5,
    max_scaling_frac=0.5,
    msecs_per_sample=0.05,
    stim_start=100,
    exponential_fit_start_idx=450,
    add_target_gp=False,
    target_gp_lengthscale=50,
    target_gp_scale=0.01,
    ):
    '''
    sample traces from templates with augmentation by jitter and scaling
    '''

    templates[:, 0:stim_start] = 0.0

    # extend templates so that we can sample using jitter
    num_samples_to_add =  int(np.round(jitter_ms / msecs_per_sample))
    extended_traces = _extend_traces(
        templates,
        msecs_per_sample,
        num_samples_to_add,
        exponential_fit_start_idx,
    )
    extended_traces_smoothed = gaussian_filter1d(
        extended_traces, sigma=smoothing_sigma)
    
    out = np.zeros((size, templates.shape[-1]))
    for i in range(size):
        this_template_idx = np.random.randint(templates.shape[0])
        this_template = np.copy(extended_traces_smoothed[this_template_idx])
        this_scale = 1.0 + np.random.uniform(low=-max_scaling_frac, high=max_scaling_frac)
        this_template *= this_scale

        # sample jitter in number of samples to shift
        this_jitter_samples = np.random.randint(low=0, high=num_samples_to_add)
        start_idx = num_samples_to_add + this_jitter_samples
        out[i] = this_template[start_idx:start_idx + window_len]

    if add_target_gp:
        stim_start_idx = int(stim_start // msecs_per_sample)
        key = jrand.fold_in(key, 0)
        target_gp = np.array(_sample_gp(
            key, 
            out,
            gp_lengthscale=target_gp_lengthscale,
            gp_scale=target_gp_scale,
        ))
        target_gp = np.maximum(0, target_gp)
        out = np.array(out)
        out[:, stim_start_idx+10:] += target_gp[:, stim_start_idx+10:]
        out = cm.neural_waveform_demixing._monotone_decay_filter(
            out,
            inplace=True,
        )

    return out


def sample_photocurrent_shapes(
        key, num_expts,
        onset_jitter_ms=2.0,
        onset_latency_ms=0.2,
        msecs_per_sample=0.05,
        stim_start=5.0,
        tstart=-10.0,
        tend=45.0,
        time_zero_idx: int = 200,
        pc_shape_params=None,
        linear_onset_frac=0.5,
        add_target_gp=True,
        target_gp_lengthscale=50,
        target_gp_scale=0.01,
    ):
    keys = jrand.split(key, num=num_expts)
    if pc_shape_params is None:
        pc_shape_params = _default_pc_shape_params()

    # generate all photocurrent templates.
    # We create a separate function to sample each of previous, current, and
    # next PSC shapes.
    prev_pc_params = jax.vmap(
        functools.partial(
            _sample_photocurrent_params,
            **pc_shape_params,
            t_on_min=-7.0 + onset_latency_ms, t_on_max=-7.0 + onset_latency_ms + onset_jitter_ms,
            t_off_min=-2.0 + onset_latency_ms, t_off_max=-2.0 + onset_latency_ms + onset_jitter_ms,
        )
    )(keys)
    curr_pc_params = jax.vmap(
        functools.partial(
            _sample_photocurrent_params,
            **pc_shape_params,
            t_on_min=5.0 + onset_latency_ms, t_on_max=5.0 + onset_latency_ms + onset_jitter_ms,
            t_off_min=10.0 + onset_latency_ms, t_off_max=10.0 + onset_latency_ms + onset_jitter_ms,
        )
    )(keys)
    next_pc_params = jax.vmap(
        functools.partial(
            _sample_photocurrent_params,
            **pc_shape_params,
            t_on_min=38.0 + onset_latency_ms, t_on_max=38.0 + onset_latency_ms + onset_jitter_ms,
            t_off_min=43.0 + onset_latency_ms, t_off_max=43.0 + onset_latency_ms + onset_jitter_ms,
        )
    )(keys)
    
    # form boolean for each trace deciding whether to use linear onset
    key = jrand.fold_in(key, 0)
    linear_onset_bools = jrand.uniform(key, shape=(num_expts,)) > linear_onset_frac

    # Note that we simulate using a longer window than we'll eventually use.
    time = jnp.arange(tstart / msecs_per_sample, tend / msecs_per_sample) * msecs_per_sample
    batched_photocurrent_shape = jax.vmap(
        functools.partial(
            photocurrent_shape,
            t=time,
            time_zero_idx=time_zero_idx,
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
    )
    prev_pc_shapes = batched_photocurrent_shape(*prev_pc_params, linear_onset_bools)[0]
    curr_pc_shapes = batched_photocurrent_shape(*curr_pc_params, linear_onset_bools)[0]
    next_pc_shapes = batched_photocurrent_shape(*next_pc_params, linear_onset_bools)[0]

    # Add variability to target waveforms to account for mis-specification of 
    # photocurrent model.
    if add_target_gp:
        stim_start_idx = int(stim_start // msecs_per_sample)
        key = jrand.fold_in(key, 0)
        target_gp = np.array(_sample_gp(
            key, 
            curr_pc_shapes,
            gp_lengthscale=target_gp_lengthscale,
            gp_scale=target_gp_scale,
        ))
        target_gp = np.maximum(0, target_gp)
        curr_pc_shapes = np.array(curr_pc_shapes)
        curr_pc_shapes[:, stim_start_idx+10:] += target_gp[:, stim_start_idx+10:]
        curr_pc_shapes = cm.neural_waveform_demixing._monotone_decay_filter(
            curr_pc_shapes,
            inplace=True,
        )

    return prev_pc_shapes, curr_pc_shapes, next_pc_shapes


def sample_photocurrent_expts_batch(
    key, num_expts, num_traces_per_expt, trial_dur,
    pc_scale_range=(0.05, 2.0),
    onset_jitter_ms=1.0,
    onset_latency_ms=0.2,
    pc_shape_params=None,
    iid_noise_scale_range=(0.01, 0.05),
    gp_scale_range=(0.01, 2.0),
    min_pc_scale = 0.05,
    min_pc_fraction = 0.1,
    max_pc_fraction = 0.9,
    prev_pc_fraction = 0.1,
    gp_lengthscale = 50,
    add_target_gp=True,
    target_gp_lengthscale=25.0,
	target_gp_scale=0.01,
    linear_onset_frac=0.5
    ):

    if pc_shape_params is None:
        pc_shape_params = _default_pc_shape_params()

    # generate all photocurrent templates.
    # We create a separate function to sample each of previous, current, and
    # next PSC shapes.
    prev_pc_shapes, curr_pc_shapes, next_pc_shapes = \
			sample_photocurrent_shapes(
				key,
				num_expts,
				onset_jitter_ms=onset_jitter_ms,
				onset_latency_ms=onset_latency_ms,
				pc_shape_params=pc_shape_params,
				add_target_gp=add_target_gp,
				target_gp_lengthscale=target_gp_lengthscale,
				target_gp_scale=target_gp_scale,
				linear_onset_frac=linear_onset_frac)
    key = jax.random.fold_in(key, 0)

    # Generate all psc traces from neural demixer.
    # This is faster than calling it separately many times.
    # Here, we generate noiseless PSC traces, since we will add noise
    # the summed traces (PCs + PSCs) at once.
    demixer = cm.NeuralDemixer()
    demixer.generate_training_data(
        size=num_expts * num_traces_per_expt,
        training_fraction=1.0,
        noise_std_upper=0.0,
        noise_std_lower=0.0,
        gp_scale=0.0,
    )
    pscs, _ = demixer.training_data
    pscs_batched = pscs.reshape(num_expts, num_traces_per_expt, trial_dur)

    sample_experiment_noise_and_scales_batch = jax.vmap(
        _sample_experiment_noise_and_scales,
        in_axes=(0, 0, 0, 0, 0, None, 0, None, None, None, None, 0, 0),
    )

    # mimic varying opsin / noise levels by experiment:
    # each experiment will have a different maximum photocurrent scale.
    key = jrand.fold_in(key, 0)
    max_pc_scales = jrand.uniform(
        key, minval=pc_scale_range[0], maxval=pc_scale_range[1],
        shape=(num_expts,)
    )
    key = jrand.fold_in(key, 0)
    gp_scales = jrand.uniform(
        key, minval=gp_scale_range[0], maxval=gp_scale_range[1],
        shape=(num_expts,)
    )
    key = jrand.fold_in(key, 0)
    iid_noise_scales = jrand.uniform(
        key, minval=iid_noise_scale_range[0], maxval=iid_noise_scale_range[1],
        shape=(num_expts,)

    )
    
    keys = jrand.split(key, num=num_expts)
    return sample_experiment_noise_and_scales_batch(
        keys,
        curr_pc_shapes,
        prev_pc_shapes,
        next_pc_shapes,
        pscs_batched,
        min_pc_scale,
        max_pc_scales,
        min_pc_fraction,
        max_pc_fraction,
        prev_pc_fraction,
        gp_lengthscale,
        gp_scales,
        iid_noise_scales,
    )