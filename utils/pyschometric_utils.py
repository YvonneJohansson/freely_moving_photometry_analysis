
def heat_map_psychometric(trial_data, demod_signal, params):
    num_trial_types = 7
    fig, axs = plt.subplots(2, ncols=num_trial_types, figsize=(25, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    unsorted_traces = []
    x_vals = []
    all_norm_traces =[]
    all_last_events = []
    images = []
    for trial_type in range(1, 4):
        events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == trial_type]
        if params.response != 0:
            events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
        if params.last_response != 0:
            events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        events_of_int = events_of_int.loc[events_of_int['Trial outcome'] == params.outcome]
        # events_of_int = events_of_int.loc[events_of_int['Last outcome'] == params.last_outcome]

        if params.instance == -1:
            events_of_int = events_of_int.loc[
                (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
        elif params.instance == 1:
            events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
            if params.no_repeats == 1:
                events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]

        event_times = events_of_int[params.align_to].values
        state_name = events_of_int['State name'].values[0]
        last_event = np.asarray(
            np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(
                events_of_int[params.align_to].values))

        num_state_types = trial_data['State type'].unique().shape[0]

        event_photo_traces = get_photometry_around_event(event_times, demod_signal, pre_window=5, post_window=5)
        norm_traces = (event_photo_traces.T - np.mean(event_photo_traces, axis=1)) / np.std(event_photo_traces, axis=1)

        # sorts traces based on state duration
        arr1inds = last_event.argsort()
        sorted_last_event = last_event[arr1inds[::-1]]
        sorted_traces = norm_traces.T[arr1inds[::-1]]

        all_last_events.append(sorted_last_event)
        unsorted_traces.append(sorted_traces)
        all_norm_traces.append(norm_traces)
        x_vals.append(np.linspace(-5, 5, norm_traces.shape[0], endpoint=True, retstep=False, dtype=None, axis=0))

    for trial_type in range(num_trial_types):
        axs[0, trial_type].title.set_text(state_name + ' mean')
        y_vals = np.mean(all_norm_traces[trial_type], axis=1)
        sem = np.std(all_norm_traces[trial_type], axis=1)

        axs[0, trial_type].plot(x_vals[trial_type], y_vals, lw=3, color='#3F888F')
        axs[0, trial_type].fill_between(x_vals[trial_type], y_vals - sem, y_vals + sem, alpha=0.5, facecolor='#7FB5B5',
                                        linewidth=0)

        # for trace_num in range(0,all_norm_traces[trial_type].shape[1]):
        #    axs[0, trial_type].plot(x_vals[trial_type], all_norm_traces[trial_type][:,trace_num], alpha=0.5, color='b', lw=0.2)
        # axs[0, trial_type].plot(x_vals[trial_type], (np.mean(all_norm_traces[trial_type], axis=1)), lw=3,color='k')
        axs[0, trial_type].axvline(0, color='r', linewidth=2)
        axs[0, trial_type].set_xlim(params.plot_range)

        images.append(axs[1, trial_type].imshow(unsorted_traces[trial_type], aspect='auto',
                                                extent=[-5, 5, unsorted_traces[trial_type].shape[0], 0], cmap='jet'))
        axs[1, trial_type].axvline(0, color='k', linewidth=2)
        axs[1, trial_type].scatter(all_last_events[trial_type], np.arange(all_last_events[trial_type].shape[0]) + 0.5,
                                   color='k', s=1)
        axs[1, trial_type].tick_params(labelsize=10)
        axs[1, trial_type].title.set_text(state_name + ' heatmap')
        axs[1, trial_type].set_xlim(params.plot_range)
        # axs[1, trial_type].set_ylim([unsorted_traces[trial_type].shape[0], 0])

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    edge = max(abs(vmin), abs(vmax))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

    for trial_type in range(num_trial_types):
        axs[0, trial_type].set_ylim(-4 * max(np.std(all_norm_traces[trial_type], axis=1)),
                                    4 * max(np.std(all_norm_traces[trial_type], axis=1)))
    return unsorted_traces


def get_photometry_around_event_trial_only(all_trial_event_times, demodulated_trace, trial_starts, trial_ends,
                                           pre_window=10, post_window=10, sample_rate=10000):
    num_events = len(all_trial_event_times)
    num_samples = sample_rate * (pre_window + post_window)
    event_photo_traces = np.full((num_events, num_samples), np.nan)
    for event_num, event_time in enumerate(all_trial_event_times):
        pre_event = int((event_time - trial_starts[event_num]) * sample_rate)
        post_event = int((trial_ends[event_num] - event_time) * sample_rate)
        if pre_event < (pre_window * sample_rate):
            start_ind = int(num_samples / 2 - pre_event)
            plot_start = int(event_time * sample_rate - pre_event)
        else:
            start_ind = int(num_samples / 2 - pre_window * sample_rate)
            plot_start = int(event_time * sample_rate - pre_window * sample_rate)
        if post_event < (post_window * sample_rate):
            end_ind = int(num_samples / 2 + post_event)
            plot_end = int(event_time * sample_rate + post_event)
        else:
            end_ind = int(num_samples / 2 + post_window * sample_rate)
            plot_end = int(event_time * sample_rate + post_window * sample_rate)

        event_photo_traces[event_num, start_ind:end_ind] = demodulated_trace[plot_start:plot_end]
    norm_traces = (event_photo_traces.T - np.nanmean(event_photo_traces, axis=1)).T
    fig, ax = plt.subplots(1, ncols=2, figsize=(25, 15))
    ax[0].plot((np.nanmean(norm_traces, axis=0)))
    heatmap = ax[1].imshow(norm_traces, aspect='auto', extent=[-10, 10, len(list(event_times)), 0],
                           vmin=np.nanmin(norm_traces), vmax=np.nanmax(norm_traces))
    ax[1].axvline(0, color='w', linewidth=2)
