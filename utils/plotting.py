import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from tqdm import tqdm
from utils.individual_trial_analysis_utils import SessionData, ChoiceAlignedData

class HeatMapParams(object):
    def __init__(self, state_type_of_interest, response, first_choice, last_response, outcome, last_outcome,first_choice_correct, align_to, instance, no_repeats, plot_range):

        self.state = state_type_of_interest
        self.outcome = outcome
        #self.last_outcome = last_outcome
        self.response = response
        self.last_response = last_response
        self.align_to = align_to
        self.other_time_point = np.array(['Time start', 'Time end'])[np.where(np.array(['Time start', 'Time end']) != align_to)]
        self.instance = instance
        self.plot_range = plot_range
        self.no_repeats = no_repeats
        self.first_choice_correct = first_choice_correct
        self.first_choice = first_choice

def get_photometry_around_event(all_trial_event_times, demodulated_trace, pre_window=5, post_window=5, sample_rate=10000):
    num_events = len(all_trial_event_times)
    event_photo_traces = np.zeros((num_events, sample_rate*(pre_window + post_window)))
    for event_num, event_time in enumerate(all_trial_event_times):
        plot_start = int(round(event_time*sample_rate)) - pre_window*sample_rate
        plot_end = int(round(event_time*sample_rate)) + post_window*sample_rate
        if plot_end - plot_start != sample_rate*(pre_window + post_window):
            print(event_time)
            plot_start = plot_start + 1
            print(plot_end - plot_start)
        event_photo_traces[event_num, :] = demodulated_trace[plot_start:plot_end]
        #except:
        #   event_photo_traces = event_photo_traces[:event_num,:] 
    print(event_photo_traces.shape)
    return event_photo_traces

def get_next_centre_poke(trial_data, events_of_int):
    trial_numbers = events_of_int['Trial num'].values
    next_centre_poke_times = []
    for event_trial_num in range(len(trial_numbers)-1):
        trial_num = trial_numbers[event_trial_num]
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
        wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        next_centre_poke_times.append(next_wait_for_poke['Time end'].values[0])
    return next_centre_poke_times
    
    
def get_mean_and_sem(trial_data, demod_signal, params, norm_window=8, sort=False, error_bar_method='sem'):     
    response_names = ['both left and right','left', 'right']
    outcome_names = ['incorrect', 'correct']

    if  params.state == 10:
        omission_events = trial_data.loc[(trial_data['State type'] == params.state)]
        trials_of_int = omission_events['Trial num'].values
        omission_trials_all_states = trial_data.loc[(trial_data['Trial num'].isin(trials_of_int))]
        events_of_int = omission_trials_all_states.loc[(omission_trials_all_states['State type'] == 4)]
    else:
        events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
    if params.response != 0:
        events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
    if params.first_choice != 0:
        events_of_int = events_of_int.loc[events_of_int['First response'] == params.first_choice]
    if params.last_response != 0:
        events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        title = ' last response: ' + response_names[params.last_response]
    else:
        title = response_names[params.response]
    if not params.outcome == 3:
        events_of_int = events_of_int .loc[events_of_int['Trial outcome'] == params.outcome]
    #events_of_int = events_of_int.loc[events_of_int['Last outcome'] == 0]
    
    if params.state ==10 or params.outcome == 3:
        title = title +' ' + 'omission'
    else:
        title = title +' ' + outcome_names[params.outcome]
        
    if params.instance == -1:
        events_of_int = events_of_int.loc[
            (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
    elif params.instance == 1:
        events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
        if params.no_repeats == 1:
            events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]
    if params.first_choice_correct:
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]
        
    event_times = events_of_int[params.align_to].values
    state_name = events_of_int['State name'].values[0]
    last_event = np.asarray(
        np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int)
    next_centre_poke.append(event_times[-1])
    next_centre_poke_norm = next_centre_poke - event_times
    event_photo_traces = get_photometry_around_event(event_times, demod_signal,  pre_window=norm_window, post_window=norm_window)
    norm_traces = stats.zscore(event_photo_traces.T, axis=0)
    
    if len(last_event) != norm_traces.shape[1]:
        last_event = last_event[:norm_traces.shape[1]]
    print(last_event.shape, event_times.shape)
    if sort:
        arr1inds =  last_event.argsort()
        sorted_last_event = last_event [arr1inds[::-1]]
        sorted_traces = norm_traces.T [arr1inds[::-1]]
        sorted_next_poke = next_centre_poke_norm [arr1inds[::-1]]
    else:
        sorted_last_event = last_event 
        sorted_traces = norm_traces.T
        sorted_next_poke = next_centre_poke_norm

    x_vals = np.linspace(-norm_window, norm_window, norm_traces.shape[0], endpoint=True, retstep=False, dtype=None, axis=0)
    y_vals = np.mean(sorted_traces, axis=0)
    if error_bar_method == 'ci':
        sem = bootstrap(sorted_traces, n_boot=1000, ci=95) 
    elif error_bar_method == 'sem':
        sem = np.std(sorted_traces, axis=0)
    print(np.mean(next_centre_poke_norm), np.std(next_centre_poke_norm))
    print(sorted_last_event[-1])
    return x_vals, y_vals, sem, sorted_traces, sorted_last_event, state_name, title, sorted_next_poke


def heat_map_and_mean(aligned_session_data, error_bar_method='sem'):
    fig, axs = plt.subplots(1, ncols=4, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    axs[0].plot(aligned_session_data.ipsi_time_points, aligned_session_data.ipsi_mean_trace, lw=3, color='#3F888F')
    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(aligned_session_data.ipsi_mean_trace, aligned_session_data.ipsi_sorted_traces, error_bar_method=error_bar_method)
        axs[0].fill_between(aligned_session_data.ipsi_time_points, error_bar_lower, error_bar_upper, alpha=0.5, facecolor='#7FB5B5', linewidth=0)

    axs[0].axvline(0, color='k', linewidth=2)
    axs[0].set_xlim(aligned_session_data.ipsi_params.plot_range)
    # axs[0].set_ylim([-1, 2.2])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('z-score')

    heat_im = axs[1].imshow(aligned_session_data.ipsi_sorted_traces, aspect='auto', extent=[-10, 10, aligned_session_data.ipsi_sorted_traces.shape[0], 0], cmap='jet')
    
    axs[1].axvline(0, color='w', linewidth=2)
    axs[1].scatter(aligned_session_data.ipsi_reaction_times, np.arange(aligned_session_data.ipsi_reaction_times.shape[0]) + 0.5, color='w', s=6)
    axs[1].scatter(aligned_session_data.ipsi_sorted_next_poke, np.arange(aligned_session_data.ipsi_sorted_next_poke.shape[0]) + 0.5, color='k', s=6)
    axs[1].tick_params(labelsize=10)
    axs[1].set_xlim(aligned_session_data.ipsi_params.plot_range)
    axs[1].set_ylim([aligned_session_data.ipsi_sorted_traces.shape[0], 0])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Trial number (sorted)')
    vmin = aligned_session_data.ipsi_sorted_traces.min()
    vmax = aligned_session_data.ipsi_sorted_traces.max()
    edge = max(abs(vmin), abs(vmax))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    heat_im.set_norm(norm)
    fig.colorbar(heat_im, ax=axs[1], orientation='horizontal', fraction=.1, label='z-score')
    
    #axs[0].set_ylim(-0.01,0.1)
    plt.show()
    return fig, axs


def multiple_conditions_plot(trial_data, demod_signal, mouse, date, *param_set):
    norm_window = 10
    fig, axs = plt.subplots(1, ncols=1, figsize=(4, 7))
    mean_colours = ['#3F888F', '#CC4F1B', '#7BC17E', '#B39283']
    sem_colours = ['#7FB5B5', '#CC6600', '#A4D1A2', '#D8CFC4']
    legend_list = []

    for param_num, params in enumerate(param_set):
        x_vals, y_vals, sem, sorted_traces, sorted_last_event, state_name, title = get_mean_and_sem(trial_data, demod_signal, params)
        legend_list.append(title)
    
        num_state_types = trial_data['State type'].unique().shape[0]
        
        axs.title.set_text(state_name + ' mean')
        axs.plot(x_vals, y_vals,lw=3,color=mean_colours[param_num], label=title)
        axs.fill_between(x_vals, y_vals-sem, y_vals+sem, alpha=0.5, facecolor=sem_colours[param_num], linewidth=0)

        #for trace_num in range(0,norm_traces.shape[1]):
        #    axs[0].plot(x_vals, norm_traces[:,trace_num],alpha=0.5, color='#7FB5B5', lw=0.2)

        axs.axvline(0, color='k', linewidth=2)
        axs.set_xlim(params.plot_range)
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('z-score')
        axs.set_ylim(-6,6)
    
        axs.legend(frameon=False)
    fig.text(0.06, 0.02, mouse + ' ' + date, fontsize=12)
    return sorted_traces


def calculate_error_bars(mean_trace, data, error_bar_method='sem'):
    if error_bar_method == 'sem':
        std = np.std(data, axis=0)
        lower_bound = mean_trace - std
        upper_bound = mean_trace + std
    elif error_bar_method == 'ci':
        lower_bound, upper_bound = bootstrap(data)
    return lower_bound, upper_bound


def bootstrap(data, n_boot=10000, ci=68):
    """Helper function for lineplot_boot. Bootstraps confidence intervals for plotting time series.

    :param data:
    :param n_boot:
    :param ci:
    :return:
    """
    boot_dist = []
    for i in tqdm(range(int(n_boot)), desc='Bootstrapping...'):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. - ci / 2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. + ci / 2.)
    return s1, s2


class correctData(object):
    def __init__(self, fiber_side, mouse_id, date, trial_data, dff):
        
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.date = date
        
        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]
        
        state_type_of_interest = 4
        outcome = 1
        last_outcome = 0 # NOT USED CURRENLY
        no_repeats = 1
        last_response = 0
        align_to = 'Time start'
        instance = -1
        plot_range = [-2,3]
        first_choice_correct = 1
           
        response = fiber_side_numeric
        first_choice = fiber_side_numeric
        plotting_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome, last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        dff_events_ipsi = heat_map_and_mean(trial_data,dff, plotting_params, mouse_id, date)
        
        self.ipsi_mean_x_vals = dff_events_ipsi[1]
        self.ipsi_mean_y_vals = dff_events_ipsi[2]
        self.ipsi_CI = dff_events_ipsi[3]
        
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]
        response = contra_fiber_side_numeric
        first_choice = contra_fiber_side_numeric
        plotting_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome, last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        dff_events_contra = heat_map_and_mean(trial_data,dff, plotting_params, mouse_id, date)
        
        self.contra_mean_x_vals = dff_events_contra[1]
        self.contra_mean_y_vals = dff_events_contra[2]
        self.contra_CI = dff_events_contra[3]


class cueData(object):
    def __init__(self, fiber_side, mouse_id, date, trial_data, dff):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.date = date

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]

        state_type_of_interest = 3
        outcome = 1
        last_outcome = 0  # NOT USED CURRENLY
        no_repeats = 1
        last_response = 0
        align_to = 'Time start'
        instance = 1
        plot_range = [-2, 3]
        first_choice_correct = 1

        response = fiber_side_numeric
        first_choice = fiber_side_numeric
        plotting_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                        last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        dff_events_ipsi = heat_map_and_mean(trial_data, dff, plotting_params, mouse_id, date)

        self.ipsi_mean_x_vals = dff_events_ipsi[1]
        self.ipsi_mean_y_vals = dff_events_ipsi[2]
        self.ipsi_CI = dff_events_ipsi[3]

        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]
        response = contra_fiber_side_numeric
        first_choice = contra_fiber_side_numeric
        plotting_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                        last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        dff_events_contra = heat_map_and_mean(trial_data, dff, plotting_params, mouse_id, date)

        self.contra_mean_x_vals = dff_events_contra[1]
        self.contra_mean_y_vals = dff_events_contra[2]
        self.contra_CI = dff_events_contra[3]

class rewardData(object):
    def __init__(self, fiber_side, mouse_id, date, trial_data, dff):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.date = date

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]

        state_type_of_interest = 5
        outcome = 1
        last_outcome = 0  # NOT USED CURRENLY
        no_repeats = 1
        last_response = 0
        align_to = 'Time end'
        instance = 1
        plot_range = [-2, 3]
        first_choice_correct = 1

        response = fiber_side_numeric
        first_choice = fiber_side_numeric
        plotting_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                        last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        dff_events_ipsi = heat_map_and_mean(trial_data, dff, plotting_params, mouse_id, date)

        self.ipsi_mean_x_vals = dff_events_ipsi[1]
        self.ipsi_mean_y_vals = dff_events_ipsi[2]
        self.ipsi_CI = dff_events_ipsi[3]

        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]
        response = contra_fiber_side_numeric
        first_choice = contra_fiber_side_numeric
        plotting_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                        last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        dff_events_contra = heat_map_and_mean(trial_data, dff, plotting_params, mouse_id, date)

        self.contra_mean_x_vals = dff_events_contra[1]
        self.contra_mean_y_vals = dff_events_contra[2]
        self.contra_CI = dff_events_contra[3]

        

