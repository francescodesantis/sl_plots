import logging
from bisect import bisect_left
from collections import defaultdict
from contextlib import ExitStack
from itertools import batched
import math
from math import ceil
from pathlib import PurePath
from typing import Iterable, List

import brian2 as b2
from brian2 import Hz
import brian2hears as b2h
from brian2hears import erbspace
import dill
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sorcery import dict_of

# from analyze import sound_analysis as SA
# from cochleas.hrtf_utils import run_hrtf

from cochleas.consts import CFMAX, CFMIN
from utils.custom_sounds import Tone, ToneBurst

plt.rcParams["axes.grid"] = True
plt.rcParams['axes.titlesize']= 'large'
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.spines.top']= False


def flatten(items):
    """Yield items from any nested iterable.
    from https://stackoverflow.com/a/40857703
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return (myList[0], 0)
    if pos == len(myList):
        return (myList[-1], len(myList))
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return (after, pos)
    else:
        return (before, pos - 1)

def avg_fire_rate_actv_neurons(x):
    active_neurons = set(x["senders"])
    return (len(x["times"]) / len(active_neurons)) if len(active_neurons) > 0 else 0

def firing_neurons_distribution(x):
    "returns {neuron_id: num_spikes}.keys()"
    n2s = {id: 0 for id in x["global_ids"]}
    for sender in x["senders"]:
        n2s[sender] += 1
    return n2s.values()

def shift_senders(x, hist_logscale=False):
    "returns list of 'senders' with ids shifted to [0,num_neurons]. optionally ids are CFs"
    if hist_logscale:
        cf = b2h.erbspace(CFMIN, CFMAX, len(x["global_ids"])) / b2.Hz
        old2newid = {oldid: cf[i] for i, oldid in enumerate(x["global_ids"])}
    else:
        old2newid = {oldid: i for i, oldid in enumerate(x["global_ids"])}
    return [old2newid[i] for i in x["senders"]]

def draw_hist(
    ax,
    senders_renamed,
    angles,
    num_neurons,
    max_spikes_single_neuron,
    logscale=True,
    freq=None,
):
    """draws a low opacity horizontal histogram for each angle position

    includes a secondary y-axis, optionally logarithmic.
    if logscale, expects senders to be renamed to CFs
    if freq, include a horizontal line at corresponding frequency
    """
    max_histogram_height = 0.25
    bin_count = 50
    alpha = 0.5
    freqlinestyle = {
        "color": "black",
        "linestyle": ":",
        "label": "freq_in",
        "alpha": 0.4,
    }
    if logscale:
        bins = b2h.erbspace(CFMIN, CFMAX, bin_count) / b2.Hz

        for j, angle in enumerate(angles):
            left_data = senders_renamed["L"][j]
            right_data = senders_renamed["R"][j]

            left_hist, _ = np.histogram(left_data, bins=bins)
            right_hist, _ = np.histogram(right_data, bins=bins)
            max_value = max(max(left_hist), max(right_hist))
            left_hist_normalized = left_hist / (max_value * max_histogram_height)
            right_hist_normalized = right_hist / (max_value * max_histogram_height)

            ax.barh(
                bins[:-1],
                -left_hist_normalized,
                height=np.diff(bins),  # bins have different sizes
                left=angle,
                color="m",
                alpha=alpha,
                align="edge",
            )
            ax.barh(
                bins[:-1],
                right_hist_normalized,
                height=np.diff(bins),
                left=angle,
                color="g",
                alpha=alpha,
                align="edge",
            )
        ax.set_yscale("log")
        ax.set_ylim(CFMIN, CFMAX)
        yticks = [20, 100, 500, 1000, 5000, 10000, 20000]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{freq} Hz" for freq in yticks])
        if freq is not None:
            ax.axhline(y=freq / b2.Hz, **freqlinestyle)
        ax.set_ylabel("approx CF (Hz)")
    else:
        bins = np.linspace(0, num_neurons, bin_count)

        for j, angle in enumerate(angles):
            left_data = senders_renamed["L"][j]
            right_data = senders_renamed["R"][j]

            left_hist, _ = np.histogram(left_data, bins=bins)
            right_hist, _ = np.histogram(right_data, bins=bins)
            left_hist_normalized = (
                left_hist / max_spikes_single_neuron * max_histogram_height
            )
            right_hist_normalized = (
                right_hist / max_spikes_single_neuron * max_histogram_height
            )

            ax.barh(
                bins[:-1],
                -left_hist_normalized,
                height=num_neurons / bin_count,
                left=angle,
                color="C0",
                alpha=alpha,
                align="edge",
            )
            ax.barh(
                bins[:-1],
                right_hist_normalized,
                height=num_neurons / bin_count,
                left=angle,
                color="C1",
                alpha=alpha,
                align="edge",
            )
        ax.set_ylabel("neuron id")
        ax.set_ylim(0, num_neurons)

        if freq is not None:
            cf = b2h.erbspace(CFMIN, CFMAX, num_neurons)
            freq, neur_n = take_closest(cf, freq)
            ax.axhline(y=neur_n)
    ax.yaxis.set_minor_locator(plt.NullLocator())  # remove minor ticks

def draw_single_angle_histogram(data, angle, population="SBC", fontsize=16, alpha=0.8):
    """
    Draw horizontal histograms of spike distributions across frequencies for a single angle,
    with left population growing downward and right population growing upward from a central axis.

    Parameters:
    -----------
    data : dict
        The full dataset containing angle_to_rate information
    angle : float
        The specific angle to visualize
    population : str
        Name of the neural population to visualize
    fontsize : int
        Base fontsize for the plot. Other elements will scale relative to this.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Constants
    bin_count = 50

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 2.42))

    # Get data for this angle and population
    pop_data = {
        "L": data["angle_to_rate"][angle]["L"][population],
        "R": data["angle_to_rate"][angle]["R"][population],
    }

    # Create logarithmic bins for frequency
    bins = b2h.erbspace(CFMIN, CFMAX, bin_count) / b2.Hz

    # Process data for histograms
    senders_renamed = {
        side: shift_senders(pop_data[side], True)  # True for logscale
        for side in ["L", "R"]
    }

    # Create histograms
    left_hist, _ = np.histogram(senders_renamed["L"], bins=bins)
    right_hist, _ = np.histogram(senders_renamed["R"], bins=bins)

    # Normalize histograms
    max_value = max(max(left_hist), max(right_hist))
    if max_value > 0:  # Avoid division by zero
        left_hist = left_hist / max_value
        right_hist = right_hist / max_value

    # Plot histograms - note the negative values for left histogram
    ax.bar(
        bins[:-1],
        -left_hist,
        width=np.diff(bins),
        color="C0",
        alpha=alpha,
        label="Left",
        align="edge",
    )
    ax.bar(
        bins[:-1],
        right_hist,
        width=np.diff(bins),
        color="C1",
        alpha=alpha,
        label="Right",
        align="edge",
    )

    # Configure axes
    ax.set_xscale("log")
    ax.set_xlim(CFMIN, CFMAX)
    xticks = [20, 100, 500, 1000, 5000, 20000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{freq} Hz" for freq in xticks], fontsize=fontsize)

    # Set y-axis limits symmetrically around zero
    ylim = 1.1  # Slightly larger than 1 to give some padding
    ax.set_ylim(-ylim, ylim)
    ax.tick_params(axis="y", labelsize=fontsize)  # Set y-tick font size

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Set font sizes for labels and title
    ax.set_xlabel("Characteristic Frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel("Normalized spikes", fontsize=fontsize)
    # ax.legend(fontsize=fontsize)

    # plt.title(
    #     f"{population} population response at {angle}° azimuth\n"
    #     f'Sound: {data["conf"]["sound_key"]}',
    #     fontsize=fontsize * 1.2,
    # )  # Title slightly larger

    plt.tight_layout()

    return fig

def synthetic_angle_to_itd(angle, w_head: int = 22, v_sound: int = 33000):
    delta_x = w_head * np.sin(np.deg2rad(angle))
    return round(1000 * delta_x / v_sound, 2)

def get_spike_phases(spike_times: np.ndarray, frequency: float) -> np.ndarray:
    times_sec = spike_times
    return 2 * np.pi * frequency * (times_sec % (1 / frequency))

def calculate_vector_strength(spike_times: np.ndarray, frequency: float) -> float:
    if len(spike_times) == 0:
        return 0
    phases = get_spike_phases(spike_times, frequency)
    x = np.mean(np.cos(phases))
    y = np.mean(np.sin(phases))
    return np.sqrt(x**2 + y**2)
    
def range_around_center(center, radius, min_val=0, max_val=np.iinfo(np.int64).max):
    start = max(min_val, center - radius)
    end = min(max_val + 1, center + radius + 1)
    return np.arange(start, end)

# def draw_ITD_ILD(data):
#     previous_level = logger.level
#     # itd and ild functions are VERY verbose
#     tone: Tone = data["basesound"]
#     angle_to_ild = {}
#     angle_to_itd = {}
#     angles = list(data["angle_to_rate"].keys())
#     coc = data["conf"]["cochlea_type"]
#     angle_to_hrtfed_sound = data.get("angle_to_hrtfed_sound", None)
#     if coc != "ppg":
#         logger.debug(
#             f"used cochlea {coc}, with parameters {
#                 data["conf"]["parameters"]["cochlea"][coc]["hrtf_params"]
#             }"
#         )
#         for angle in angles:
#             if angle_to_hrtfed_sound is None:
#                 logger.info(
#                     "old result file, does not include HRTFed sounds. Generating (beware of possible differences)..."
#                 )
#                 binaural_sound = run_hrtf(
#                     tone,
#                     angle,
#                     data["conf"]["parameters"]["cochlea"][coc]["hrtf_params"],
#                 )
#                 left = binaural_sound.left
#                 right = binaural_sound.right
#             else:
#                 left, right = (
#                     angle_to_hrtfed_sound[angle]["left"],
#                     angle_to_hrtfed_sound[angle]["right"],
#                 )
#             logger.setLevel(logging.WARNING)
#             angle_to_itd[angle] = SA.itd(left, right)
#             ild_res, all_freq_diff = SA.ild(left, right, tone.sound)
#             logger.setLevel(logging.DEBUG)
#             angle_to_ild[angle] = ild_res

#             # total_diff = np.sum(all_freq_diff)
#     else:
#         angle_to_itd = {angle: synthetic_angle_to_itd(angle) for angle in angles}
#         angle_to_ild = {angle: 0 for angle in angles}

#     fig, ild = plt.subplots(1, sharex=True, figsize=(10, 2.3))
#     fig.suptitle(
#         f"diff = RMS(left)-RMS(right), freq={tone.frequency}"
#         # f"diff = max(|spectrum(left)|)-max(|spectrum(right)|), freq={tone.frequency}"
#     )

#     ild.set_ylabel("Level diff (dB)", color="r")
#     ild.plot(
#         angles,
#         [angle_to_ild[angle] for angle in angles],
#         label="ILD",
#         marker=".",
#         color="r",
#     )
#     ild.tick_params(axis="y", labelcolor="r")
#     itd = ild.twinx()
#     itd.set_ylabel("seconds", color="b")
#     itd.plot(
#         angles,
#         [angle_to_itd[angle] for angle in angles],
#         label="ITD",
#         marker=".",
#         color="b",
#     )
#     itd.tick_params(axis="y", labelcolor="b")
#     _ = fig.legend()

#     fig.tight_layout()
#     # plt.subplots_adjust(hspace=0.6, wspace=1)
#     plt.setp([ild, itd], xticks=angles)
#     logger.setLevel(previous_level)
#     return fig

def calculate_vector_strength_from_result(
        # result file (loaded)
        res,
        angle,
        side,
        pop,
        freq = None, # if None: freq = res['basesound'].frequency
        color = None,
        cf_target = None,
        bandwidth=0,
        n_bins = 7,
        figsize = (7,5),
        display=False # if True also return fig, show() in caller function
        ):
    
    spikes = res["angle_to_rate"][angle][side][pop]
    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        sender2times[sender].append(time)
    sender2times = {k: np.array(v) / 1000 for k, v in sender2times.items()}
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)

    if(freq == None):
        if(type(res['basesound'])  in (Tone,ToneBurst)):
            freq = res['basesound'].frequency
        else:
            print("Frequency needs to be specified for non-Tone sounds")
    else:
        freq = freq * Hz

    if(cf_target == None):    
        cf_neuron, center_neuron_for_freq = take_closest(cf, freq)
    else:
        cf_neuron, center_neuron_for_freq = take_closest(cf, cf_target *Hz)

    old2newid = {oldid: i for i, oldid in enumerate(spikes["global_ids"])}
    new2oldid = {v: k for k, v in old2newid.items()}

    relevant_neurons = range_around_center(
        center_neuron_for_freq, radius=bandwidth, max_val=num_neurons - 1
    )
    relevant_neurons_ids = [new2oldid[i] for i in relevant_neurons]

    spike_times_list = [sender2times[i] for i in relevant_neurons_ids]  
    spike_times_array = np.concatenate(spike_times_list)  # Flatten into a single array

    phases = get_spike_phases(
        spike_times= spike_times_array, frequency=freq / Hz
    )
    vs = calculate_vector_strength(
        spike_times=spike_times_array, frequency=freq / Hz
    )


    if not display:
        return (vs)
    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'
        else: color = 'k'
    # plot phases
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hist1, _ = np.histogram(phases, bins=bins)
    ax.bar(bin_centers, hist1, width=2 * np.pi / n_bins, alpha=0.7, color = color)
    # if(bandwidth == 0):
    #     ax.set_title(
    #         f"Neuron {relevant_neurons_ids[0]} (CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}"
    #     )
    # else:
    #     ax.set_title(
    #         f"Neurons {relevant_neurons_ids[0]} : {relevant_neurons_ids[-1]} (center CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}"
    #     )
    ax.set_title(f"{freq}\nR={vs:.3f}")
    ax.set_xlabel("Phase (radians)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return (vs, fig)

def calculate_vector_strength_from_result_polar(
        res,
        angle,
        side,
        pop,
        freq=None,  # if None: freq = res['basesound'].frequency
        cf_target=None,
        bandwidth=0,
        n_bins=7,
        display=False,
        color = None,
        figsize = [7,5]  # if True also return fig, show() in caller function
        ):
    
    # Get spikes and organize times per sender
    spikes = res["angle_to_rate"][angle][side][pop] 
    print(spikes)
    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        sender2times[sender].append(time)
    sender2times = {k: np.array(v) / 1000 for k, v in sender2times.items()}
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    
    # Determine the frequency to use
    if freq is None:
        if type(res['basesound']) in (Tone, ToneBurst):
            freq = res['basesound'].frequency
        else:
            print("Frequency needs to be specified for non-Tone sounds")
    else:
        freq = freq * Hz
    
    # Determine the closest characteristic frequency (CF) neuron
    if cf_target is None:    
        cf_neuron, center_neuron_for_freq = take_closest(cf, freq)
    else:
        cf_neuron, center_neuron_for_freq = take_closest(cf, cf_target * Hz)
    
    # Map between old and new neuron IDs
    old2newid = {oldid: i for i, oldid in enumerate(spikes["global_ids"])}
    new2oldid = {v: k for k, v in old2newid.items()}
    
    # Choose relevant neurons based on the center neuron and bandwidth
    relevant_neurons = range_around_center(
        center_neuron_for_freq, radius=bandwidth, max_val=num_neurons - 1
    )
    relevant_neurons_ids = [new2oldid[i] for i in relevant_neurons]
    
    # Concatenate the spike times from the relevant neurons
    spike_times_list = [sender2times[i] for i in relevant_neurons_ids]
    spike_times_array = np.concatenate(spike_times_list)  # Flatten into a single array
    
    # Compute phases and vector strength
    phases = get_spike_phases(spike_times=spike_times_array, frequency=freq / Hz)
    vs = calculate_vector_strength(spike_times=spike_times_array, frequency=freq / Hz)
    
    if not display:
        return (vs, None)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'
        else: color = 'k'

    
    # Plot phases in polar coordinates
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create a polar subplot
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=figsize)
    hist1, _ = np.histogram(phases, bins=bins, density = True)
    ax.bar(bin_centers, hist1, width=2 * np.pi / n_bins, alpha=0.7, color = color)
    
    if bandwidth == 0:
        ax.set_title(f"Neuron {relevant_neurons_ids[0]} (CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}")
    else:
        ax.set_title(f"Neurons {relevant_neurons_ids[0]} : {relevant_neurons_ids[-1]} (center CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}")
    
    # Remove all but the last yticks
    ax.set_yticks([])  # Keep only the last tick
    #ax.yaxis.set_tick_params(labelsize=10)  # Adjust size if needed

    plt.show()
    return

def draw_spikes_pop(
    res,
    angle,
    side,
    pop,
    y_ax = 'ids',
    f_ticks = [100,1000,10000],
    title=None,
    xlim=None,
    color = None,
    figsize = (7,4)
):
    spikes = res["angle_to_rate"][angle][side][pop]  
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    fig, ax = plt.subplots(1, figsize = figsize)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

    if xlim == None: 
        xlim_array = [0,duration]
    else: xlim_array = xlim

    if y_ax == 'ids':
        y_values = spikes['senders']
        ylabel = "id_senders"
    elif y_ax == 'cf':
        y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
        ylabel = "Characteristic Frequency [Hz]"
    elif y_ax == 'global_ids':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = "Global Neuron IDs"
    elif y_ax == 'cf_custom':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = "Characteristic Frequency"
        label_indexes = np.zeros_like(f_ticks)
        for i, f in enumerate(f_ticks):
            l, label_indexes[i] = take_closest(cf, f*Hz)
        ax.set_yticks(label_indexes)
        ax.set_yticklabels(f_ticks*Hz) 
    else:
        raise ValueError("Invalid value for 'ax'. Choose 'ids', 'cf', or 'global_ids'.")

    ax.plot(spikes['times'], y_values, '.', color = color, markersize=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time [ms]")
    ax.set_xlim(xlim_array)
    plt.show()
    return

def draw_rate_vs_angle_pop(
    data,
    title,
    pop = 'LSO',
    rate=True,
    mode = 'default',
    hist_logscale=True,
    ylim=None,
    show_hist=True,
    figsize = [7,4] 
):
    angle_to_rate = data["angle_to_rate"]
    duration = (
        data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms
    )
    print(f"simulation time={duration}")

    angles = list(angle_to_rate.keys())
    sides = ["L", "R"]

    tot_spikes = {
        side: [
            len(angle_to_rate[angle][side][pop]["times"] / duration)
            for angle in angles
        ]
        for side in sides
    }
    active_neuron_rate = {
        side: [
            avg_fire_rate_actv_neurons(angle_to_rate[angle][side][pop])
            * (1 * b2.second)
            / duration
            for angle in angles
        ]
        for side in sides
    }
    distr = {
        side: [
            firing_neurons_distribution(angle_to_rate[angle][side][pop])
            for angle in angles
        ]
        for side in sides
    }

    fig, ax = plt.subplots(figsize=figsize)
# Plot based on the mode:
    if mode == 'default':
        plotted_rate = active_neuron_rate if rate else tot_spikes
        ax.plot(angles, plotted_rate["L"], 'o-', color='m')
        ax.plot(angles, plotted_rate["R"], 'o-', color='g')
        ylabel_text = "Avg Firing Rate [Hz]" if rate else "Pop Firing Rate [Hz]"
        ax.set_ylabel(ylabel_text)
    elif mode == 'diff':
        # Normalize the two metrics for each side by their respective maxima.
        norm_active_L = np.array(active_neuron_rate["L"]) / np.max(active_neuron_rate["L"])
        norm_tot_L    = np.array(tot_spikes["L"]) / np.max(tot_spikes["L"])
        norm_active_R = np.array(active_neuron_rate["R"]) / np.max(active_neuron_rate["R"])
        norm_tot_R    = np.array(tot_spikes["R"]) / np.max(tot_spikes["R"])

        ax.plot(angles, norm_active_L, 'o-', color='m', label='avg_rate')
        ax.plot(angles, norm_tot_L, 'o-', color='darkmagenta', label='pop_rate')
        ax.plot(angles, norm_active_R, 'o-', color='g', label='avg_rate')
        ax.plot(angles, norm_tot_R, 'o-', color='darkgreen', label='pop_rate')
        ax.legend()
    else:
        raise ValueError("Unknown mode. Use 'default' or 'diff'.")
    
    ax.set_ylim(ylim)

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{j}°" for j in angle_to_rate.keys()])
    ax.set_title(f"{title} \n {pop}")

    if show_hist:
        v = ax.twinx()
        v.grid(visible=False)  # or use linestyle='--'

        senders_renamed = {
            side: [
                shift_senders(angle_to_rate[angle][side][pop], hist_logscale)
                for angle in angles
            ]
            for side in sides
        }
        max_spikes_single_neuron = max(flatten(distr.values()))
        draw_hist(
            v,
            senders_renamed,
            angles,
            num_neurons=len(angle_to_rate[0]["L"][pop]["global_ids"]),
            max_spikes_single_neuron=max_spikes_single_neuron,
            logscale=hist_logscale,
        )    

    plt.tight_layout()
    plt.show()
    return

def draw_rate_vs_angle(
    data,
    title,
    rate=True,
    mode = 'default',
    hist_logscale=True,
    show_pops=["LSO"],
    ylim=None,
    show_hist=True,
):
    angle_to_rate = data["angle_to_rate"]
    duration = (
        data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms
    )
    print(f"simulation time={duration}")

    angles = list(angle_to_rate.keys())
    sides = ["L", "R"]

    with plt.ioff():
        fig, ax = plt.subplots(math.ceil(len(show_pops)/2),2, figsize=(20, 2*len(show_pops)))
    ax = list(flatten([ax]))

    for i, pop in enumerate(show_pops):
        num_active = {
            side: [len(set(angle_to_rate[angle][side][pop])) for angle in angles]
            for side in sides
        }
        tot_spikes = {
            side: [
                len(angle_to_rate[angle][side][pop]["times"] / duration)
                for angle in angles
            ]
            for side in sides
        }
        active_neuron_rate = {
            side: [
                avg_fire_rate_actv_neurons(angle_to_rate[angle][side][pop])
                * (1 * b2.second)
                / duration
                for angle in angles
            ]
            for side in sides
        }
        distr = {
            side: [
                firing_neurons_distribution(angle_to_rate[angle][side][pop])
                for angle in angles
            ]
            for side in sides
        }
        if mode == 'default':
            plotted_rate = active_neuron_rate if rate else tot_spikes
            ax[i].plot(angles, plotted_rate["L"], 'o-', color='m')
            ax[i].plot(angles, plotted_rate["R"], 'o-', color='g')
            ylabel_text = "Avg Firing Rate [Hz]" if rate else "Pop Firing Rate [Hz]"
            ax[i].set_ylabel(ylabel_text)
        elif mode == 'diff':
        # Normalize the two metrics for each side by their respective maxima.
            norm_active_L = np.array(active_neuron_rate["L"]) / np.max(active_neuron_rate["L"])
            norm_tot_L    = np.array(tot_spikes["L"]) / np.max(tot_spikes["L"])
            norm_active_R = np.array(active_neuron_rate["R"]) / np.max(active_neuron_rate["R"])
            norm_tot_R    = np.array(tot_spikes["R"]) / np.max(tot_spikes["R"])
            
            ax.plot(angles, norm_active_L, 'o-', color='m', label='avg_rate')
            ax.plot(angles, norm_tot_L, 'o-', color='darkmagenta', label='pop_rate')
            ax.plot(angles, norm_active_R, 'o-', color='g', label='avg_rate')
            ax.plot(angles, norm_tot_R, 'o-', color='darkgreen', label='pop_rate')
            ax.legend()
        else:
            raise ValueError("Unknown mode. Use 'default' or 'diff'.")

        ax[i].plot(angles, plotted_rate["L"], 'o-', color = 'm')
        ax[i].set_title(pop)
        ax[i].plot(angles, plotted_rate["R"], 'o-', color = 'g')
        ax[i].set_ylabel("Avg Fring Rate [Hz]" if rate else "Population Firing Rate")
        ax[i].set_ylim(ylim)
        ax[i].set_xticks(angles)
        ax[i].set_xticklabels([f"{j}°" for j in angle_to_rate.keys()])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

        if show_hist:
            v = ax[i].twinx()
            v.grid(visible=False)  # or use linestyle='--'

            senders_renamed = {
                side: [
                    shift_senders(angle_to_rate[angle][side][pop], hist_logscale)
                    for angle in angles
                ]
                for side in sides
            }
            max_spikes_single_neuron = max(flatten(distr.values()))
            draw_hist(
                v,
                senders_renamed,
                angles,
                num_neurons=len(angle_to_rate[0]["L"][pop]["global_ids"]),
                max_spikes_single_neuron=max_spikes_single_neuron,
                logscale=hist_logscale,
            )    

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return


