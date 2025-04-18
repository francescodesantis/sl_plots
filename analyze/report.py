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
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight']= 'bold'
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.spines.top']= False
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['xtick.labelsize'] = 12   # Size of x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12   # Size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 14   # Size of the legend text
# Make axis labels bold
plt.rcParams['axes.labelweight'] = 'bold'  # Makes x and y axis labels bold

def create_xax_time_sound(res):
    x_times = np.linspace(0, res['simulation_time'], int((res['basesound'].sound.samplerate / b2.kHz)*res['simulation_time']))
    return x_times

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
        display=False, # if True also return fig, show() in caller function
        x_ax = "phase",  # can be "phase" or "time"
        ylim = None,    # Added ylim parameter
        center_at_peak = False  # Center histogram so that the peak bin is at zero
        ):
    
    spikes = res["angle_to_rate"][angle][side][pop]
    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        if time <= 1000:
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
        _, center_neuron_for_freq = take_closest(cf, freq)
    else:
        _, center_neuron_for_freq = take_closest(cf, cf_target *Hz)

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
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get total number of spikes for percentage calculation
    total_spikes = len(spike_times_array)
    
    if x_ax == "phase":
        # Initial binning to find the peak
        orig_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
        hist_values, _ = np.histogram(phases, bins=orig_bins)
        peak_bin_idx = np.argmax(hist_values)
        
        if center_at_peak:
            # Calculate the center of the peak bin
            bin_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center = bin_centers[peak_bin_idx]
            
            # Create a shift that will center the peak at 0
            shift = peak_center - np.pi  # Shift to make peak at π, then will offset by π
            
            # Create bins centered around the peak
            bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            shifted_phases = np.mod(phases - shift, 2 * np.pi) - np.pi
            
            # Plot the shifted histogram as percentages
            hist1, _ = np.histogram(shifted_phases, bins=bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, hist_percent, width=2 * np.pi / n_bins, alpha=0.7, color=color)
            
            # Set x-ticks in terms of π
            pi_ticks = np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
            pi_labels = ['-0.5', '', '0', '', '0.5']
        else:
            # Original histogram from 0 to 2π
            bins = np.linspace(0, 2 * np.pi, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            hist1, _ = np.histogram(phases, bins=bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            ax.bar(bin_centers, hist_percent, width=2 * np.pi / n_bins, alpha=0.7, color=color)
            
            # Set x-ticks in terms of π
            pi_ticks = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
            pi_labels = ['0', '', '0.5', '', '1']
        
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        ax.set_xlabel("Phase (cycles)")
        
    elif x_ax == "time":
        # Convert phases to time in milliseconds
        period_ms = 1000 / (freq / Hz)  # Period in milliseconds
        time_values = (phases / (2 * np.pi)) * period_ms
        
        # Initial binning to find the peak
        orig_time_bins = np.linspace(0, period_ms, n_bins + 1)
        hist_values, _ = np.histogram(time_values, bins=orig_time_bins)
        peak_bin_idx = np.argmax(hist_values)
        
        if center_at_peak:
            # Calculate the center of the peak bin
            bin_centers = (orig_time_bins[:-1] + orig_time_bins[1:]) / 2
            peak_center = bin_centers[peak_bin_idx]
            
            # Create a shift that will center the peak at 0
            shift = peak_center - period_ms/2  # Shift to make peak at period/2, then will offset
            
            # Create bins centered around the peak
            time_bins = np.linspace(-period_ms/2, period_ms/2, n_bins + 1)
            shifted_time_values = np.mod(time_values - shift, period_ms) - period_ms/2
            
            # Plot the shifted histogram as percentages
            hist1, _ = np.histogram(shifted_time_values, bins=time_bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            ax.bar(time_bin_centers, hist_percent, width=period_ms / n_bins, alpha=0.7, color=color)
        else:
            # Original time from 0 to period
            time_bins = np.linspace(0, period_ms, n_bins + 1)
            time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            
            hist1, _ = np.histogram(time_values, bins=time_bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            ax.bar(time_bin_centers, hist_percent, width=period_ms / n_bins, alpha=0.7, color=color)
        
        ax.set_xlabel("Time [ms]")
    
    # Set y-axis label to percentage
    ax.set_ylabel("Spikes/bin (% of total)")
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_title(f"R={vs:.3f}")
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
    y_ax = 'cf_custom',
    f_ticks = [100,1000,10000],
    title=None,
    plot_sound = False,
    xlim=None,
    ylim=None,
    color = None,
    figsize = (7,4)
):
    spikes = res["angle_to_rate"][angle][side][pop]  
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

    if xlim == None: 
        xlim_array = [0,duration]
    else: xlim_array = xlim

    if y_ax == 'global_ids':
        y_values = spikes['senders']
        ylabel = f"{pop} global IDS"
    elif y_ax == 'cf':
        y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
        ylabel = f"{pop} CF [Hz]"
    elif y_ax == 'ids':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = f"{pop} IDS"
    elif y_ax == 'cf_custom':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = f"{pop} CF [Hz]"
        label_indexes = np.zeros_like(f_ticks)
        for i, f in enumerate(f_ticks):
            _, label_indexes[i] = take_closest(cf, f*Hz)
    else:
        raise ValueError("Invalid value for 'ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")

    if(plot_sound):
        # Create figure with gridspec to control subplot sizes
        fig = plt.figure(figsize=(figsize[0], figsize[1]+1))
        # Create smaller subplot for sound (25% of height) and keep full width for spikes plot
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 7])
        
        # Sound plot - smaller and without axes/grid
        ax0 = fig.add_subplot(gs[0])
        x_sound = create_xax_time_sound(res)
        if len(x_sound) != len(res['basesound'].sound):
            x_sound = x_sound[0:len(res['basesound'].sound)]
        ax0.plot(x_sound, res['basesound'].sound, 'k', linewidth = '1')
        ax0.set_xlim(xlim_array)
        # Remove axes and grid
        ax0.axis('off')
        
        # Spikes plot - same size as when plot_sound is False
        ax1 = fig.add_subplot(gs[1])
        if ylim != None:
            if y_ax == 'cf':
                ax1.set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax1.set_ylim([y0,y1])
        else:
            ax1.set_yticks(label_indexes)
            ax1.set_yticklabels(f_ticks*Hz) 
        
        ax1.plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel("Time [ms]")
        ax1.set_xlim(xlim_array)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if ylim != None:
            if y_ax == 'cf':
                ax.set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax.set_ylim([y0,y1])
        else:
            ax.set_yticks(label_indexes)
            ax.set_yticklabels(f_ticks*Hz) 
        
        ax.plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [ms]")
        ax.set_xlim(xlim_array)
    
    plt.show()
    return

def draw_spikes_pop_bothside(
    res,
    angle,
    pop,
    y_ax = 'cf_custom',
    f_ticks = [100,1000,10000],
    title=None,
    plot_sound = False,
    xlim=None,
    ylim=None,
    color = None,
    figsize = (7,8)
):
    fig, ax = plt.subplots(2, figsize = figsize, sharex = True)
    for i, side in enumerate(['L', 'R']):
        spikes = res["angle_to_rate"][angle][side][pop]  
        num_neurons = len(spikes["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)
        neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
        duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

        if xlim == None: 
            xlim_array = [0,duration]
        else: xlim_array = xlim
            
        if ylim is None:
            ylim = [CFMIN/Hz, CFMAX/Hz]

        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

        if y_ax == 'global_ids':
            y_values = spikes['senders']
            ylabel = f"{pop} global IDS"
        elif y_ax == 'cf':
            y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
            ylabel = f"{pop} CF [Hz]"
        elif y_ax == 'ids':
            y_values = spikes['senders'] - spikes['global_ids'][0]
            ylabel = f"{pop} IDS"
        elif y_ax == 'cf_custom':
            y_values = spikes['senders'] - spikes['global_ids'][0]
            ylabel = f"{pop} CF [Hz]"
            label_indexes = np.zeros_like(f_ticks)
            for i, f in enumerate(f_ticks):
                _, label_indexes[i] = take_closest(cf, f*Hz)
        else:
            raise ValueError("Invalid value for 'ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")
  
        if ylim != None:
            if y_ax == 'cf':
                ax[i].set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax[i].set_ylim([y0,y1])
        else:
            ax[i].set_yticks(label_indexes)
            ax[i].set_yticklabels(f_ticks*Hz) 
        
        ax[i].plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlim(xlim_array)
    ax[1].set_xlabel("Time [ms]")
    plt.show()
    return

def draw_psth_pop_bothside(
    res,
    angle,
    pop,
    title=None,
    xlim=None,
    ylim=None,
    bin_size = 1, #ms
    color = None,
    figsize = (7,4)
):
    fig, ax = plt.subplots(1, figsize = figsize)
    for color, side in zip(['m', 'g'],['L', 'R']):
        spikes = res["angle_to_rate"][angle][side][pop] 
        spike_times = spikes['times']
        spike_senders = spikes['senders']
        num_neurons = len(spikes["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)
        duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

        if xlim == None: 
            xlim_array = [0,duration]
        else: xlim_array = xlim
        
        if ylim is None:
            ylim = [CFMIN/Hz, CFMAX/Hz]

        time_mask = (spike_times >= xlim_array[0]) & (spike_times <= xlim_array[1])
        filtered_times = spike_times[time_mask]
        filtered_senders = spike_senders[time_mask]
        # Fix for the error - get indices properly
        _, ymin_idx = take_closest(cf, ylim[0]*Hz)
        _, ymax_idx = take_closest(cf, ylim[1]*Hz)
    
        # Convert indices to actual IDs
        ymin = spikes["global_ids"][0] + ymin_idx
        ymax = spikes["global_ids"][0] + ymax_idx
        cluster_mask = (filtered_senders >= ymin) & (filtered_senders <= ymax)
        cluster_times = filtered_times[cluster_mask]
        bins = np.arange(xlim_array[0], xlim_array[1] + bin_size, bin_size)
        ax.hist(cluster_times, bins=bins, alpha=0.7, color = color)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Spike Count')
        ax.grid(True, alpha=0.3)

    # You can add more customization to the axes if needed
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return

def draw_spikes_and_psth_bothside(
    res,
    angle,
    pop,
    y_ax='cf_custom',
    f_ticks=[100, 1000, 10000],
    title=None,
    plot_sound=False,
    xlim=None,
    ylim=None,
    bin_size=1,  # ms
    figsize=(7, 12)
):
    """
    Combined function that draws rasterplots for both sides (L and R), a PSTH histogram,
    and optionally plots the sound waveform at the top.
    
    Parameters:
    -----------
    res : dict
        Results dictionary containing spike data
    angle : float
        Angle value for which to plot data
    pop : str
        Population name (e.g., 'AN', 'IC')
    y_ax : str
        Type of y-axis to use ('cf_custom', 'ids', 'cf', or 'global_ids')
    f_ticks : list
        Frequency ticks for the y-axis (used with 'cf_custom')
    title : str, optional
        Title for the plot
    plot_sound : bool
        Whether to plot the sound waveform at the top
    xlim : tuple, optional
        Limits for the x-axis (min, max)
    ylim : tuple, optional
        Limits for the y-axis (min, max)
    bin_size : float
        Bin size for the histogram in ms
    figsize : tuple
        Figure size
    """
    # Determine number of subplots based on whether we're plotting sound
    n_plots = 4 if plot_sound else 3
    
    # Create figure with appropriate subplots
    fig, ax = plt.subplots(n_plots, figsize=figsize, sharex=True, 
                          gridspec_kw={'height_ratios': [0.5, 1, 1, 0.8] if plot_sound else [1, 1, 0.8]})
    
    # Set colors for left and right sides
    side_colors = {'L': 'm', 'R': 'g'}
    
    # Get duration for x-axis limits
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)
    
    if xlim is None:
        xlim = [0, duration]

    if ylim is None:
        ylim = [CFMIN/Hz, CFMAX/Hz]
    
    # Plot sound waveform if requested
    if plot_sound:
        sound_ax_index = 0
        raster_start_index = 1
        
        # Get sound data from res
        sound = res["basesound"].sound
        t = np.arange(len(sound)) / sound.samplerate * 1000  # Convert to ms
        
        # Plot the sound waveform
        ax[sound_ax_index].plot(t, sound, 'k', lw=0.5)
        ax[sound_ax_index].set_ylabel('Amplitude')
        ax[sound_ax_index].set_xlim(xlim)
        ax[sound_ax_index].set_title('Sound Waveform')
        ax[sound_ax_index].grid(True, alpha=0.3)
    else:
        raster_start_index = 0
    
    # Create the rasterplots
    for i, side in enumerate(['L', 'R']):
        plot_index = raster_start_index + i
        spikes = res["angle_to_rate"][angle][side][pop]
        num_neurons = len(spikes["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)
        neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
        
        
        if y_ax == 'cf':
            y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
            ylabel = f"{pop} CF [Hz]"
            ax[plot_index].set_ylim(ylim)
        else:
            _, y0 = take_closest(cf, ylim[0]*Hz)
            _, y1 = take_closest(cf, ylim[1]*Hz)
            ax[plot_index].set_ylim([y0, y1])
            if y_ax == 'global_ids':
                y_values = spikes['senders']
                ylabel = f"{pop} global IDs"
            if y_ax == 'ids':
                y_values = spikes['senders'] - spikes['global_ids'][0]
                ylabel = f"{pop} IDs"
            if y_ax == 'cf_custom':
                y_values = spikes['senders'] - spikes['global_ids'][0]
                ylabel = f"{pop} CF [Hz]"
                label_indexes = np.zeros_like(f_ticks)
                for j, f in enumerate(f_ticks):
                    _, label_indexes[j] = take_closest(cf, f*Hz)
                ax[plot_index].set_yticks(label_indexes)
                ax[plot_index].set_yticklabels(f_ticks)
            else:
                raise ValueError("Invalid value for 'y_ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")
        
        ax[plot_index].plot(spikes['times'], y_values, '.', color=side_colors[side], markersize=1)
        ax[plot_index].set_ylabel(ylabel)
        ax[plot_index].set_xlim(xlim)
        
        # Add a side label to indicate L or R
        ax[plot_index].text(0.01, 0.95, f"{side} side", transform=ax[plot_index].transAxes, 
                          fontsize=14, fontweight='bold', color=side_colors[side])
    
    # Create the PSTH histogram as the last subplot
    psth_index = n_plots - 1
    for side in ['L', 'R']:
        color = side_colors[side]
        spikes = res["angle_to_rate"][angle][side][pop]
        spike_times = spikes['times']
        spike_senders = spikes['senders']
        num_neurons = len(spikes["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)
        
        time_mask = (spike_times >= xlim[0]) & (spike_times <= xlim[1])
        filtered_times = spike_times[time_mask]
        filtered_senders = spike_senders[time_mask]
        
        # Get indices for the specified frequency range
        _, ymin_idx = take_closest(cf, ylim[0]*Hz)
        _, ymax_idx = take_closest(cf, ylim[1]*Hz)
        
        # Convert indices to actual IDs
        ymin = spikes["global_ids"][0] + ymin_idx
        ymax = spikes["global_ids"][0] + ymax_idx
        
        # Filter spikes within the specified range
        cluster_mask = (filtered_senders >= ymin) & (filtered_senders <= ymax)
        cluster_times = filtered_times[cluster_mask]
        
        # Create histogram
        bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
        ax[psth_index].hist(cluster_times, bins=bins, alpha=0.5, color=color, label=f"{side} side")
    
    # Customize PSTH subplot
    ax[psth_index].set_xlabel('Time [ms]')
    ax[psth_index].set_ylabel('Spike Count')
    ax[psth_index].legend()
    ax[psth_index].grid(True, alpha=0.3)
    ax[psth_index].spines['top'].set_visible(False)
    ax[psth_index].spines['right'].set_visible(False)
    
    # Set the main title if provided
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.92)  # Make room for the title
    
    plt.show()
    return fig, ax

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
        ax[i].set_xlabel("Azimuth Angle")
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

    plt.tight_layout()
    plt.show()
    return

def draw_psth_pop(
    res,
    angle,
    side,
    pop,
    title=None,
    xlim=None,
    ylim=None,
    bin_size = 1, #ms
    color = None,
    figsize = (7,4)
):
    spikes = res["angle_to_rate"][angle][side][pop]  
    spike_times = spikes['times']
    spike_senders = spikes['senders']
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

    if xlim == None: 
        xlim_array = [0,duration]
    else: xlim_array = xlim

    time_mask = (spike_times >= xlim[0]) & (spike_times <= xlim[1])
    filtered_times = spike_times[time_mask]
    filtered_senders = spike_senders[time_mask]
    # Fix for the error - get indices properly
    fmin, ymin_idx = take_closest(cf, ylim[0]*Hz)
    fmax, ymax_idx = take_closest(cf, ylim[1]*Hz)
    
    # Convert indices to actual IDs
    ymin = spikes["global_ids"][0] + ymin_idx
    ymax = spikes["global_ids"][0] + ymax_idx
    cluster_mask = (filtered_senders >= ymin) & (filtered_senders <= ymax)
    cluster_times = filtered_times[cluster_mask]
    bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
    fig, ax = plt.subplots(1, figsize = figsize)
    ax.hist(cluster_times, bins=bins, alpha=0.7, color = color)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spike Count')
    ax.grid(True, alpha=0.3)

    # You can add more customization to the axes if needed
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return

def draw_rate_vs_angle_pop(
    data,
    title = None,
    pop = 'LSO',
    rate=True,
    mode = 'default',
    hist_logscale=True,
    cf_interval=None,
    show_hist=True,
    figsize = [7,4] 
):
    angle_to_rate = data["angle_to_rate"]
    duration = (data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms)
    angles = list(angle_to_rate.keys())
    sides = ["L", "R"]
    num_neurons = len(angle_to_rate[0]['L'][pop]["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)

    if cf_interval == None:
        tot_spikes = {
            side: [
                len(angle_to_rate[angle][side][pop]["times"]) / duration
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
    else:
        tot_spikes = {}
        active_neuron_rate = {}
        cluster_numerosity = {}
        for side in sides:
            tot_spikes[side] = []
            active_neuron_rate[side] = []
            for angle in angles:
                # Find indices in CF array corresponding to interval bounds
                _, ymin_idx = take_closest(cf, cf_interval[0]*Hz)
                _, ymax_idx = take_closest(cf, cf_interval[1]*Hz)

                # Calculate actual neuron IDs from global_ids and indices
                base_id = angle_to_rate[angle][side][pop]["global_ids"][0]
                ymin = base_id + ymin_idx
                ymax = base_id + ymax_idx

                # Filter spikes within the specified range
                cluster_mask = (angle_to_rate[angle][side][pop]['senders'] >= ymin) & (angle_to_rate[angle][side][pop]['senders'] <= ymax)
                cluster_times = angle_to_rate[angle][side][pop]['times'][cluster_mask]

                # Calculate rate for this angle and side
                tot_spikes[side].append(len(cluster_times) / duration)
                active_neuron_rate[side].append(len(cluster_times)/((ymax - ymin)*duration))

                # NEW: Compute cluster numerosity (unique senders)
                unique_senders = np.unique(angle_to_rate[angle][side][pop]['senders'][cluster_mask])
            cluster_numerosity[side] = len(unique_senders)
            print(f"side {side}, considered {cluster_numerosity[side]} cells of {num_neurons} total")

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

    ax.set_xticks(angles)
    ax.set_xlabel("Azimuth Angle")
    ax.set_xticklabels([f"{j}°" for j in angle_to_rate.keys()])

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

