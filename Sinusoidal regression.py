import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import messagebox

round_to = 0

def read_excel_data(filepath):
    data = pd.read_excel(filepath)
    return data.iloc[:, 0], data.iloc[:, 1]  # Assuming first column is x and second is y

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def perform_regression(x, y):
    result = fit_sin(x, y)
    print(f"Fitted parameters: Amplitude={result['amp']}, Angular freq.={result['omega']}, Phase={result['phase']}, Offset={result['offset']}")
    return result

def plot_results(x, y, result):
    global round_to
    est_amp = result['amp']
    est_freq = result['omega']
    est_phase = result['phase']
    est_mean = result['offset']
    fine_x = np.linspace(min(x), max(x), 1000)
    data_fit = result['fitfunc'](fine_x)
    graph_equation = f"{est_amp:.{round_to}f}sin({est_freq:.{round_to}f}x + {est_phase:.{round_to}f}) + {est_mean:.{round_to}f}"

    # For loop to iterate over x and y
    residuals = []
    for xi, yi in zip(x, y):
        print(f"x: {xi}, y: {yi}")
        residual = yi - result['fitfunc'](xi)
        residuals.append(residual)

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, label='Data Points', color='red')
    plt.plot(fine_x, data_fit, label='Fitted Curve', color='blue')
    plt.scatter(x, residuals, label='Residuals', color='green')
    plt.title('Sinusoidal Regression')
    plt.xlabel(graph_equation)
    print(graph_equation)
    plt.ylabel('')
    plt.legend()
    plt.grid()

    plt.show()

def main():
    global round_to
    file_path = input("Enter Excel file path: ")
    round_to = input("What decimal place would you like to round to? ")
    x, y = read_excel_data(filepath=file_path)
    result = perform_regression(x, y)
    plot_results(x, y, result)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    try:
        main()
    except Exception as e:
        messagebox.showerror("Error", str(e))