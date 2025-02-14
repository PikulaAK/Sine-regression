import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import tkinter as tk
from tkinter import messagebox

def sinusoidal_model(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def read_excel_data(filepath):
    data = pd.read_excel(filepath)
    return data.iloc[:, 0], data.iloc[:, 1]  # Assuming first column is x and second is y

def perform_regression(x, y):
    guess_mean = np.mean(y)
    guess_std = 3 * np.std(y) / (2**0.5)
    guess_phase = 0
    guess_freq = 1
    guess_amp = 1

    optimize_func = lambda params: params[0] * np.sin(params[1] * x + params[2]) + params[3] - y
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
    
    return est_amp, est_freq, est_phase, est_mean

def plot_results(x, y, params):
    est_amp, est_freq, est_phase, est_mean = params
    fine_x = np.linspace(min(x), max(x), 1000)
    data_fit = est_amp * np.sin(est_freq * fine_x + est_phase) + est_mean

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, label='Data Points', color='red')
    plt.plot(fine_x, data_fit, label='Fitted Curve', color='blue')
    plt.title('Sinusoidal Regression')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    file_path = input("Enter Excel file path: ")
    x, y = read_excel_data(filepath=file_path)
    params = perform_regression(x, y)
    plot_results(x, y, params)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    try:
        main()
    except Exception as e:
        messagebox.showerror("Error", str(e))