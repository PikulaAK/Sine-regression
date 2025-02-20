# filepath: /C:/Data/Projects/Avash/Python/_git/Sine-regression/sine_regression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import messagebox

def sinusoidal_model(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def read_excel_data(filepath):
    data = pd.read_excel(filepath)
    return data.iloc[:, 0], data.iloc[:, 1]  # Assuming first column is x and second is y

def perform_regression(x, y):
    initial_guess = [1, 1, 1, 1]  # Initial guess for parameters a, b, c, d
    params, _ = curve_fit(sinusoidal_model, x, y, p0=initial_guess)
    return params

def plot_results(x, y, params):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, label='Data Points', color='red')
    plt.plot(x, sinusoidal_model(x, *params), label='Fitted Curve', color='blue')
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

