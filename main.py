import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stock_data import download_stock_data
from predictor import (
    forecast_prices,
    compare_models,
    save_forecast
)


COMPANIES = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA"
}


class StockPredictorApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Stock Predictor")
        self.root.geometry("1100x780")
        self.root.minsize(900, 650)

        self.data = None
        self.loaded_ticker = None
        self.last_forecast = None

        title = ttk.Label(
            root,
            text="AI Stock Predictor",
            font=("Arial", 23, "bold")
        )
        title.pack(pady=15)

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill="x")

        ttk.Label(
            controls,
            text="Company:"
        ).grid(row=0, column=0, padx=5, pady=5)

        self.company_var = tk.StringVar(
            value="Apple (AAPL)"
        )

        ttk.Combobox(
            controls,
            textvariable=self.company_var,
            values=list(COMPANIES.keys()),
            state="readonly",
            width=20
        ).grid(row=0, column=1, padx=5)

        ttk.Label(
            controls,
            text="Forecast days:"
        ).grid(row=0, column=2, padx=5)

        self.horizon_var = tk.StringVar(value="5")

        ttk.Combobox(
            controls,
            textvariable=self.horizon_var,
            values=["5", "10", "30"],
            state="readonly",
            width=10
        ).grid(row=0, column=3, padx=5)


        ttk.Label(
            controls,
            text="AI Model:"
        ).grid(row=0, column=4, padx=5)

        self.model_var = tk.StringVar(
            value="Chronos-T5 Small"
        )

        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=[
                "Chronos-T5 Small",
                "Chronos-Bolt Small"
            ],
            state="readonly",
            width=22
        ).grid(row=0, column=5, padx=5)

        ttk.Button(
            controls,
            text="Load stock data",
            command=self.load_data
        ).grid(row=1, column=0, padx=5, pady=10)

        ttk.Button(
            controls,
            text="Forecast",
            command=self.run_forecast
        ).grid(row=1, column=1, padx=5)

        ttk.Button(
            controls,
            text="Compare models",
            command=self.run_comparison
        ).grid(row=1, column=2, padx=5)

        ttk.Button(
            controls,
            text="Compare all companies",
            command=self.run_all_comparison
        ).grid(row=1, column=3, padx=5)

        self.status = tk.StringVar(
            value="Load historical stock data to begin."
        )

        ttk.Label(
            root,
            textvariable=self.status,
            foreground="blue",
            wraplength=1000
        ).pack(pady=5)


        self.result = tk.Text(
            root,
            height=12,
            wrap="word",
            font=("Consolas", 10)
        )

        self.result.pack(
            fill="x",
            padx=12,
            pady=5
        )

        # Chart area

        self.plot_frame = ttk.Frame(root)

        self.plot_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        ttk.Label(
            root,
            text="Forecasts are experimental and are not investment advice.",
            foreground="red"
        ).pack(pady=8)

    def selected_ticker(self):

        return COMPANIES[self.company_var.get()]


    def load_data(self):

        ticker = self.selected_ticker()

        self.status.set("Downloading historical data...")
        self.root.update_idletasks()

        try:

            self.data = download_stock_data(ticker)

            self.loaded_ticker = ticker
            self.last_forecast = None

            self.result.delete("1.0", tk.END)

            self.result.insert(
                tk.END,
                f"Company: {ticker}\n"
            )

            self.result.insert(
                tk.END,
                f"Records: {len(self.data)}\n"
            )

            self.result.insert(
                tk.END,
                f"First date: {self.data.index.min().date()}\n"
            )

            self.result.insert(
                tk.END,
                f"Last date: {self.data.index.max().date()}\n"
            )

            self.result.insert(
                tk.END,
                f"Last closing price: ${self.data['Close'].iloc[-1]:.2f}\n"
            )

            self.draw_chart(
                self.data,
                None,
                "Historical stock price"
            )

            self.status.set(
                "Historical data successfully downloaded."
            )

        except Exception as error:

            messagebox.showerror(
                "Download error",
                str(error)
            )

            self.status.set("Data download failed.")

    # Forecast

    def run_forecast(self):

        ticker = self.selected_ticker()

        if self.data is None or self.loaded_ticker != ticker:

            messagebox.showwarning(
                "Warning",
                "Load data for the selected company first."
            )

            return

        horizon = int(self.horizon_var.get())
        model_name = self.model_var.get()

        self.status.set("AI model is generating a forecast...")
        self.root.update_idletasks()

        try:

            forecast = forecast_prices(
                self.data["Close"],
                horizon,
                model_name
            )

            self.last_forecast = forecast

            last_price = float(
                self.data["Close"].iloc[-1]
            )

            save_forecast(
                ticker,
                last_price,
                forecast,
                model_name
            )

            self.result.delete("1.0", tk.END)

            self.result.insert(
                tk.END,
                f"Company: {ticker}\n"
            )

            self.result.insert(
                tk.END,
                f"AI Model: {model_name}\n"
            )

            self.result.insert(
                tk.END,
                f"Last actual price: ${last_price:.2f}\n\n"
            )

            self.result.insert(
                tk.END,
                "Date          Forecast       Change       Change %\n"
            )

            for date, price in forecast.items():

                change = price - last_price

                percent = (
                    change / last_price
                ) * 100

                self.result.insert(
                    tk.END,
                    f"{date.date()}   "
                    f"${price:.2f}   "
                    f"{change:+.2f}   "
                    f"{percent:+.2f}%\n"
                )

            self.draw_chart(
                self.data,
                forecast,
                f"{ticker}: Historical data and forecast"
            )

            self.status.set(
                "Forecast completed and saved to CSV."
            )

        except Exception as error:

            messagebox.showerror(
                "Forecast error",
                str(error)
            )

            self.status.set("Forecast failed.")

    # Compare two models for selected company

    def run_comparison(self):

        ticker = self.selected_ticker()

        if self.data is None or self.loaded_ticker != ticker:

            messagebox.showwarning(
                "Warning",
                "Load data for the selected company first."
            )

            return

        self.status.set(
            "Testing two AI models..."
        )

        self.root.update_idletasks()

        try:

            results = compare_models(
                self.data["Close"],
                ticker
            )

            self.result.delete("1.0", tk.END)

            self.result.insert(
                tk.END,
                "MODEL COMPARISON\n\n"
            )

            self.result.insert(
                tk.END,
                results.to_string(index=False)
            )

            self.result.insert(
                tk.END,
                "\n\nResults saved to model_comparison.csv"
            )

            self.status.set(
                "Model comparison completed."
            )

        except Exception as error:

            messagebox.showerror(
                "Comparison error",
                str(error)
            )

            self.status.set("Comparison failed.")

    # Compare models for all three companies

    def run_all_comparison(self):

        self.status.set(
            "Comparing models for AAPL, MSFT and NVDA..."
        )

        self.root.update_idletasks()

        try:

            all_results = []

            for ticker in ["AAPL", "MSFT", "NVDA"]:

                company_data = download_stock_data(ticker)

                result = compare_models(
                    company_data["Close"],
                    ticker
                )

                all_results.append(result)

            combined = pd.concat(
                all_results,
                ignore_index=True
            )

            self.result.delete("1.0", tk.END)

            self.result.insert(
                tk.END,
                "COMPARISON OF ALL COMPANIES\n\n"
            )

            self.result.insert(
                tk.END,
                combined.to_string(index=False)
            )

            self.status.set(
                "All companies successfully compared."
            )

        except Exception as error:

            messagebox.showerror(
                "Error",
                str(error)
            )

            self.status.set("Comparison failed.")

    # Draw graph

    def draw_chart(
        self,
        history,
        forecast,
        chart_title
    ):

        for widget in self.plot_frame.winfo_children():

            widget.destroy()

        fig, ax = plt.subplots(
            figsize=(10, 4.5),
            dpi=95
        )

        ax.plot(
            history.index,
            history["Close"],
            label="Historical price",
            linewidth=1.5
        )

        if forecast is not None:

            ax.plot(
                forecast.index,
                forecast.values,
                marker="o",
                linestyle="--",
                label="AI forecast",
                linewidth=2
            )

            ax.axvline(
                history.index[-1],
                linestyle=":",
                label="Forecast boundary"
            )

        ax.set_title(chart_title)

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")

        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.autofmt_xdate()
        fig.tight_layout()

        os.makedirs("graphs", exist_ok=True)

        safe_name = (
            chart_title
            .replace(":", "")
            .replace(" ", "_")
            .replace("/", "_")
        )

        fig.savefig(
            os.path.join(
                "graphs",
                safe_name + ".png"
            ),
            dpi=150
        )

        canvas = FigureCanvasTkAgg(
            fig,
            master=self.plot_frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)


if __name__ == "__main__":

    root = tk.Tk()

    app = StockPredictorApp(root)

    root.mainloop()
