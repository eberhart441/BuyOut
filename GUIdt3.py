import customtkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd
import warnings

import multiprocessing as mp
import threading
import asyncio
import queue
import sys

from AutoTrade import ALGOat3
from Algo import ALGOqueryProcesor

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class stdoutRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass


class App(tk.CTk):
    def __init__(self):
        super().__init__()

        def updateApp():
            try:
                self.liveAlgoLogUpdates.delete("0.0", tk.END)
                self.liveAlgoLogUpdates.insert(tk.END, open("algoLog.log").read())
                self.liveAlgoLogUpdates.see(tk.END)

                while True:  # Keep checking progression in the queue
                    progress = self.progressQueue.get_nowait()
                    if progress:
                        # Update the progress bar with the current value of self.progressValue
                        self.progressBar.set(progress)
            except queue.Empty:
                pass
            finally:
                self.after(100, updateApp)

        # Declare int var to use for graph history radiobutton
        self.AQbuttonVar = tk.IntVar()

        self.ALGOqueryQueue = pd.DataFrame(
            columns=[
                "ticker",
                "duration",
                "dataSize",
                "resultSize",
            ]
        )
        self.ALGOqueryHistory = pd.DataFrame(
            columns=[
                "ticker",
                "duration",
                "dataSize",
                "resultSize",
            ]
        )

        self.ALGOqueryResultHistory = []
        self.ALGOqueryDataHistory = []
        self.ALGOqueryButtonHistory = []

        self.ALGOqueryID = 0

        # Declare progressBar and set it to 0
        self.progressBar = tk.CTkProgressBar(self, height=10)
        self.progressBar.grid(row=3, columnspan=3, sticky="ew")
        self.progressBar.set(0)

        # Shared queue of progress updates
        manager = mp.Manager()
        self.progressQueue = manager.Queue()

        # Initilize AQP
        AQthread = threading.Thread(target=lambda: ALGOqueryProcesor.AQP(self))
        AQthread.start()

        self.configureWindow()
        self.declareGraphHistoryFrame()
        self.declareGraph()
        self.declareTabView()

        # Run function to check for progress updates inside other thread
        updateProgressBarThread = threading.Thread(target=updateApp)
        updateProgressBarThread.start()

    def configureWindow(self):
        # Set title and resizing properties
        self.title("BuyIn")
        self.resizable(True, True)

        # Set weight for expandable columns and rows
        self.grid_columnconfigure(1, weight=1)  # Graph column to expand horizontally
        self.grid_rowconfigure(0, weight=0)  # Graph row to expand vertically
        self.grid_rowconfigure(
            1, weight=1
        )  # The row for self.liveAlgoUpdates to expand vertically

        # Default apperence properties for app
        tk.set_appearance_mode("dark")
        tk.set_default_color_theme("blue")

    def declareGraphHistoryFrame(self):
        # Declare graph history frame
        self.graphHistoryFrame = tk.CTkScrollableFrame(
            self, label_text="BuyIn", width=225
        )
        self.graphHistoryFrame.grid(rowspan=2, row=0, column=0, sticky="nsew")
        self.graphHistoryFrame.columnconfigure(0, weight=1)

    def declareGraph(self):
        # Configure default graph settings
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self)

        # Set the color of figure and axes backgrounds to match the GUI background
        self.plot.set_facecolor("#1d1e1e")

        # Shrink graph border
        self.plot.axes.get_xaxis().set_visible(False)
        self.plot.axes.get_yaxis().set_visible(False)
        self.fig.tight_layout(pad=0)

        self.plot.tick_params(labelcolor="white")
        self.plot.plot(color="#1f6aa5")

        # Make sure the graph expands in all directions
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2, sticky="nsew")

    def declareTabView(self):
        def declareStockPredictTab(self):
            def updateResultSize(resultSize):
                if self.ALGOqueryHistory.shape[0] != 0:
                    # Determine which radio button is currently clicked
                    index = self.AQbuttonVar.get()

                    averageResult = self.ALGOqueryResultHistory[index]
                    tickerData = self.ALGOqueryDataHistory[index]

                    averageResult = averageResult.iloc[:, : int(resultSize)].mean(
                        axis=1
                    )
                    averageResult = averageResult.mul(
                        tickerData.iloc[-1] / averageResult[tickerData.shape[0]]
                    )

                    # Graph data
                    self.plot.cla()  # Clears data

                    self.plot.plot(averageResult, color="white")

                    self.plot.plot(tickerData, color="#1f6aa5", linewidth=3)
                    self.canvas.draw()

            # Declare 'Stock Predict' objects
            self.spTickerEntryBox = tk.CTkEntry(
                self.tabview.tab("Stock Predict"), placeholder_text="Ticker"
            )
            self.spTickerEntryBox.grid(row=0, column=0, padx=10, pady=(10, 0))
            self.spDurationBox = tk.CTkOptionMenu(
                self.tabview.tab("Stock Predict"),
                values=["12h", "3d", "5d"],
            )
            self.spDurationBox.grid(row=1, column=0, padx=10, pady=(10, 0))
            self.spDurationBox.set("5d")
            self.spButton = tk.CTkButton(
                self.tabview.tab("Stock Predict"),
                text="Find Matches",
                command=lambda: self.ALGOquery(
                    self.spTickerEntryBox.get(),
                    self.spDurationBox.get(),
                    self.spDataSizeSlider.get(),
                    self.spResultSizeSlider.get(),
                ),
            )
            self.spButton.grid(row=2, column=0, padx=10, pady=(10, 0))

            self.spDataSizeLabel = tk.CTkLabel(
                self.tabview.tab("Stock Predict"), text="Database Size - Result Size"
            )
            self.spDataSizeLabel.grid(row=0, column=1, padx=0, pady=(10, 0))

            self.spDataSizeSlider = tk.CTkSlider(
                self.tabview.tab("Stock Predict"), from_=100, to=10000
            )
            self.spDataSizeSlider.grid(row=1, column=1, padx=10, pady=(10, 0))
            self.spResultSizeSlider = tk.CTkSlider(
                self.tabview.tab("Stock Predict"),
                from_=1,
                to=100,
                command=updateResultSize,
            )
            self.spResultSizeSlider.grid(row=2, column=1, padx=10, pady=(10, 0))
            self.spResultSizeSlider.configure(number_of_steps=10)

        def declareAutotradingTab(self):
            def callAt():
                sys.stdout = stdoutRedirector(self.atLiveAlgorithmUpdates)
                ALGOat3.AutoTrade.repeat = True

                def runATthread():
                    AutoTrade = ALGOat3.AutoTrade(self)
                    AutoTrade.ALGOat

                # Start the thread and target the runATthread function.
                ALGOatThread = threading.Thread(target=runATthread)
                ALGOatThread.start()
                print("AT COMMAND SENT")

            def closeAt():
                print("Stopping Auto Trade...")
                ALGOat3.AutoTrade.repeat = False

            # Declare 'Stock Predict' objects
            self.atPathBox = tk.CTkEntry(
                self.tabview.tab("Auto Trade"), placeholder_text="Ticker"
            )
            self.atPathBox.grid(row=0, column=0, padx=10, pady=(10, 0))
            self.atPathBox.insert(
                tk.END,
                "C:/Users/Simon E/Documents/BuyIn/Resources/atProcesses.csv",
            )

            self.atStopButton = tk.CTkButton(
                self.tabview.tab("Auto Trade"),
                text="STOP",
                fg_color="#3B3B3B",
                command=closeAt,
            )
            self.atStopButton.grid(row=1, column=0, padx=10, pady=(10, 0))
            self.atStartButton = tk.CTkButton(
                self.tabview.tab("Auto Trade"),
                text="START",
                command=callAt,
            )
            self.atStartButton.grid(row=2, column=0, padx=10, pady=(10, 0))

            self.atLiveAlgorithmUpdates = tk.CTkTextbox(
                self.tabview.tab("Auto Trade"), width=200, height=100
            )
            self.atLiveAlgorithmUpdates.grid(
                row=0,
                rowspan=3,
                column=1,
                pady=(10, 0),
                sticky="ns",
            )

        # Declare tabview
        self.tabview = tk.CTkTabview(self, width=400, height=175)
        self.tabview.grid(row=0, column=2, padx=20, pady=(0, 20), sticky="nsew")

        # Add tabs to tabview
        self.tabview.add("Stock Predict")
        self.tabview.add("Auto Trade")

        declareStockPredictTab(self)
        declareAutotradingTab(self)

        self.liveAlgoLogUpdates = tk.CTkTextbox(self)
        self.liveAlgoLogUpdates.grid(
            row=1,
            column=2,
            padx=20,
            pady=(0, 20),
            sticky="nsew",
        )

    def ALGOquery(self, ticker, duration, dataSize, resultSize):
        ALGOquery = pd.DataFrame(
            {
                "ticker": [ticker],
                "duration": [duration],
                "dataSize": [dataSize],
                "resultSize": [resultSize],
                "ALGOqueryID": [self.ALGOqueryID],
            }
        )
        self.ALGOqueryID += 1

        self.ALGOqueryQueue = pd.concat([self.ALGOqueryQueue, ALGOquery])

    def loadALGOquery(self):
        # Determine which radio button is currently clicked
        ALGOquery = self.ALGOqueryHistory[
            self.ALGOqueryHistory.iloc[:, 4] == self.AQbuttonVar.get()
        ]

        # Configure sp widget values based on selected radiobutton
        self.spTickerEntryBox.delete(0, tk.END)
        self.spTickerEntryBox.insert(0, ALGOquery.iloc[0, 0])
        self.spDurationBox.set(ALGOquery.iloc[0, 1])
        self.spDataSizeSlider.set(int(ALGOquery.iloc[0, 2]))
        self.spResultSizeSlider.set(int(ALGOquery.iloc[0, 3]))

        # Retreive dand process data from list
        averageResult = self.ALGOqueryResultHistory[self.AQbuttonVar.get()]
        tickerData = self.ALGOqueryDataHistory[self.AQbuttonVar.get()]

        averageResult = averageResult.iloc[
            :, : int(self.spResultSizeSlider.get())
        ].mean(axis=1)
        averageResult = averageResult.mul(
            tickerData.iloc[-1] / averageResult[tickerData.shape[0]]
        )

        # Graph data
        self.plot.cla()  # Clears data

        self.plot.plot(averageResult, color="white")

        self.plot.plot(tickerData, color="#1f6aa5", linewidth=3)
        self.canvas.draw()


if __name__ == "__main__":
    # Initilize app
    app = App()
    app.mainloop()

# Crypto data downloaded 7/21/23
