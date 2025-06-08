import pandas as pd
import yfinance as yf
import customtkinter as tk
import queue
import threading
import time
import logging

from Algo import ALGOdt3

# Set the logging level for yfinance to WARNING or higher
yfinanceLogger = logging.getLogger("yfinance")
yfinanceLogger.setLevel(logging.WARNING)


class AQP:
    def __init__(self, main):
        self.main = main
        self.AQcheck()

    def AQcheck(self):
        while True:
            if self.main.ALGOqueryQueue.shape[0] != 0:
                # Evaluate query
                self.retreveQueryData(self.main.ALGOqueryQueue.iloc[0])
            time.sleep(0.1)

    def retreveQueryData(self, ALGOquery):
        ticker = ALGOquery["ticker"]
        duration = ALGOquery["duration"]
        dataSize = ALGOquery["dataSize"]
        ALGOqueryID = int(ALGOquery["ALGOqueryID"])

        successfulDownload = False
        attempts = 0
        maxAttempts = 10
        while not successfulDownload and attempts < maxAttempts:
            try:
                tickerData = yf.download(
                    ticker,
                    group_by="Ticker",
                    period=duration,
                    interval="5m",
                    prepost=False,
                    repair=True,
                ).reset_index()["Close"]
                if not tickerData.empty:
                    successfulDownload = True
            except Exception as e:
                print(f"ERROR: {e}")
                attempts += 1
        if not successfulDownload:
            print("FAILED TO DOWNLOAD DATA AFTER SEVERAL ATTEMPTS")

        # Multiply data to end with "1"
        multiplyFactor = 1 / tickerData.iloc[-1]
        tickerDataMultiplied = tickerData.mul(multiplyFactor)

        # Algo predicts 20% of downloaded data length into the future
        predictionLen = tickerDataMultiplied.shape[0] / 5

        # Configure radiobutton text
        text = ticker + " - " + duration + "\nIn Queue"
        self.AQradioButton = tk.CTkRadioButton(
            self.main.graphHistoryFrame,
            text=text,
            variable=self.main.AQbuttonVar,
            value=ALGOqueryID,
            command=self.main.loadALGOquery,
            state="disabled",
        )
        self.AQradioButton.grid(row=ALGOqueryID, column=0, padx=10, pady=(0, 10))

        self.queryALGO(
            ticker,
            duration,
            dataSize,
            tickerData,
            tickerDataMultiplied,
            predictionLen,
            ALGOquery,
        )

    def queryALGO(
        self,
        ticker,
        duration,
        dataSize,
        tickerData,
        tickerDataMultiplied,
        predictionLen,
        ALGOquery,
    ):
        def queryLocalMachine():
            # Call algo
            algoSettings = ALGOdt3.Algo(
                int(dataSize),
                predictionLen,
                tickerDataMultiplied,
                self.main.progressQueue,
            )

            # Return result
            workerResultQueue.put(algoSettings.startPool())

        # Dataframe to store results from algo
        averageResult = pd.DataFrame()

        # Queue for results from worker
        workerResultQueue = queue.Queue()

        # Start local algo instance and wait for completion
        localMachineThread = threading.Thread(target=queryLocalMachine)
        localMachineThread.start()
        localMachineThread.join()

        # Number of workers. average result should have 100 columns (100 results)
        for result in range(1):
            averageResult = pd.concat([averageResult, workerResultQueue.get()], axis=1)

        # Turn on radiobutton
        self.AQradioButton.configure(text=ticker + " - " + duration, state="normal")

        # Remove AQ from ALGOqueryQueue
        self.main.ALGOqueryQueue = self.main.ALGOqueryQueue[1:]

        # Add AQ to history
        self.main.ALGOqueryHistory = self.main.ALGOqueryHistory.append(
            ALGOquery, ignore_index=True
        )

        # Delete data after threshold to prevent unecissary space use
        self.main.ALGOqueryButtonHistory.append(self.AQradioButton)
        if len(self.main.ALGOqueryButtonHistory) > 50:
            self.main.ALGOqueryResultHistory[
                len(self.main.ALGOqueryResultHistory) - 50
            ] = 0
            self.main.ALGOqueryDataHistory[len(self.main.ALGOqueryDataHistory) - 50] = 0
            self.main.ALGOqueryButtonHistory.pop(0).destroy()

            self.main.ALGOqueryHistory = self.main.ALGOqueryHistory.drop(
                self.main.ALGOqueryHistory.index[0]
            )

        # Append AQ dataFrames for future use
        self.main.ALGOqueryResultHistory.append(averageResult)
        self.main.ALGOqueryDataHistory.append(tickerData)

        self.AQradioButton.invoke()
