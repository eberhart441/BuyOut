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
                self.retrieveQueryData(self.main.ALGOqueryQueue.iloc[0])
            time.sleep(0.1)

    def retrieveQueryData(self, ALGOquery):
        ticker = ALGOquery["ticker"]
        duration = ALGOquery["duration"]
        dataSize = ALGOquery["dataSize"]
        ALGOqueryID = int(ALGOquery["ALGOqueryID"])

        # Enhanced data download with volume support
        successfulDownload = False
        attempts = 0
        maxAttempts = 10
        tickerData = None

        while not successfulDownload and attempts < maxAttempts:
            try:
                # Download with volume data
                raw_data = yf.download(
                    ticker,
                    group_by="Ticker",
                    period=duration,
                    interval="5m",
                    prepost=False,
                    repair=True,
                )

                if not raw_data.empty:
                    # Prepare enhanced dataset with multiple features
                    tickerData = pd.DataFrame(
                        {
                            "Close": raw_data["Close"].reset_index(drop=True),
                            "Volume": raw_data["Volume"].reset_index(drop=True),
                        }
                    )

                    # Add datetime if available in index
                    if hasattr(raw_data.index, "to_pydatetime"):
                        tickerData["datetime"] = raw_data.index.to_pydatetime()

                    # Remove any NaN values
                    tickerData = tickerData.dropna()

                    if len(tickerData) > 10:  # Minimum data requirement
                        successfulDownload = True
                    else:
                        logging.warning(
                            f"Insufficient data for {ticker}: {len(tickerData)} rows"
                        )

            except Exception as e:
                logging.error(
                    f"Download attempt {attempts + 1} failed for {ticker}: {e}"
                )
                attempts += 1
                time.sleep(1)  # Brief pause between retries

        if not successfulDownload:
            logging.error(
                f"Failed to download data for {ticker} after {maxAttempts} attempts"
            )
            # Remove failed query from queue
            self.main.ALGOqueryQueue = self.main.ALGOqueryQueue.iloc[1:].reset_index(
                drop=True
            )
            return

        # Calculate prediction length (20% of data length)
        predictionLen = max(1, len(tickerData) // 5)

        # Configure radiobutton text
        text = f"{ticker} - {duration}\nProcessing..."
        self.AQradioButton = tk.CTkRadioButton(
            self.main.graphHistoryFrame,
            text=text,
            variable=self.main.AQbuttonVar,
            value=ALGOqueryID,
            command=self.main.loadALGOquery,
            state="disabled",
        )
        self.AQradioButton.grid(row=ALGOqueryID, column=0, padx=10, pady=(0, 10))

        # Process the query with enhanced algorithm
        self.queryALGO(
            ticker,
            duration,
            dataSize,
            tickerData,
            predictionLen,
            ALGOquery,
        )

    def queryALGO(
        self,
        ticker,
        duration,
        dataSize,
        tickerData,
        predictionLen,
        ALGOquery,
    ):
        def queryLocalMachine():
            try:
                # Determine algorithm configuration
                use_zscore = getattr(
                    self.main, "use_zscore", True
                )  # Default to enhanced mode
                use_volume = getattr(
                    self.main, "use_volume", True
                )  # Use volume if available

                # Check if volume data is meaningful
                if "Volume" in tickerData.columns:
                    volume_variance = tickerData["Volume"].var()
                    if volume_variance == 0 or pd.isna(volume_variance):
                        use_volume = False
                        logging.info(
                            f"Disabling volume for {ticker} due to zero variance"
                        )

                # Create enhanced algorithm instance
                if hasattr(ALGOdt3, "create_algo_instance"):
                    # Use new factory function if available
                    algoSettings = ALGOdt3.create_algo_instance(
                        int(dataSize),
                        predictionLen,
                        tickerData,
                        self.main.progressQueue,
                        use_zscore=use_zscore,
                        use_volume=use_volume,
                    )
                else:
                    # Fallback to original constructor
                    algoSettings = ALGOdt3.Algo(
                        int(dataSize),
                        predictionLen,
                        tickerData,
                        self.main.progressQueue,
                    )

                # Execute algorithm
                result = algoSettings.startPool()
                workerResultQueue.put(result)

            except Exception as e:
                logging.error(f"Error in queryLocalMachine for {ticker}: {e}")
                # Return empty DataFrame on error
                workerResultQueue.put(pd.DataFrame())

        # Initialize result storage
        averageResult = pd.DataFrame()
        workerResultQueue = queue.Queue()

        # Start algorithm execution
        localMachineThread = threading.Thread(target=queryLocalMachine)
        localMachineThread.start()
        localMachineThread.join()

        # Collect results
        try:
            result = workerResultQueue.get(timeout=5)  # 5 second timeout
            if not result.empty:
                averageResult = result
            else:
                logging.warning(f"Empty result received for {ticker}")
        except queue.Empty:
            logging.error(f"Timeout waiting for results for {ticker}")

        # Update UI and data structures
        self._updateQueryResults(ticker, duration, ALGOquery, averageResult, tickerData)

    def _updateQueryResults(
        self, ticker, duration, ALGOquery, averageResult, tickerData
    ):
        """
        Update the UI and internal data structures with query results.

        Args:
            ticker: Stock ticker symbol
            duration: Time duration of the query
            ALGOquery: Original query parameters
            averageResult: Algorithm prediction results
            tickerData: Original ticker data
        """
        try:
            # Update radiobutton to show completion
            status_text = f"{ticker} - {duration}"
            if not averageResult.empty:
                status_text += f"\n{averageResult.shape[1]} matches"
            else:
                status_text += "\nNo matches"

            self.AQradioButton.configure(text=status_text, state="normal")

            # Remove query from queue
            self.main.ALGOqueryQueue = self.main.ALGOqueryQueue.iloc[1:].reset_index(
                drop=True
            )

            # Add to history
            self.main.ALGOqueryHistory = pd.concat(
                [self.main.ALGOqueryHistory, pd.DataFrame([ALGOquery])],
                ignore_index=True,
            )

            # Manage memory - remove old entries if we have too many
            self.main.ALGOqueryButtonHistory.append(self.AQradioButton)
            if len(self.main.ALGOqueryButtonHistory) > 50:
                # Remove oldest entries
                old_button = self.main.ALGOqueryButtonHistory.pop(0)
                old_button.destroy()

                # Clean up data arrays
                if len(self.main.ALGOqueryResultHistory) > 50:
                    self.main.ALGOqueryResultHistory.pop(0)
                if len(self.main.ALGOqueryDataHistory) > 50:
                    self.main.ALGOqueryDataHistory.pop(0)

                # Clean up history DataFrame
                if len(self.main.ALGOqueryHistory) > 50:
                    self.main.ALGOqueryHistory = self.main.ALGOqueryHistory.iloc[
                        1:
                    ].reset_index(drop=True)

            # Store results for future reference
            self.main.ALGOqueryResultHistory.append(averageResult)

            # Store original close prices for plotting compatibility
            if "Close" in tickerData.columns:
                close_data = tickerData["Close"]
            else:
                close_data = tickerData.iloc[:, 0]  # First column as fallback

            self.main.ALGOqueryDataHistory.append(close_data)

            # Auto-select this query for immediate visualization
            self.AQradioButton.invoke()

        except Exception as e:
            logging.error(f"Error updating query results for {ticker}: {e}")


# Enhanced configuration options for the main application
class EnhancedAQP(AQP):
    """
    Extended version of AQP with additional configuration options.
    """

    def __init__(self, main, config=None):
        """
        Initialize enhanced AQP with configuration options.

        Args:
            main: Main application instance
            config: Configuration dictionary with algorithm options
        """
        # Set default configuration
        default_config = {
            "use_zscore": True,
            "use_volume": True,
            "min_data_points": 10,
            "max_retries": 10,
            "retry_delay": 1.0,
            "result_timeout": 30,
        }

        if config:
            default_config.update(config)

        self.config = default_config

        # Apply configuration to main app
        for key, value in self.config.items():
            setattr(main, key, value)

        # Initialize parent class
        super().__init__(main)

    def get_algorithm_status(self):
        """
        Get current algorithm configuration status.

        Returns:
            dict: Current algorithm settings
        """
        return {
            "use_zscore": getattr(self.main, "use_zscore", True),
            "use_volume": getattr(self.main, "use_volume", True),
            "active_queries": len(self.main.ALGOqueryQueue),
            "completed_queries": len(self.main.ALGOqueryHistory),
        }
