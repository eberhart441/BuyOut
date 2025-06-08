import numpy as np
import pandas as pd
import os
import time
import multiprocessing as mp
import logging
import heapq
from datetime import datetime

# Setting up logging with detailed formatting
logging.basicConfig(
    filename="algoLog.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class Algo:
    def __init__(
        self,
        dataSize,
        predictionLen,
        tickerData,
        progressQueue,
        use_zscore=True,
        use_volume=True,
    ):
        """
        Initialize the Algo class with enhanced settings for processing financial data.

        Args:
            dataSize (int): The size of the data to process.
            predictionLen (int): The length of the prediction interval.
            tickerData (DataFrame): The ticker data with datetime, close, and optionally volume.
            progressQueue (Queue): A queue for progress updates.
            use_zscore (bool): Whether to use z-score normalization instead of ratio normalization.
            use_volume (bool): Whether to include volume in pattern matching.
        """
        self.use_zscore = use_zscore
        self.use_volume = use_volume

        # Prepare the ticker data based on normalization method
        if self.use_zscore:
            self.tickerDataZScores = self._prepare_zscore_data(tickerData)
        else:
            # Legacy ratio normalization for backward compatibility
            close_prices = self._extract_close_prices(tickerData)
            self.tickerDataZScores = (close_prices / close_prices[-1]).reshape(-1, 1)

        # Extract timing information if available
        self.target_time = None
        if isinstance(tickerData, pd.DataFrame):
            if "datetime" in tickerData.columns:
                self.target_time = pd.to_datetime(tickerData["datetime"].iloc[-1])
            elif hasattr(tickerData.index, "to_pydatetime"):
                self.target_time = tickerData.index[-1]

        self.settings = {
            "dataSize": dataSize,
            "predictionLen": predictionLen,
            "tickerDataZScores": self.tickerDataZScores,
            "progressQueue": progressQueue,
        }

        self.tickerDataPath = "Resources/Ticker_Data_Multiplied"
        self.tickerDataLen = len(self.tickerDataZScores)
        self.processSplitIndex = max(1, dataSize // 16)

    def _extract_close_prices(self, tickerData):
        """Extract close prices from various data formats."""
        if isinstance(tickerData, pd.DataFrame):
            if "Close" in tickerData.columns:
                return tickerData["Close"].to_numpy()
            elif "close" in tickerData.columns:
                return tickerData["close"].to_numpy()
            else:
                return tickerData.iloc[:, 0].to_numpy()
        else:
            return (
                tickerData.to_numpy() if hasattr(tickerData, "to_numpy") else tickerData
            )

    def _prepare_zscore_data(self, tickerData):
        """
        Prepare z-score normalized data from ticker input.

        Args:
            tickerData (DataFrame): Input ticker data

        Returns:
            numpy.ndarray: Processed data array with shape (n, features)
        """
        try:
            # Extract close prices
            close_prices = self._extract_close_prices(tickerData)

            # Get volume if available and requested
            volume_data = None
            if self.use_volume and isinstance(tickerData, pd.DataFrame):
                if "Volume" in tickerData.columns:
                    volume_data = tickerData["Volume"].to_numpy()
                elif "volume" in tickerData.columns:
                    volume_data = tickerData["volume"].to_numpy()

            # Calculate z-scores for close prices
            std_close = np.std(close_prices)
            mean_close = np.mean(close_prices)

            if std_close == 0:
                close_zscores = np.zeros_like(close_prices)
            else:
                close_zscores = 4 * (close_prices - mean_close) / std_close

            # Prepare final data structure
            if volume_data is not None and self.use_volume:
                # Calculate z-scores for volume
                std_volume = np.std(volume_data)
                mean_volume = np.mean(volume_data)

                if std_volume == 0:
                    volume_zscores = np.zeros_like(volume_data)
                else:
                    volume_zscores = (volume_data - mean_volume) / std_volume

                # Create 3-column array: [datetime_placeholder, close_zscore, volume_zscore]
                # First column will be filled with time data during processing
                result = np.column_stack(
                    [
                        np.zeros_like(close_zscores),  # Placeholder for time
                        close_zscores,
                        volume_zscores,
                    ]
                )
            else:
                # Create 2-column array: [datetime_placeholder, close_zscore]
                result = np.column_stack(
                    [
                        np.zeros_like(close_zscores),  # Placeholder for time
                        close_zscores,
                    ]
                )

            return result

        except Exception as e:
            logging.error(f"Error in z-score preparation: {e}")
            # Fallback to simple normalization
            close_prices = self._extract_close_prices(tickerData)
            normalized = close_prices / close_prices[-1]
            return np.column_stack([np.zeros_like(normalized), normalized])

    def startPool(self):
        """
        Starts a multiprocessing pool to parallelize data processing with enhanced pattern matching.

        Returns:
            DataFrame: The prediction results in a DataFrame format.
        """
        numProcesses = mp.cpu_count()
        manager = mp.Manager()
        progressList = manager.list([0] * numProcesses)
        resultsList = []

        try:
            try:
                files = os.listdir(self.tickerDataPath)
            except FileNotFoundError:
                logging.error(f"Directory not found: {self.tickerDataPath}")
                return pd.DataFrame()

            with mp.Pool(processes=numProcesses) as pool:
                results = [
                    pool.apply_async(self.algo, args=(processId, progressList, files))
                    for processId in range(numProcesses)
                ]

                # Progress bar logic
                prevTotal = 0
                while any(not result.ready() for result in results):
                    currentTotal = sum(progressList)
                    if currentTotal != prevTotal:
                        self.settings["progressQueue"].put(
                            currentTotal / self.settings["dataSize"]
                        )
                    prevTotal = currentTotal
                    time.sleep(0.1)

                # Collect results from all processes
                for result in results:
                    processResult = result.get()
                    if processResult:
                        resultsList.extend(processResult)

            # Use heap to get top 100 results efficiently
            if len(resultsList) > 100:
                top100Results = heapq.nsmallest(100, resultsList, key=lambda x: x[0])
            else:
                top100Results = sorted(resultsList, key=lambda x: x[0])

            # Process predictions from top matches
            resultsList100 = []
            for _, filePath, i in top100Results:
                try:
                    # Load prediction segment
                    if self.use_zscore:
                        # Load with proper column handling
                        try:
                            tickerData = pd.read_csv(
                                filePath,
                                usecols=["close"],
                                dtype={"close": "float"},
                                skiprows=range(1, i),
                                nrows=self.tickerDataLen
                                + self.settings["predictionLen"],
                            )
                        except:
                            # Try uppercase columns
                            tickerData = pd.read_csv(
                                filePath,
                                usecols=["Close"],
                                dtype={"Close": "float"},
                                skiprows=range(1, i),
                                nrows=self.tickerDataLen
                                + self.settings["predictionLen"],
                            )

                        # Extract close prices
                        close_col = (
                            "close" if "close" in tickerData.columns else "Close"
                        )
                        tickerPrices = tickerData[close_col].to_numpy()

                        # Calculate z-scores for this segment
                        std_close = np.std(tickerPrices)
                        mean_close = np.mean(tickerPrices)

                        if std_close == 0:
                            tickerDataZScores = np.zeros_like(tickerPrices)
                        else:
                            tickerDataZScores = (
                                4 * (tickerPrices - mean_close)
                            ) / std_close

                        resultsList100.append(tickerDataZScores)
                    else:
                        # Legacy method for backward compatibility
                        tickerCache = pd.read_csv(filePath)["Close"].to_numpy()
                        segment = tickerCache[
                            i : i + self.tickerDataLen + self.settings["predictionLen"]
                        ]
                        normalized = segment / segment[self.tickerDataLen]
                        resultsList100.append(normalized)

                except Exception as e:
                    logging.error(f"Error reading file {filePath}: {e}")
                    continue

            # Convert to DataFrame
            if resultsList100:
                resultsList100 = [np.squeeze(array) for array in resultsList100]
                return pd.DataFrame(resultsList100).T
            else:
                return pd.DataFrame()

        except Exception as e:
            logging.exception("An error occurred during startPool execution: " + str(e))
            return pd.DataFrame()

    def algo(self, processId, progressList, files):
        """
        Enhanced pattern matching algorithm with z-score normalization and multi-feature support.

        Args:
            processId (int): The identifier for the process in the pool.
            progressList (Manager.list): A shared list for tracking progress.
            files (list): List of data files to process.

        Returns:
            list: The results of the algorithm for this process.
        """
        resultList = []

        # Get target time for time-based matching
        tickerDataTime = None
        if self.target_time:
            tickerDataTime = (
                self.target_time.time()
                if hasattr(self.target_time, "time")
                else self.target_time
            )

        # Calculate work distribution
        num_files = len(files)
        start_index = self.processSplitIndex * processId
        end_index = min(start_index + self.processSplitIndex, num_files)

        for x in range(end_index - start_index):
            file_index = x + start_index
            if file_index >= num_files:
                break

            filePath = os.path.join(self.tickerDataPath, files[file_index])

            try:
                if self.use_zscore:
                    # Enhanced z-score processing
                    required_cols = ["datetime", "close"]
                    if self.use_volume:
                        required_cols.append("volume")

                    try:
                        tickerCache = pd.read_csv(
                            filePath,
                            usecols=required_cols,
                            dtype={"close": "float", "volume": "float"},
                            parse_dates=["datetime"],
                        )
                    except:
                        # Try uppercase columns
                        required_cols = ["datetime", "Close"]
                        if self.use_volume:
                            required_cols.append("Volume")
                        tickerCache = pd.read_csv(
                            filePath,
                            usecols=required_cols,
                            dtype={"Close": "float", "Volume": "float"},
                            parse_dates=["datetime"],
                        )

                    # Standardize column names
                    if "Close" in tickerCache.columns:
                        tickerCache.rename(
                            columns={"Close": "close", "Volume": "volume"}, inplace=True
                        )

                    # Extract time for matching
                    tickerCache["time"] = pd.to_datetime(
                        tickerCache["datetime"]
                    ).dt.time
                    tickerCacheTimes = tickerCache["time"].to_numpy()

                    # Pattern matching loop
                    max_iterations = (
                        len(tickerCache)
                        - self.tickerDataLen
                        - int(self.settings["predictionLen"])
                    )

                    for i in range(max_iterations):
                        # Time-based filtering if available
                        if (
                            tickerDataTime
                            and tickerCacheTimes[i + self.tickerDataLen - 1]
                            != tickerDataTime
                        ):
                            continue

                        # Extract segments
                        tickerCacheSegmentClose = tickerCache.iloc[
                            i : i + self.tickerDataLen, 1
                        ].to_numpy()

                        # Calculate close z-scores
                        std_close = np.std(tickerCacheSegmentClose)
                        mean_close = np.mean(tickerCacheSegmentClose)

                        if std_close == 0:
                            closeZScores = np.zeros_like(tickerCacheSegmentClose)
                        else:
                            closeZScores = (
                                4 * (tickerCacheSegmentClose - mean_close)
                            ) / std_close

                        if self.use_volume and "volume" in tickerCache.columns:
                            # Extract and process volume
                            tickerCacheSegmentVolume = tickerCache.iloc[
                                i : i + self.tickerDataLen, 2
                            ].to_numpy()
                            std_volume = np.std(tickerCacheSegmentVolume)
                            mean_volume = np.mean(tickerCacheSegmentVolume)

                            if std_volume == 0:
                                volumeZScores = np.zeros_like(tickerCacheSegmentVolume)
                            else:
                                volumeZScores = (
                                    tickerCacheSegmentVolume - mean_volume
                                ) / std_volume

                            segmentZScores = np.column_stack(
                                [closeZScores, volumeZScores]
                            )

                            # Compare with input data (skip first column which is time placeholder)
                            if self.settings["tickerDataZScores"].shape[1] >= 3:
                                distance = np.linalg.norm(
                                    self.settings["tickerDataZScores"][:, [1, 2]]
                                    - segmentZScores
                                )
                            else:
                                distance = np.linalg.norm(
                                    self.settings["tickerDataZScores"][:, 1:]
                                    - closeZScores.reshape(-1, 1)
                                )
                        else:
                            # Only close prices
                            if self.settings["tickerDataZScores"].shape[1] >= 2:
                                distance = np.linalg.norm(
                                    self.settings["tickerDataZScores"][:, 1]
                                    - closeZScores
                                )
                            else:
                                distance = np.linalg.norm(
                                    self.settings["tickerDataZScores"][:, 0]
                                    - closeZScores
                                )

                        resultList.append((distance, filePath, i))
                else:
                    # Legacy ratio normalization processing
                    tickerCache = pd.read_csv(filePath, usecols=["Close"])
                    tickerPrices = tickerCache["Close"].to_numpy()

                    max_iterations = (
                        len(tickerPrices)
                        - self.tickerDataLen
                        - int(self.settings["predictionLen"])
                    )

                    for i in range(max_iterations):
                        segment_prices = tickerPrices[i : i + self.tickerDataLen]
                        segment_normalized = segment_prices / segment_prices[-1]

                        distance = np.linalg.norm(
                            self.settings["tickerDataZScores"][:, -1]
                            - segment_normalized
                        )

                        # Apply threshold
                        if distance < self.tickerDataLen / 50:
                            resultList.append([distance, file_index, i])

            except Exception as e:
                logging.error(f"Error processing file {filePath}: {e}")
                continue

            progressList[processId] += 1

        return resultList


# Factory function for easy instantiation
def create_algo_instance(dataSize, predictionLen, tickerData, progressQueue, **kwargs):
    """
    Factory function to create Algo instance with automatic feature detection.

    Args:
        dataSize: Size of dataset to search
        predictionLen: Length of prediction horizon
        tickerData: Input ticker data (DataFrame or Series)
        progressQueue: Progress update queue
        **kwargs: Additional options (use_zscore, use_volume, etc.)

    Returns:
        Algo: Configured algorithm instance
    """
    # Auto-detect if we should use z-scores based on data availability
    use_zscore = kwargs.get("use_zscore", True)  # Default to True for new version
    use_volume = kwargs.get("use_volume", False)  # Conservative default

    # Check if volume data is available
    if isinstance(tickerData, pd.DataFrame):
        has_volume = any(col.lower() in ["volume", "vol"] for col in tickerData.columns)
        if has_volume and "use_volume" not in kwargs:
            use_volume = True

    return Algo(
        dataSize,
        predictionLen,
        tickerData,
        progressQueue,
        use_zscore=use_zscore,
        use_volume=use_volume,
    )
